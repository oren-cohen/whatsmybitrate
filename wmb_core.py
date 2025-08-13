import os
import sys
import json
import logging
import warnings
import subprocess
from contextlib import contextmanager
from uuid import uuid4
import numpy as np
import soundfile as sf
import librosa
import matplotlib
import scipy.signal
matplotlib.use("Agg")

warnings.filterwarnings('ignore', category=UserWarning, message='Xing stream size.*')
warnings.filterwarnings('ignore', category=UserWarning, message='PySoundFile failed.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*audioread_load.*')

MAX_LOAD_SECONDS = 300.0
SUPPORTED_FORMATS = {'wav', 'flac', 'mp3', 'aac', 'ogg', 'm4a', 'aiff', 'alac'}
logger = logging.getLogger("audio_analysis")

@contextmanager
def suppress_stderr():
    try:
        fd = sys.stderr.fileno()
    except Exception:
        yield
        return
    saved = os.dup(fd)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, fd)
        os.close(devnull)
        yield
    finally:
        try:
            os.dup2(saved, fd)
        finally:
            os.close(saved)

class AudioFile:
    ffprobe_path = "ffprobe"

    def __init__(self, file_path):
        self.path = file_path
        self.filename = os.path.basename(file_path)
        self.y = None
        self.sr = None
        self.error = None
        self.codec = None
        self.bit_rate_str = "N/A"
        self.bit_rate_numeric = None
        self.max_frequency_peak = None
        self.estimated_bitrate = None
        self.estimated_bitrate_numeric = None
        self.is_lossless = False
        self.nyquist_frequency = None
        self.peak_frequency_ratio = None
        self.spectrogram_path = None
        self.log_entries = []

    def to_dict(self):
        return {
            "file": self.path, "error": self.error, "codec": self.codec,
            "sample_rate": self.sr, "max_frequency": self.max_frequency_peak,
            "nyquist_frequency": self.nyquist_frequency,
            "peak_frequency_ratio": self.peak_frequency_ratio,
            "bit_rate": self.bit_rate_str,
            "estimated_bitrate": self.estimated_bitrate,
            "estimated_bitrate_numeric": self.estimated_bitrate_numeric,
            "is_lossless": self.is_lossless, "spectrogram_path": self.spectrogram_path,
            "log_entries": self.log_entries
        }

    def analyze(self, generate_spectrogram_flag=False, assets_dir=None):
        self.log_entries.append("INFO - Starting full analysis workflow.")
        try:
            self.log_entries.append("DEBUG - Step 1: Loading audio data.")
            with suppress_stderr():
                self._load_audio_data()

            self.log_entries.append("DEBUG - Step 2: Extracting metadata.")
            self._extract_metadata()

            self.log_entries.append("DEBUG - Step 3: Analyzing spectrum.")
            with suppress_stderr():
                self._analyze_spectrum()

            self.log_entries.append("DEBUG - Step 4: Estimating quality.")
            self._estimate_quality()

            if generate_spectrogram_flag and assets_dir and self.y is not None:
                self.log_entries.append("DEBUG - Step 5: Generating spectrogram.")
                self._generate_spectrogram_image(assets_dir)
            
            self.log_entries.append("INFO - Analysis workflow completed successfully.")

        except Exception as e:
            self.error = f"An unexpected error occurred during analysis: {e}"
            self.log_entries.append(f"FATAL - {self.error}")
        finally:
            if self.y is not None:
                del self.y
                self.y = None
                self.log_entries.append("DEBUG - Audio data cleared from memory.")

    def _load_audio_data(self):
        self.log_entries.append(f"INFO - Loading up to {MAX_LOAD_SECONDS}s of audio data.")
        try:
            self.log_entries.append("DEBUG - Attempting to load with soundfile.")
            info = sf.info(self.path)
            samplerate = info.samplerate
            stop_frame = int(samplerate * MAX_LOAD_SECONDS)
            self.y, self.sr = sf.read(self.path, always_2d=False, stop=stop_frame)
            self.log_entries.append(f"DEBUG - soundfile loaded {len(self.y)} samples at {self.sr} Hz.")
            if self.y.ndim > 1:
                self.log_entries.append(f"DEBUG - Downmixing from {self.y.ndim} channels to mono.")
                self.y = np.mean(self.y, axis=1)
            self.log_entries.append("INFO - Audio loaded successfully using soundfile.")
        except Exception as e_sf:
            self.log_entries.append(f"WARN - soundfile failed: {e_sf}. Trying librosa as fallback.")
            try:
                self.y, self.sr = librosa.load(self.path, sr=None, mono=True, duration=MAX_LOAD_SECONDS)
                self.log_entries.append(f"DEBUG - librosa loaded {len(self.y)} samples at {self.sr} Hz.")
                self.log_entries.append("INFO - Audio loaded successfully using librosa.")
            except Exception as e_lr:
                raise RuntimeError(f"All audio loading methods failed. Last error: {e_lr}")
        if self.y is None or self.sr is None or len(self.y) == 0:
            raise RuntimeError("Audio data could not be loaded or is empty.")

    def _extract_metadata(self):
        try:
            cmd = [self.ffprobe_path, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=bit_rate,codec_name,duration", "-of", "json", self.path]
            
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=60, startupinfo=startupinfo)
            stream = json.loads(result.stdout).get("streams", [{}])[0]
            
            self.codec = stream.get("codec_name", "Unknown")
            stated_bit_rate = stream.get("bit_rate")

            if stated_bit_rate and stated_bit_rate != 'N/A':
                self.bit_rate_numeric = int(stated_bit_rate) // 1000
                self.bit_rate_str = f"{self.bit_rate_numeric} kbps (Stated)"
            else:
                duration_str = stream.get("duration")
                if duration_str:
                    duration = float(duration_str)
                    file_size_bytes = os.path.getsize(self.path)
                    if duration > 0:
                        self.bit_rate_numeric = int((file_size_bytes * 8) / duration / 1000)
                        self.bit_rate_str = f"{self.bit_rate_numeric} kbps (Average)"

        except Exception as e:
            raise RuntimeError(f"Metadata extraction failed: {e}")
        
    def _analyze_spectrum(self):
        TARGET_HZ_RESOLUTION = 5.0
        RELATIVE_DB_THRESHOLD = -90.0
        FALLBACK_DB_THRESHOLD = -75.0
        GAP_HZ_THRESHOLD = 700.0

        if self.y is None or self.sr is None or self.sr == 0:
            self.max_frequency_peak = self.max_frequency_sustained = 0.0
            return

        ideal_nperseg = self.sr / TARGET_HZ_RESOLUTION
        nperseg = 1 << (int(ideal_nperseg) - 1).bit_length()

        if len(self.y) < nperseg:
            self.max_frequency_peak = self.max_frequency_sustained = 0.0
            return

        normalized_y = librosa.util.normalize(self.y)
        frequencies, psd = scipy.signal.welch(normalized_y, fs=self.sr, nperseg=nperseg, noverlap=nperseg // 2)

        if np.max(psd) <= 0:
            self.max_frequency_peak = self.max_frequency_sustained = 0.0
            return
            
        psd_dB = 10 * np.log10(psd / np.max(psd))

        significant_indices = np.where(psd_dB > RELATIVE_DB_THRESHOLD)[0]
        
        if significant_indices.size == 0:
            max_freq = 0.0
        else:
            diffs_hz = np.diff(frequencies[significant_indices])            
            gap_locations = np.where(diffs_hz > GAP_HZ_THRESHOLD)[0]
            if gap_locations.size > 0:
                end_of_signal_index = significant_indices[gap_locations[0]]
                max_freq = frequencies[end_of_signal_index]
            else:
                max_freq = frequencies[significant_indices[-1]]

        lossless_codecs = {"wav", "flac", "aiff", "alac", "pcm_s16le", "pcm_s24le", "pcm_s32le"}
        is_high_res_lossless = (self.codec and self.codec.lower() in lossless_codecs and self.sr > 48000)

        if is_high_res_lossless and max_freq > 24000:
            candidate_indices = significant_indices[frequencies[significant_indices] < 24000]
            if candidate_indices.size > 0:
                max_freq = frequencies[candidate_indices[-1]]
        
        if max_freq/(self.sr/2) > 0.99:
            significant_indices = np.where(psd_dB > FALLBACK_DB_THRESHOLD)[0]
            
            if significant_indices.size == 0:
                max_freq = 0.0
            else:
                diffs_hz = np.diff(frequencies[significant_indices])            
                gap_locations = np.where(diffs_hz > GAP_HZ_THRESHOLD)[0]
                if gap_locations.size > 0:
                    end_of_signal_index = significant_indices[gap_locations[0]]
                    max_freq = frequencies[end_of_signal_index]
                else:
                    max_freq = frequencies[significant_indices[-1]]

            lossless_codecs = {"wav", "flac", "aiff", "alac", "pcm_s16le", "pcm_s24le", "pcm_s32le"}
            is_high_res_lossless = (self.codec and self.codec.lower() in lossless_codecs and self.sr > 48000)

            if is_high_res_lossless and max_freq > 24000:
                candidate_indices = significant_indices[frequencies[significant_indices] < 24000]
                if candidate_indices.size > 0:
                    max_freq = frequencies[candidate_indices[-1]]        
    
        self.max_frequency_peak = max_freq

    def _classify_by_absolute_frequency(self, freq_hz):
        if freq_hz >= 19500:
            return "320 kbps Equivalent", 320
        elif freq_hz >= 19090:
            return "256 kbps Equivalent", 256
        elif freq_hz >= 17000:
            return "192 kbps Equivalent", 192
        else:
            return "128 kbps Equivalent or less", 128

    def _estimate_quality(self):
        if not self.sr or self.max_frequency_peak is None:
            self.estimated_bitrate = "Invalid Sample Rate or Frequency Data"
            self.estimated_bitrate_numeric = "N/A"
            return

        self.nyquist_frequency = self.sr / 2
        codec_lower = (self.codec or "").lower()
        self.peak_frequency_ratio = self.max_frequency_peak / self.nyquist_frequency

        lossless_codecs = {"wav", "flac", "aiff", "alac", "pcm_s16le", "pcm_s24le", "pcm_s32le"}
        br, num = "", 0
        context = ""
        
        if codec_lower in lossless_codecs:
            if self.peak_frequency_ratio >= 0.95:
                self.estimated_bitrate = "Lossless"
                self.estimated_bitrate_numeric = "Lossless"
                self.is_lossless = True
            else:
                context = "(Transcoded)"
                br, num = self._classify_by_absolute_frequency(self.max_frequency_peak)
        
        else:
            context = f"({self.codec.upper() if self.codec else 'Lossy'})"
            br, num = self._classify_by_absolute_frequency(self.max_frequency_peak)

        if not self.is_lossless:
            self.estimated_bitrate = f"{br} {context}".strip()
            self.estimated_bitrate_numeric = num


    def _generate_spectrogram_image(self, assets_dir):
        if self.y is None or self.sr is None:
            self.log_entries.append("WARN - Skipping spectrogram generation: audio data not available.")
            return
            
        self.log_entries.append("INFO - Generating spectrogram.")
        try:
            import matplotlib.pyplot as plt
            import librosa.display
        except ImportError:
            self.log_entries.append("ERROR - Matplotlib or librosa not found, cannot generate spectrogram.")
            return

        max_samples = int(self.sr * 180.0)
        y_plot = self.y[:max_samples]
        n_fft = 4096
        if len(y_plot) < n_fft:
            self.log_entries.append("WARN - Skipping spectrogram: not enough audio data for one FFT window.")
            return
        hop = n_fft // 4
        self.log_entries.append(f"DEBUG - Spectrogram params: n_fft={n_fft}, hop={hop}, length={len(y_plot) / float(self.sr):.2f}s")

        out_path = os.path.join(assets_dir, f"{uuid4().hex}.png")
        S = np.abs(librosa.stft(librosa.util.normalize(y_plot), n_fft=n_fft, hop_length=hop))
        S_dB = librosa.amplitude_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(12, 6))
        librosa.display.specshow(S_dB, sr=self.sr, x_axis="time", y_axis="linear", hop_length=hop, cmap="viridis", fmax=self.sr / 2, ax=ax)
        fig.colorbar(ax.collections[0], format="%+2.0f dB", ax=ax)
        ax.set_title(f"Spectrogram (first {len(y_plot) / float(self.sr):.2f}s)")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

        self.spectrogram_path = out_path
        self.log_entries.append(f"INFO - Spectrogram saved to {out_path}")