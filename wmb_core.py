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
matplotlib.use("Agg")

warnings.filterwarnings('ignore', category=UserWarning, message='Xing stream size.*')
warnings.filterwarnings('ignore', category=UserWarning, message='PySoundFile failed.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*audioread_load.*')

MAX_LOAD_SECONDS = 300.0
SUPPORTED_FORMATS = {'wav','flac','mp3','aac','ogg','m4a','aiff','opus','alac'}
logger = logging.getLogger(__name__)  # independent logger; inherits root handlers


@contextmanager
def suppress_stderr():
    """Safely suppress stderr; no-op if fileno/dup fails."""
    try:
        original_stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(original_stderr_fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, original_stderr_fd)
        os.close(devnull_fd)
        try:
            yield
        finally:
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.close(saved_stderr_fd)
    except Exception:
        # Fallback: do nothing if we can't manipulate fds (e.g., environment restrictions)
        yield


class AudioFile:
    ffprobe_path = "ffprobe"

    def __init__(self, file_path: str):
        self.path = file_path
        self.filename = os.path.basename(file_path)
        self.y = None
        self.sr = None
        self.error = None
        self.codec = None
        self.bit_rate = None
        self.max_frequency_peak = None
        self.max_frequency_sustained = None
        self.estimated_bitrate = None
        self.estimated_bitrate_numeric = None
        self.is_lossless = False
        self.nyquist_frequency = None
        self.spectrogram_path = None
        self.peak_frequency_ratio = None
        self.sustained_frequency_ratio = None
        self.log_entries = []

    def analyze(self, generate_spectrogram_flag=False, assets_dir=None):
        try:
            with suppress_stderr():
                self._load_audio_data()

            self._extract_metadata()

            with suppress_stderr():
                self._analyze_spectrum()

            self._estimate_quality()

            if generate_spectrogram_flag and assets_dir:
                self._generate_spectrogram_image(assets_dir)
        except Exception as e:
            self.error = f"An unexpected error occurred during analysis: {e}"
            logger.exception(self.error)
            self.log_entries.append(f"FATAL - {self.error}")
        finally:
            # free memory
            self.y = None

    def to_dict(self):
        return {
            "file": self.path,
            "error": self.error,
            "codec": self.codec,
            "sample_rate": self.sr,
            "max_frequency": self.max_frequency_peak,
            "sustained_frequency": self.max_frequency_sustained,
            "nyquist_frequency": self.nyquist_frequency,
            "peak_frequency_ratio": self.peak_frequency_ratio,
            "sustained_frequency_ratio": self.sustained_frequency_ratio,
            "bit_rate": f"{self.bit_rate} kbps" if self.bit_rate else "N/A",
            "estimated_bitrate": self.estimated_bitrate,
            "estimated_bitrate_numeric": self.estimated_bitrate_numeric,
            "is_lossless": self.is_lossless,
            "spectrogram": self.spectrogram_path,
            "log_entries": self.log_entries
        }

    def _load_audio_data(self):
        try:
            info = sf.info(self.path)
            stop_frame = int(info.samplerate * MAX_LOAD_SECONDS)
            self.y, self.sr = sf.read(self.path, always_2d=False, stop=stop_frame)
            if hasattr(self.y, "ndim") and self.y.ndim > 1:
                self.y = np.mean(self.y, axis=1)
        except Exception:
            self.y, self.sr = librosa.load(self.path, sr=None, mono=True, duration=MAX_LOAD_SECONDS)

        if self.y is None or self.sr is None or len(self.y) == 0:
            raise RuntimeError("Audio data could not be loaded.")

        # ✅ Hard cap regardless of loader behavior
        max_samples = int(self.sr * MAX_LOAD_SECONDS)
        if len(self.y) > max_samples:
            self.y = self.y[:max_samples]

        # Helpful for debugging & titles
        self.loaded_seconds = len(self.y) / float(self.sr)
        logger.debug(f"Loaded {self.filename}: sr={self.sr}Hz, samples={len(self.y)}, "
                    f"seconds_kept={self.loaded_seconds:.3f} / cap={MAX_LOAD_SECONDS}s")

    def _extract_metadata(self):
        try:
            cmd = [
                AudioFile.ffprobe_path, "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=bit_rate,codec_name,sample_rate,channels,duration",
                "-of", "json", self.path
            ]
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, check=True, timeout=60, startupinfo=startupinfo
            )
            stream = json.loads(result.stdout).get("streams", [{}])[0]
            self.codec = stream.get("codec_name", "Unknown")
            duration = float(stream.get("duration", 0)) if stream.get("duration") else None
            self.bit_rate = int(stream.get("bit_rate", 0)) // 1000 if stream.get("bit_rate") else None
            if self.bit_rate is None and duration and duration > 0:
                self.bit_rate = int((os.path.getsize(self.path) * 8) / duration / 1000)
            logger.debug(f"Metadata: codec={self.codec}, bit_rate={self.bit_rate} kbps")
        except Exception as e:
            raise RuntimeError(f"Metadata extraction failed: {e}")

    def _analyze_spectrum(self):
        n_fft = 4096
        if self.y is None or len(self.y) < n_fft or not self.sr:
            self.max_frequency_peak = 0.0
            self.max_frequency_sustained = 0.0
            return

        normalized_y = librosa.util.normalize(self.y)
        S = np.abs(librosa.stft(normalized_y, n_fft=n_fft, hop_length=n_fft // 4))

        frequencies = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        S_dB = librosa.amplitude_to_db(S, ref=np.max)

        intensity_per_peak = np.percentile(S_dB, 99, axis=1)
        significant_indices_peak = np.where(intensity_per_peak > -60)[0]
        self.max_frequency_peak = frequencies[significant_indices_peak[-1]] if significant_indices_peak.size > 0 else 0.0

        intensity_sustained = np.median(S_dB, axis=1)
        significant_indices_sustained = np.where(intensity_sustained > -78)[0]
        self.max_frequency_sustained = frequencies[significant_indices_sustained[-1]] if significant_indices_sustained.size > 0 else 0.0

    def _classify_quality_by_sustained_frequency(self, freq_hz):
        if freq_hz >= 17530:
            return "320 kbps Equivalent"
        elif freq_hz >= 17088:
            return "256 kbps Equivalent"
        elif freq_hz >= 16537:
            return "192 kbps Equivalent"
        elif freq_hz >= 13450:
            return "128 kbps Equivalent"
        else:
            return "Less than 128 kbps Equivalent"

    def _estimate_quality(self):
        if not self.sr or self.sr == 0:
            self.estimated_bitrate = "Invalid Sample Rate"
            self.estimated_bitrate_numeric = "N/A"
            return

        self.nyquist_frequency = self.sr / 2
        codec_lower = (self.codec or "").lower()

        self.peak_frequency_ratio = (self.max_frequency_peak or 0) / self.nyquist_frequency
        self.sustained_frequency_ratio = (self.max_frequency_sustained or 0) / self.nyquist_frequency

        lossless_codecs = {"wav","flac","aiff","pcm_s16le","pcm_s24le","pcm_s32le","pcm_s16be","pcm_f32le","alac"}
        lossy_codecs = {"mp3","aac","m4a","opus","ogg","vorbis"}

        br = ""
        if codec_lower in lossless_codecs:
            context = "(Transcoded)"
            if self.peak_frequency_ratio >= 0.915:
                self.estimated_bitrate = "Lossless"
                self.estimated_bitrate_numeric = "Lossless"
                self.is_lossless = True
            else:
                if self.peak_frequency_ratio >= 0.8935:
                    br, num = "320 kbps Equivalent", 320
                elif self.peak_frequency_ratio >= 0.86:
                    br, num = "256 kbps Equivalent", 256
                else:
                    br = self._classify_quality_by_sustained_frequency(self.max_frequency_sustained)
                    num = 320 if "320" in br else 256 if "256" in br else 192 if "192" in br else 128 if "128" in br else "<128"
                self.estimated_bitrate = f"{br} {context}"
                self.estimated_bitrate_numeric = num
        elif codec_lower in lossy_codecs:
            context = f"({self.codec.upper()})"
            if self.peak_frequency_ratio >= 0.883:
                br, num = "320 kbps Equivalent", 320
            elif self.peak_frequency_ratio >= 0.84:
                if self.sustained_frequency_ratio >= 0.76:
                    br, num = "256 kbps Equivalent", 256
                else:
                    br, num = "192 kbps Equivalent", 192
            else:
                br = self._classify_quality_by_sustained_frequency(self.max_frequency_sustained)
                num = 320 if "320" in br else 256 if "256" in br else 192 if "192" in br else 128 if "128" in br else "<128"
            self.estimated_bitrate = f"{br} {context}"
            self.estimated_bitrate_numeric = num
        else:
            self.estimated_bitrate = "Unknown Format"
            self.estimated_bitrate_numeric = "Unknown"

    def _generate_spectrogram_image(self, assets_dir):
        import matplotlib.pyplot as plt
        import librosa.display
        if self.y is None or self.sr is None:
            return

        max_samples = int(self.sr * MAX_LOAD_SECONDS)
        y_plot = self.y[:max_samples]

        n_fft = 4096
        if len(y_plot) < n_fft:
            return
        hop = n_fft // 4

        from uuid import uuid4
        out_path = os.path.join(assets_dir, f"{uuid4().hex}.png")

        S = np.abs(librosa.stft(librosa.util.normalize(y_plot), n_fft=n_fft, hop_length=hop))
        S_dB = librosa.amplitude_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(12, 6))
        librosa.display.specshow(
            S_dB, sr=self.sr, x_axis="time", y_axis="linear",
            hop_length=hop, cmap="viridis", fmax=self.sr / 2, ax=ax
        )
        fig.colorbar(ax.collections[0], format="%+2.0f dB", ax=ax)

        secs = len(y_plot) / float(self.sr)
        ax.set_title(f"Spectrogram (first {secs:.2f}s)")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

        self.spectrogram_path = out_path
        logger.debug(f"Spectrogram saved: {out_path} (secs={secs:.3f}, cap={MAX_LOAD_SECONDS}s)")
