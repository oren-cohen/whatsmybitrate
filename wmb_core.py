import os
import sys
import uuid
import json
import logging
import warnings
import subprocess
import gc
from contextlib import contextmanager

import numpy as np
import soundfile as sf
import librosa
import matplotlib
import scipy.signal
matplotlib.use("Agg")

warnings.filterwarnings('ignore', category=UserWarning, message='Xing stream size.*')
warnings.filterwarnings('ignore', category=UserWarning, message='PySoundFile failed.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*audioread_load.*')

MAX_LOAD_SECONDS = None
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
    ffmpeg_path = "ffmpeg"

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
        self.is_brickwall_fake = False
        self.estimated_bitrate = None
        self.estimated_bitrate_numeric = None
        self.is_lossless = False
        self.spectrogram_path = None
        self.log_entries = []

    def to_dict(self):
        return {
            "file": self.path,
            "error": self.error,
            "codec": self.codec,
            "sample_rate": self.sr,
            "max_frequency": self.max_frequency_peak,
            "bit_rate": self.bit_rate_str,
            "estimated_bitrate": self.estimated_bitrate,
            "estimated_bitrate_numeric": self.estimated_bitrate_numeric,
            "is_lossless": self.is_lossless,
            "spectrogram_path": self.spectrogram_path,
            "log_entries": self.log_entries,
        }

    def _is_lossless_codec(self):
        if not self.codec:
            return False
        c = self.codec.lower()
        LOSSLESS_CODECS = {"flac", "alac", "wavpack", "ape", "tak", "tta", "mlp", "truehd", "dsd_lsbf", "dsd_msbf"}
        return c in LOSSLESS_CODECS or c.startswith("pcm_")

    def analyze(self, generate_spectrogram_flag=False, assets_dir=None):
        self.log_entries.append("INFO - Starting analysis workflow.")
        try:
            with suppress_stderr():
                self._load_audio_data()
            self._extract_metadata()
            self._analyze_spectrum()

            freq = self.max_frequency_peak or 0.0
            if freq <= 15500:
                detected_bracket = 96
            elif freq <= 16500:
                detected_bracket = 128
            elif freq <= 18500:
                detected_bracket = 192
            elif freq <= 19600:
                detected_bracket = 256
            else:
                detected_bracket = 320

            self.log_entries.append(
                f"DEBUG - Spectrum result: max_freq={self.max_frequency_peak:.0f} Hz, "
                f"is_brickwall_fake={self.is_brickwall_fake}, detected_bracket={detected_bracket}kbps"
            )

            if self._is_lossless_codec():
                if self.is_brickwall_fake:
                    self.log_entries.append(
                        f"DEBUG - Lossless container with brickwall at {self.max_frequency_peak:.0f} Hz. Flagging as transcode."
                    )
                    self.estimated_bitrate = f"~{detected_bracket}kbps (Transcoded to Lossless)"
                    self.estimated_bitrate_numeric = detected_bracket
                    self.is_lossless = False
                else:
                    self.log_entries.append("DEBUG - Lossless container, no definitive brickwall. Genuine lossless.")
                    self.estimated_bitrate = "Lossless"
                    self.estimated_bitrate_numeric = "Lossless"
                    self.is_lossless = True
            else:
                self.log_entries.append("DEBUG - Explicitly lossy codec.")
                
                def _snap_to_bracket(kbps):
                    brackets = [96, 128, 160, 192, 224, 256, 320]
                    return min(brackets, key=lambda x: abs(x - kbps))

                stated_snapped = _snap_to_bracket(self.bit_rate_numeric) if self.bit_rate_numeric else None
                VBR_CODECS = {'vorbis', 'opus', 'aac', 'mp4a', 'he-aac', 'eac3'}
                is_vbr_codec = self.codec and self.codec.lower() in VBR_CODECS

                margin = 70 if is_vbr_codec else 20
                if self.is_brickwall_fake and stated_snapped and stated_snapped > (detected_bracket + margin):
                                                                                                    
                    self.log_entries.append(
                        f"DEBUG - Fake upscale: metadata says {self.bit_rate_numeric}kbps, spectrum brickwall says ~{detected_bracket}kbps."
                    )
                    self.estimated_bitrate = f"~{detected_bracket}kbps (Re-encoded)"
                    self.estimated_bitrate_numeric = detected_bracket
                else:
                    if stated_snapped:
                        if not is_vbr_codec and not self.is_brickwall_fake and detected_bracket > stated_snapped:
                                                                                                              
                            reported_kbps = detected_bracket
                        else:
                            reported_kbps = stated_snapped
                            
                        self.estimated_bitrate = f"LOSSY ~{reported_kbps}kbps"
                        self.estimated_bitrate_numeric = reported_kbps
                    else:
                        self.estimated_bitrate = f"LOSSY ~{detected_bracket}kbps"
                        self.estimated_bitrate_numeric = detected_bracket
                self.is_lossless = False

            if generate_spectrogram_flag and assets_dir and self.y is not None:
                self._generate_spectrogram_image(assets_dir)

            self.log_entries.append("INFO - Analysis workflow completed successfully.")

        except Exception as e:
            self.error = f"An unexpected error occurred during analysis: {e}"
            self.log_entries.append(f"FATAL - {self.error}")
        finally:
            if self.y is not None:
                del self.y
                self.y = None
            gc.collect()
            
    def _load_audio_data(self):
        try:
            self.y, self.sr = librosa.load(self.path, sr=None, mono=True)
        except Exception:
            import tempfile
            temp_wav = os.path.join(tempfile.gettempdir(), f"load_{uuid.uuid4().hex}.wav")
            try:
                subprocess.run(
                    [self.ffmpeg_path, "-y", "-i", self.path,
                     "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", temp_wav],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                )
                self.y, self.sr = sf.read(temp_wav)
            except Exception as e_ff:
                raise RuntimeError(f"All audio loading methods failed. FFmpeg fallback error: {e_ff}")
            finally:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

    def _extract_metadata(self):
        try:
                                                                               
            cmd = [
                self.ffprobe_path, "-v", "error", "-select_streams", "a:0",
                "-show_entries",
                "stream=bit_rate,codec_name,duration:stream_tags=BPS,BPS-eng:format=bit_rate,duration",
                "-of", "json", self.path
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, check=True, timeout=60
            )
            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]
            fmt = data.get("format", {})
            tags = stream.get("tags", {})

            self.codec = stream.get("codec_name", "Unknown")

            bps_tag = tags.get("BPS") or tags.get("BPS-eng")
            stream_br = stream.get("bit_rate")
            fmt_br = fmt.get("bit_rate")
            duration_str = stream.get("duration") or fmt.get("duration")

            if bps_tag and bps_tag != 'N/A':
                self.bit_rate_numeric = int(bps_tag) // 1000
                self.bit_rate_str = f"{self.bit_rate_numeric} kbps (Stated)"
            elif stream_br and stream_br != 'N/A':
                self.bit_rate_numeric = int(stream_br) // 1000
                self.bit_rate_str = f"{self.bit_rate_numeric} kbps (Stated)"
            elif fmt_br and fmt_br != 'N/A':
                self.bit_rate_numeric = int(fmt_br) // 1000
                self.bit_rate_str = f"{self.bit_rate_numeric} kbps (Stated)"
            elif duration_str:
                duration = float(duration_str)
                file_size_bytes = os.path.getsize(self.path)
                if duration > 0:
                    self.bit_rate_numeric = int((file_size_bytes * 8) / duration / 1000)
                    self.bit_rate_str = f"{self.bit_rate_numeric} kbps (Estimated)"
        except Exception as e:
            raise RuntimeError(f"Metadata extraction failed: {e}")

    def _analyze_spectrum(self):
        TARGET_HZ_RESOLUTION = 5.0
        RELATIVE_DB_THRESHOLD = -90.0
        FALLBACK_DB_THRESHOLD = -75.0
        GAP_HZ_THRESHOLD = 700.0

        if self.y is None or self.sr is None or self.sr == 0:
            self.max_frequency_peak = 0.0
            self.is_brickwall_fake = True
            return

        ideal_nperseg = self.sr / TARGET_HZ_RESOLUTION
        nperseg = 1 << (int(ideal_nperseg) - 1).bit_length()

        if len(self.y) < nperseg:
            self.max_frequency_peak = 0.0
            self.is_brickwall_fake = True
            return

        normalized_y = librosa.util.normalize(self.y)
        bh_window = scipy.signal.windows.blackmanharris(nperseg)
        S = np.abs(librosa.stft(normalized_y, n_fft=nperseg, hop_length=nperseg // 2, window=bh_window))
        S_power = S ** 2
        psd = np.median(S_power, axis=1)
        frequencies = librosa.fft_frequencies(sr=self.sr, n_fft=nperseg)
        if np.max(psd) <= 0:
            self.max_frequency_peak = 0.0
            self.is_brickwall_fake = True
            return
            
        psd_dB = 10 * np.log10(psd / np.max(psd))

        def _psd_at(hz):
            return float(psd_dB[np.argmin(np.abs(frequencies - hz))])

        self.log_entries.append(
            f"DEBUG - STFT: n_fft={nperseg}, sr={self.sr}, frames={S.shape[1]}, "
            f"freq_bins={S.shape[0]}, freq_res={frequencies[1]-frequencies[0]:.2f}Hz"
        )
        self.log_entries.append(
            f"DEBUG - PSD profile: "
            f"1kHz={_psd_at(1000):.1f}dB  5kHz={_psd_at(5000):.1f}dB  "
            f"10kHz={_psd_at(10000):.1f}dB  14kHz={_psd_at(14000):.1f}dB  "
            f"16kHz={_psd_at(16000):.1f}dB  18kHz={_psd_at(18000):.1f}dB  "
            f"19kHz={_psd_at(19000):.1f}dB  20kHz={_psd_at(20000):.1f}dB  "
            f"21kHz={_psd_at(21000):.1f}dB"
        )

        significant_indices = np.where(psd_dB > RELATIVE_DB_THRESHOLD)[0]
        
        if significant_indices.size == 0:
            max_freq = 0.0
            self.log_entries.append("DEBUG - Gap detector (primary): no significant bins above -90dB")
        else:
            diffs_hz = np.diff(frequencies[significant_indices])            
            gap_locations = np.where(diffs_hz > GAP_HZ_THRESHOLD)[0]
            if gap_locations.size > 0:
                end_of_signal_index = significant_indices[gap_locations[0]]
                max_freq = float(frequencies[end_of_signal_index])
                gap_size = float(diffs_hz[gap_locations[0]])
                self.log_entries.append(
                    f"DEBUG - Gap detector (primary -90dB): gap of {gap_size:.0f}Hz found → max_freq={max_freq:.0f}Hz"
                )
            else:
                max_freq = float(frequencies[significant_indices[-1]])
                self.log_entries.append(
                    f"DEBUG - Gap detector (primary -90dB): no gap > {GAP_HZ_THRESHOLD}Hz → max_freq={max_freq:.0f}Hz (last sig bin)"
                )

        is_high_res_lossless = (self._is_lossless_codec() and self.sr > 48000)

        if is_high_res_lossless and max_freq > 24000:
            candidate_indices = significant_indices[frequencies[significant_indices] < 24000]
            if candidate_indices.size > 0:
                max_freq = float(frequencies[candidate_indices[-1]])
                self.log_entries.append(f"DEBUG - High-res lossless cap: max_freq clamped to {max_freq:.0f}Hz")
        
        if max_freq >= 21000 or max_freq / (self.sr / 2) > 0.99:
            self.log_entries.append(
                f"DEBUG - Gap detector (fallback -75dB): primary gave {max_freq:.0f}Hz (≥21kHz or near-Nyquist), re-running..."
            )
            significant_indices = np.where(psd_dB > FALLBACK_DB_THRESHOLD)[0]
            
            if significant_indices.size == 0:
                max_freq = 0.0
                self.log_entries.append("DEBUG - Gap detector (fallback): no significant bins above -75dB")
            else:
                diffs_hz = np.diff(frequencies[significant_indices])            
                gap_locations = np.where(diffs_hz > GAP_HZ_THRESHOLD)[0]
                if gap_locations.size > 0:
                    end_of_signal_index = significant_indices[gap_locations[0]]
                    max_freq = float(frequencies[end_of_signal_index])
                    gap_size = float(diffs_hz[gap_locations[0]])
                    self.log_entries.append(
                        f"DEBUG - Gap detector (fallback -75dB): gap of {gap_size:.0f}Hz found → max_freq={max_freq:.0f}Hz"
                    )
                else:
                    max_freq = float(frequencies[significant_indices[-1]])
                    self.log_entries.append(
                        f"DEBUG - Gap detector (fallback -75dB): no gap → max_freq={max_freq:.0f}Hz (last sig bin)"
                    )

            if is_high_res_lossless and max_freq > 24000:
                candidate_indices = significant_indices[frequencies[significant_indices] < 24000]
                if candidate_indices.size > 0:
                    max_freq = float(frequencies[candidate_indices[-1]])
                    self.log_entries.append(f"DEBUG - High-res lossless cap (fallback): max_freq clamped to {max_freq:.0f}Hz")
    
        self.max_frequency_peak = max_freq
        
        is_vbr_for_scan = self.codec and self.codec.lower() in {'vorbis', 'opus', 'aac', 'mp4a', 'he-aac'}
        if is_vbr_for_scan:
            scan_trigger = 15000.0
            scan_bottom = 10000.0
        elif self._is_lossless_codec():
            scan_trigger = 14000.0
            scan_bottom = 13000.0
        else:
            scan_trigger = 21000.0
            scan_bottom = 15000.0
        scan_top = min(self.max_frequency_peak, 21000.0)

        self.log_entries.append(
            f"DEBUG - Dirty scan: trigger={scan_trigger:.0f}Hz, max_freq={self.max_frequency_peak:.0f}Hz, "
            f"will_run={self.max_frequency_peak >= scan_trigger}, "
            f"scan_range={scan_top:.0f}Hz→{scan_bottom:.0f}Hz, codec={self.codec}"
        )

        if self.max_frequency_peak >= scan_trigger:
                                                 
            idx_top = np.argmin(np.abs(frequencies - scan_top))
            idx_bottom = np.argmin(np.abs(frequencies - scan_bottom))
            step_1500_hz = np.argmin(np.abs(frequencies - 1500.0))
            
            best_cliff_freq = 0.0
            cliff_threshold = 18.0 if self._is_lossless_codec() else 15.0
            silence_margin = 12.0 if is_vbr_for_scan else 8.0
            candidates_found = 0

            for i in range(idx_top, idx_bottom, -1):
                idx_end = i
                idx_start = max(0, i - step_1500_hz)
                
                drop_db = psd_dB[idx_start] - psd_dB[idx_end]
                
                if drop_db > cliff_threshold:
                    signal_level = psd_dB[idx_start]
                    idx_scan_ceil = np.argmin(np.abs(frequencies - 22000.0))
                    
                    p90_above = np.percentile(psd_dB[idx_end:idx_scan_ceil+1], 90) if idx_end < idx_scan_ceil else -100.0
                    
                    passes_silence = p90_above < signal_level - silence_margin
                    candidates_found += 1
                        
                    if passes_silence:
                        best_cliff_freq = float(frequencies[idx_end])
                        if not is_vbr_for_scan:
                            break
                         
            if best_cliff_freq > 0.0:
                self.log_entries.append(f"DEBUG - Dirty scan: brickwall confirmed at {best_cliff_freq:.0f}Hz")
                self.max_frequency_peak = best_cliff_freq
                self.is_brickwall_fake = True
            else:
                self.is_brickwall_fake = False
        else:
                                                                
            idx_max = np.argmin(np.abs(frequencies - self.max_frequency_peak))
            
            target_freq = self.max_frequency_peak - 1500.0
            if target_freq < 0:
                target_freq = 0
            idx_1500_before = np.argmin(np.abs(frequencies - target_freq))
            
            drop_db = psd_dB[idx_1500_before] - psd_dB[idx_max]
            
            if self._is_lossless_codec():
                if drop_db >= 12.0:
                    self.is_brickwall_fake = True
                else:
                    self.is_brickwall_fake = False
            else:
                if drop_db < 15.0 and self.max_frequency_peak > 18500:
                    self.is_brickwall_fake = False
                else:
                    self.is_brickwall_fake = True

        if self._is_lossless_codec() and not self.is_brickwall_fake:
            chunk_seconds = 15.0
            frames_per_sec = self.sr / (nperseg / 2)
            frames_per_chunk = int(chunk_seconds * frames_per_sec)
            num_frames = S_power.shape[1]
            
            if frames_per_chunk > 0 and num_frames > frames_per_chunk:
                self.log_entries.append(f"DEBUG - Global scan missed. Running chunked analysis ({frames_per_chunk} frames/chunk)...")
                
                cliff_frequencies = []
                
                for start_frame in range(0, num_frames, frames_per_chunk):
                    end_frame = min(start_frame + frames_per_chunk, num_frames)
                    if end_frame - start_frame < frames_per_chunk * 0.5:
                        continue                            
                        
                    chunk_power = S_power[:, start_frame:end_frame]
                    chunk_psd = np.median(chunk_power, axis=1)
                    
                    if np.max(chunk_psd) <= 0:
                        continue
                        
                    chunk_dB = 10 * np.log10(chunk_psd / np.max(chunk_psd))
                    
                    significant_indices = np.where(chunk_dB > -75.0)[0]
                    if significant_indices.size == 0:
                        continue
                        
                    diffs_hz = np.diff(frequencies[significant_indices])
                    gap_locations = np.where(diffs_hz > 700.0)[0]
                    
                    if gap_locations.size > 0:
                        end_idx = significant_indices[gap_locations[0]]
                        chunk_max_freq = float(frequencies[end_idx])
                    else:
                        chunk_max_freq = float(frequencies[significant_indices[-1]])
                        
                    if 10000.0 < chunk_max_freq < 20500.0:
                        cliff_frequencies.append(chunk_max_freq)
                        
                if cliff_frequencies:
                                                                      
                    cliff_clusters = {}
                    for freq in cliff_frequencies:
                        found_cluster = False
                        for cluster_center in cliff_clusters.keys():
                            if abs(freq - cluster_center) < 300.0:
                                cliff_clusters[cluster_center].append(freq)
                                found_cluster = True
                                break
                        if not found_cluster:
                            cliff_clusters[freq] = [freq]
                            
                    total_chunks_analyzed = num_frames // frames_per_chunk
                    min_required_chunks = max(2, min(3, total_chunks_analyzed // 4))
                    
                    best_cluster_center = None
                    max_votes = 0
                    
                    for center, freqs in cliff_clusters.items():
                        if len(freqs) >= min_required_chunks and len(freqs) > max_votes:
                            max_votes = len(freqs)
                            best_cluster_center = center
                            
                    if best_cluster_center is not None:
                        avg_cliff = sum(cliff_clusters[best_cluster_center]) / len(cliff_clusters[best_cluster_center])
                        self.log_entries.append(
                            f"DEBUG - Chunked scan: Stable brickwall found at {avg_cliff:.0f}Hz "
                            f"(appeared in {max_votes} chunks, min required {min_required_chunks})."
                        )
                        self.max_frequency_peak = avg_cliff
                        self.is_brickwall_fake = True
                    else:
                        self.log_entries.append(
                            f"DEBUG - Chunked scan: Found isolated cliffs at {[int(f) for f in cliff_frequencies]}Hz "
                            f"but none were stable across {min_required_chunks} chunks. Ignored."
                        )
            
    def _generate_spectrogram_image(self, assets_dir):
        try:
            import matplotlib.pyplot as plt
            import librosa.display
        except ImportError:
            return

        n_fft = 4096
        if len(self.y) < n_fft:
            return
        hop = n_fft // 4

        out_path = os.path.join(assets_dir, f"{uuid.uuid4().hex}.png")
        S = np.abs(librosa.stft(librosa.util.normalize(self.y), n_fft=n_fft, hop_length=hop))
        S_dB = librosa.amplitude_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(18, 8))
        librosa.display.specshow(
            S_dB, sr=self.sr, x_axis="time", y_axis="linear",
            hop_length=hop, cmap="viridis", fmax=self.sr / 2, ax=ax
        )
        fig.colorbar(ax.collections[0], format="%+2.0f dB", ax=ax)
        ax.set_title("Spectrogram (Entire Track)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.clf()
        plt.close('all')
        self.spectrogram_path = out_path
