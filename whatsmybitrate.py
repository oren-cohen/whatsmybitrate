#!/usr/bin/env python3
import os
import glob
import librosa
import subprocess
import json
import matplotlib
matplotlib.use('Agg')
import numpy as np
from datetime import datetime
import random
import string
import argparse
from tqdm import tqdm
import logging
import soundfile as sf
import audioread
import sys
import warnings
from contextlib import redirect_stderr
import fnmatch
import time
from multiprocessing import Process, Manager
from abc import ABC, abstractmethod
import hashlib
import shutil

warnings.filterwarnings('ignore', category=UserWarning, message='PySoundFile failed.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*audioread_load.*')
SUPPORTED_FORMATS = ['wav', 'flac', 'mp3', 'aac', 'ogg', 'm4a', 'aiff', 'opus', 'alac']
logger = logging.getLogger("audio_analysis")


def find_ffprobe_path(user_path=None):
    if user_path:
        if shutil.which(user_path):
            return user_path
        else:
            sys.exit(f"Error: The provided ffprobe path '{user_path}' is not a valid executable.")

    executable_name = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    local_path = os.path.join(script_dir, executable_name)
    if os.path.isfile(local_path) and os.access(local_path, os.X_OK):
        return local_path
    
    path_from_which = shutil.which("ffprobe")
    if path_from_which:
        return path_from_which

    return None


class AudioFile:
    ffprobe_path = "ffprobe"

    def __init__(self, file_path):
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
        self.is_lossless = False
        self.nyquist_frequency = None
        self.spectrogram_path = None
        self.peak_frequency_ratio = None
        self.sustained_frequency_ratio = None
        self.log_entries = []

    def analyze(self, generate_spectrogram_flag=False, assets_dir=None):
        null_device = 'nul' if sys.platform == "win32" else os.devnull
        try:
            with open(null_device, 'w') as devnull, redirect_stderr(devnull):
                self._load_audio_data()
                self._extract_metadata()
                self._analyze_spectrum()
                self._estimate_quality()
                if generate_spectrogram_flag and assets_dir:
                    self._generate_spectrogram_image(assets_dir)
        except Exception as e:
            self.error = f"An unexpected error occurred during analysis: {e}"
            self.log_entries.append(f"FATAL - {self.error}")

    def to_dict(self):
        return {
            "file": self.path,
            "error": self.error,
            "codec": self.codec,
            "sample_rate": self.sr,
            "max_frequency": self.max_frequency_peak,
            "nyquist_frequency": self.nyquist_frequency,
            "peak_frequency_ratio": self.peak_frequency_ratio,
            "sustained_frequency_ratio": self.sustained_frequency_ratio,
            "bit_rate": f"{self.bit_rate} kbps" if self.bit_rate else "N/A",
            "estimated_bitrate": self.estimated_bitrate,
            "is_lossless": self.is_lossless,
            "spectrogram": self.spectrogram_path,
            "log_entries": self.log_entries
        }

    def _load_audio_data(self):
        try:
            self.y, self.sr = sf.read(self.path, always_2d=False)
            if self.y.ndim > 1: self.y = np.mean(self.y, axis=1)
        except Exception:
            try:
                with audioread.audio_open(os.path.realpath(self.path)) as f:
                    self.sr, n_channels = f.samplerate, f.channels
                    buf = b"".join(f)
                    if not buf: raise ValueError("File is empty or could not be read.")
                    self.y = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
                    if n_channels > 1: self.y = self.y.reshape((-1, n_channels)).mean(axis=1)
            except Exception:
                self.y, self.sr = librosa.load(self.path, sr=None, mono=True, duration=60.0)
        
        if self.y is None or self.sr is None or len(self.y) == 0:
            raise RuntimeError("All audio loading methods failed.")

    def _extract_metadata(self):
        try:
            cmd = [AudioFile.ffprobe_path, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=bit_rate,codec_name,sample_rate,channels,duration", "-of", "json", self.path]
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=60, startupinfo=startupinfo)
            stream = json.loads(result.stdout).get("streams", [{}])[0]
            self.codec = stream.get("codec_name", "Unknown")
            duration = float(stream.get("duration", 0)) if stream.get("duration") else None
            self.bit_rate = int(stream.get("bit_rate", 0)) // 1000 if stream.get("bit_rate") else None
            if self.bit_rate is None and duration and duration > 0:
                self.bit_rate = int((os.path.getsize(self.path) * 8) / duration / 1000)
        except Exception as e:
            raise RuntimeError(f"Metadata extraction failed: {e}")

    def _analyze_spectrum(self):
        n_fft = 4096
        if len(self.y) < n_fft or not self.sr:
            self.max_frequency_peak = 0.0
            self.max_frequency_sustained = 0.0
            return
            
        S = np.abs(librosa.stft(self.y, n_fft=n_fft, hop_length=n_fft//4))
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
            return
        
        self.nyquist_frequency = self.sr / 2
        codec_lower = (self.codec or "").lower()

        self.peak_frequency_ratio = self.max_frequency_peak / self.nyquist_frequency if self.nyquist_frequency else 0
        self.sustained_frequency_ratio = self.max_frequency_sustained / self.nyquist_frequency if self.nyquist_frequency else 0
        
        lossless_codecs = ["wav", "flac", "aiff", "pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_s16be", "alac"]
        lossy_codecs = ["mp3", "aac", "m4a", "opus", "ogg", "vorbis"]

        if self.peak_frequency_ratio >= 0.93:
            if codec_lower in lossless_codecs:
                self.estimated_bitrate = "Lossless"
                self.is_lossless = True
            else: 
                self.estimated_bitrate = f"320 kbps Equivalent ({self.codec.upper()})"
        
        elif self.peak_frequency_ratio >= 0.903:
            if codec_lower in lossless_codecs:
                self.estimated_bitrate = "320 kbps Equivalent (Transcoded)"
            else:
                self.estimated_bitrate = f"320 kbps Equivalent ({self.codec.upper()})"

        elif self.peak_frequency_ratio >= 0.85:
            if codec_lower in lossless_codecs:
                 self.estimated_bitrate = "256 kbps Equivalent (Transcoded)"
            else:
                 self.estimated_bitrate = f"256 kbps Equivalent ({self.codec.upper()})"

        else:
            br_classification = self._classify_quality_by_sustained_frequency(self.max_frequency_sustained)
            
            if codec_lower in lossless_codecs:
                self.estimated_bitrate = f"{br_classification} (Transcoded)"
            elif codec_lower in lossy_codecs:
                self.estimated_bitrate = f"{br_classification} ({self.codec.upper()})"
            else:
                self.estimated_bitrate = "Unknown Format"

        self.log_entries.append(f"INFO - Nyquist Frequency: {self.nyquist_frequency:.2f} Hz")
        self.log_entries.append(f"INFO - Peak Frequency: {self.max_frequency_peak:.2f} Hz")
        self.log_entries.append(f"INFO - Sustained Frequency: {self.max_frequency_sustained:.2f} Hz")
        self.log_entries.append(f"INFO - Peak Ratio: {self.peak_frequency_ratio:.3f}")
        self.log_entries.append(f"INFO - Sustained Ratio: {self.sustained_frequency_ratio:.3f}")
        self.log_entries.append(f"INFO - Final Quality Estimation: {self.estimated_bitrate}")

    def _generate_spectrogram_image(self, assets_dir):
        import matplotlib.pyplot as plt
        import librosa.display
        n_fft = 16384
        if len(self.y) < n_fft: return
        S = np.abs(librosa.stft(self.y, n_fft=n_fft, hop_length=n_fft//4))
        S_dB = librosa.amplitude_to_db(S, ref=np.max)
        
        safe_basename = "".join(c for c in self.filename if c.isalnum() or c in (' ','.','_')).rstrip()
        path_hash = hashlib.md5(os.path.abspath(self.path).encode('utf-8')).hexdigest()[:8]
        unique_filename = f"{os.path.splitext(safe_basename)[0]}_{path_hash}_spectrogram.png"
        self.spectrogram_path = os.path.join(assets_dir, unique_filename)
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(S_dB, sr=self.sr, x_axis="time", y_axis="linear", hop_length=n_fft//4, cmap="viridis", fmax=self.sr/2)
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Spectrogram for {self.filename}")
        plt.tight_layout()
        plt.savefig(self.spectrogram_path)
        plt.close()


class ReportGenerator(ABC):
    def __init__(self, results_data):
        self.results = sorted(results_data, key=lambda r: os.path.basename(r.get('file', '')))
    
    @abstractmethod
    def generate(self, output_path=None):
        pass

class HtmlReportGenerator(ReportGenerator):
    def generate(self, output_path):
        report_dir = os.path.dirname(os.path.abspath(output_path))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><title>Audio Analysis Report</title><style>"
                    "body{font-family:Arial,sans-serif;margin:20px;line-height:1.6} .result{border:1px solid #ccc;padding:10px;margin-bottom:20px}"
                    ".result img{max-width:100%;height:auto;display:block;margin:10px 0} h1,h2{color:#333}</style></head><body>"
                    "<h1>Audio Analysis Report</h1>")
            for r in self.results:
                f.write("<div class='result'>")
                f.write(f"<h2>File: {os.path.basename(r.get('file', 'Unknown'))}</h2>")
                if r.get('error'):
                    f.write(f"<p><strong>ERROR:</strong> {r['error']}</p>")
                else:
                    max_freq_val = r.get('max_frequency', 0.0)
                    f.write(f"<p><strong>Codec:</strong> {r.get('codec', 'N/A')}</p>")
                    f.write(f"<p><strong>Sample Rate:</strong> {r.get('sample_rate', 'N/A')} Hz</p>")
                    f.write(f"<p><strong>Max Frequency:</strong> {max_freq_val:.2f} Hz</p>")
                    f.write(f"<p><strong>Stated Bit Rate:</strong> {r.get('bit_rate', 'N/A')}</p>")
                    f.write(f"<p><strong>Estimated Quality:</strong> {r.get('estimated_bitrate', 'N/A')}</p>")
                    f.write(f"<p><strong>Is True Lossless:</strong> {'Yes' if r.get('is_lossless') else 'No'}</p>")
                    if r.get('spectrogram'):
                        relative_image_path = os.path.relpath(r['spectrogram'], report_dir)
                        relative_image_path = relative_image_path.replace('\\', '/')
                        f.write(f"<img src='{relative_image_path}' alt='Spectrogram'>")
                f.write("</div>")
            f.write("</body></html>")

class CsvReportGenerator(ReportGenerator):
    def generate(self, output_path):
        import csv
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["File", "Codec", "Sample Rate (Hz)", "Max Frequency (Hz)", "Stated Bit Rate", "Estimated Quality", "Is True Lossless", "Error"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                stated_bit_rate_val = r.get("bit_rate", "N/A")
                if isinstance(stated_bit_rate_val, str):
                    stated_bit_rate_val = stated_bit_rate_val.replace(" kbps", "").strip()

                if r.get('error'):
                    row = {"File": os.path.basename(r.get("file", "Unknown")), "Error": r["error"]}
                    for field in fieldnames:
                        if field not in row:
                            row[field] = "N/A"
                else:
                    row = {
                        "File": os.path.basename(r.get("file", "Unknown")),
                        "Codec": r.get("codec", "N/A"),
                        "Sample Rate (Hz)": r.get("sample_rate", "N/A"),
                        "Max Frequency (Hz)": f"{r.get('max_frequency', 0.0):.2f}",
                        "Stated Bit Rate": stated_bit_rate_val,
                        "Estimated Quality": r.get("estimated_bitrate", "N/A"),
                        "Is True Lossless": "Yes" if r.get("is_lossless") else "No",
                        "Error": "no"
                    }
                writer.writerow(row)

class ConsoleReportGenerator(ReportGenerator):
     def generate(self, output_path=None):
        print("\n--- Analysis Summary ---")
        for r in self.results:
            if r.get('error'):
                print(f"File: {os.path.basename(r['file'])}\n  ERROR: {r['error']}")
            else:
                message = (f"File: {os.path.basename(r['file'])}\n"
                           f"  Codec: {r.get('codec', 'N/A')}\n"
                           f"  Sample Rate: {r.get('sample_rate', 'N/A')} Hz\n"
                           f"  Max Frequency: {r.get('max_frequency', 0.0):.2f} Hz\n")
                if r.get('peak_frequency_ratio') is not None:
                    message += f"  Peak Frequency Ratio: {r['peak_frequency_ratio']:.3f}\n"
                if r.get('sustained_frequency_ratio') is not None:
                    message += f"  Sustained Frequency Ratio: {r['sustained_frequency_ratio']:.3f}\n"
                message += f"  Estimated Quality: {r.get('estimated_bitrate', 'N/A')}\n"
                print(message)

def worker_process_file(file_path, return_dict, generate_spectrogram, assets_dir):
    audio_file = AudioFile(file_path)
    audio_file.analyze(generate_spectrogram_flag=generate_spectrogram, assets_dir=assets_dir)
    return_dict[os.getpid()] = audio_file.to_dict()

class AnalysisRunner:
    def __init__(self, config, log_filename=None, assets_dir=None):
        self.config = config
        self.files_to_process = []
        self.log_filename = log_filename
        self.assets_dir = assets_dir
        self._setup_logger()

    def _setup_logger(self):
        global logger
        logger.handlers.clear()

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

        if self.log_filename:
            file_handler = logging.FileHandler(self.log_filename, mode="w")
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(processName)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            print(f"Logging is enabled. Log file: {self.log_filename}")
        else:
            print("Logging is disabled.")
            logger.addHandler(logging.NullHandler())

    def _scan_files(self):
        paths = self.config.input
        if len(paths) == 1 and os.path.isfile(os.path.expanduser(paths[0])):
            self.files_to_process.append(os.path.expanduser(paths[0]))
            return
            
        directory_to_scan = os.getcwd()
        file_patterns = paths
        if paths and os.path.isdir(os.path.expanduser(paths[0])):
            directory_to_scan = os.path.expanduser(paths[0])
            file_patterns = paths[1:] if len(paths) > 1 else None

        search_patterns = []
        if self.config.type:
            search_patterns.append(f"*.{self.config.type.lower()}")
        elif not file_patterns or self.config.all:
            search_patterns.extend([f"*.{f}" for f in SUPPORTED_FORMATS])
        else:
            search_patterns.extend(file_patterns)

        file_set = set()
        if self.config.recursive:
            excluded = {'.git', '.svn', '.cache', 'node_modules', '__pycache__'}
            for root, dirs, files in os.walk(directory_to_scan, topdown=True):
                dirs[:] = [d for d in dirs if d not in excluded and not d.startswith('.')]
                for pattern in search_patterns:
                    for filename in fnmatch.filter(files, pattern):
                        ext = os.path.splitext(filename)[1].lower().strip('.')
                        if ext in SUPPORTED_FORMATS:
                            file_set.add(os.path.join(root, filename))
        else:
            for pattern in search_patterns:
                matched_files = glob.glob(os.path.join(directory_to_scan, pattern))
                for f in matched_files:
                    ext = os.path.splitext(f)[1].lower().strip('.')
                    if ext in SUPPORTED_FORMATS:
                        file_set.add(f)
        
        self.files_to_process = list(file_set)

    def run(self):
        self._scan_files()
        if not self.files_to_process:
            print("No matching files found to process.")
            return []

        print(f"Found {len(self.files_to_process)} file(s) to analyze.")
        
        use_mp = self.config.multiprocessing and len(self.files_to_process) > 1
        if use_mp:
            return self._run_in_parallel()
        else:
            return self._run_serially()
    
    def _run_serially(self):
        print("Running in single-threaded mode. A hanging file may freeze the script.")
        results_data = []
        gen_spec = not self.config.no_spectrogram and not self.config.csv
        for file_path in tqdm(self.files_to_process, desc="Processing files (1 thread)"):
            audio_file = AudioFile(file_path)
            audio_file.analyze(generate_spectrogram_flag=gen_spec, assets_dir=self.assets_dir)
            results_data.append(audio_file.to_dict())
        return results_data

    def _run_in_parallel(self):
        try:
            num_threads = os.cpu_count()
        except NotImplementedError:
            num_threads = 4
        print(f"Multiprocessing enabled. Using {num_threads} available cores.")

        TIMEOUT_PER_FILE = 90.0
        manager = Manager()
        return_dict = manager.dict()
        active_processes = {}
        results_data = []
        files_iterator = iter(self.files_to_process)
        gen_spec = not self.config.no_spectrogram and not self.config.csv

        with tqdm(total=len(self.files_to_process), desc=f"Processing files ({num_threads} threads)") as pbar:
            while len(results_data) < len(self.files_to_process):
                finished_pids = []
                for pid, (p, file_path, start_time) in active_processes.items():
                    if not p.is_alive():
                        if pid in return_dict:
                            results_data.append(return_dict[pid])
                            del return_dict[pid]
                        else:
                            results_data.append({"file": file_path, "error": "Worker died unexpectedly."})
                        finished_pids.append(pid)
                        pbar.update(1)
                    elif time.time() - start_time > TIMEOUT_PER_FILE:
                        p.terminate()
                        time.sleep(0.1)
                        p.join()
                        results_data.append({"file": file_path, "error": f"Processing timed out after {int(TIMEOUT_PER_FILE)}s."})
                        finished_pids.append(pid)
                        pbar.update(1)

                for pid in finished_pids:
                    if pid in active_processes:
                        del active_processes[pid]

                while len(active_processes) < num_threads:
                    try:
                        next_file = next(files_iterator)
                        p = Process(target=worker_process_file, args=(next_file, return_dict, gen_spec, self.assets_dir))
                        p.start()
                        active_processes[p.pid] = (p, next_file, time.time())
                    except StopIteration:
                        break
                
                if not active_processes and len(results_data) == len(self.files_to_process):
                    break
                    
                time.sleep(0.2)
        return results_data

    def write_logs(self, results_data):
        if not self.config.log or not hasattr(self, 'log_filename'):
            return
        
        print("Writing log file...")
        with open(self.log_filename, 'a', encoding="utf-8") as f:
            for result in results_data:
                f.write(f"\n{'='*20} Processing File: {os.path.basename(result.get('file', ''))} {'='*20}\n")
                if result.get("log_entries"):
                    timestamp_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                    for entry in result["log_entries"]:
                         f.write(f"{timestamp_now} - {entry}\n")
        print("Log file written.")


def main():
    if sys.platform.startswith('darwin') or sys.platform.startswith('win32'):
        try:
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    parser = argparse.ArgumentParser(prog="whatsmybitrate.py", description="Analyzes audio files.", formatter_class=argparse.RawTextHelpFormatter)
    io_group = parser.add_argument_group('Input & Output Arguments')
    io_group.add_argument("input", nargs="*", help="The target for analysis: one or more files, a directory, or a shell glob pattern.")
    io_group.add_argument("-c", "--csv", action='store_true', help="Output the report in CSV format instead of the default HTML.")
    
    scan_group = parser.add_argument_group('File Scanning & Filtering Arguments')
    type_group = scan_group.add_mutually_exclusive_group()
    type_group.add_argument("-t", "--type", help="Scan a directory for a single file TYPE (e.g., 'mp3', 'flac').")
    type_group.add_argument("-a", "--all", action="store_true", help="Scan for all supported audio file types.")
    scan_group.add_argument("-r", "--recursive", action="store_true", help="Scan directories recursively.")

    util_group = parser.add_argument_group('Performance & Utility Arguments')
    util_group.add_argument("-m", "--multiprocessing", action='store_true', help="Enable multiprocessing using all available CPU cores.")
    util_group.add_argument("-n", "--no-spectrogram", action="store_true", help="Disable spectrogram generation in HTML reports.")
    util_group.add_argument("-l", "--log", action="store_true", help="Enable verbose logging to a uniquely named log file.")
    util_group.add_argument("--ffprobe-path", help="Specify the full path to the ffprobe executable.")
    
    args = parser.parse_args()

    ffprobe_executable_path = find_ffprobe_path(args.ffprobe_path)
    if not ffprobe_executable_path:
        sys.exit(
            "ERROR: ffprobe executable not found.\n"
            "Please add ffprobe to your system's PATH, or place ffprobe.exe in the same\n"
            "directory as this script, or specify its location using the --ffprobe-path argument."
        )
    AudioFile.ffprobe_path = ffprobe_executable_path
    
    if not args.input:
        parser.print_help(sys.stderr)
        sys.exit("\nError: No input file, directory, or pattern specified.")

    base_name = None
    if len(args.input) == 1 and os.path.isfile(os.path.expanduser(args.input[0])):
        base_name = os.path.splitext(os.path.basename(args.input[0]))[0] + '_report'
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        base_name = f"whatsmybitrate_report_{timestamp}_{random_str}"
    
    output_dir = base_name
    os.makedirs(output_dir, exist_ok=True)
    
    output_format = 'csv' if args.csv else 'html'
    report_path = os.path.join(output_dir, f"{os.path.basename(base_name)}.{output_format}")
    print(f"Report will be saved to: {report_path}")
    
    log_path = None
    if args.log:
        log_path = os.path.join(output_dir, f"{os.path.basename(base_name)}.log")
    
    assets_path = None
    if not args.no_spectrogram and not args.csv:
        assets_path = os.path.join(output_dir, 'assets')
        os.makedirs(assets_path, exist_ok=True)

    runner = AnalysisRunner(config=args, log_filename=log_path, assets_dir=assets_path)
    results_data = runner.run()
    if not results_data:
        return
        
    if args.csv:
        reporter = CsvReportGenerator(results_data)
    else:
        reporter = HtmlReportGenerator(results_data)
    
    reporter.generate(report_path)
    
    console_reporter = ConsoleReportGenerator(results_data)
    console_reporter.generate()
    runner.write_logs(results_data)
    
    final_message = f"\nAnalysis complete. All outputs saved in directory: {output_dir}"
    print(final_message)

if __name__ == "__main__":
    main()