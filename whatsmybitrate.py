# whatsmybitrate.py
# v0.1.0 CLI wrapper

import os
import glob
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
import sys
import warnings
from abc import ABC, abstractmethod
import fnmatch
from multiprocessing import Pool
import shutil
from functools import partial

# 🔹 Import core audio analysis class and formats
from wmb_core import AudioFile, SUPPORTED_FORMATS

warnings.filterwarnings('ignore', category=UserWarning, message='Xing stream size.*')
warnings.filterwarnings('ignore', category=UserWarning, message='PySoundFile failed.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*audioread_load.*')

logger = logging.getLogger("audio_analysis")
MAX_LOAD_SECONDS = 300.0


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
            fieldnames = [
                "File", "Codec", "Sample Rate (Hz)",
                "Peak Frequency (Hz)", "Sustained Frequency (Hz)",
                "Peak Ratio", "Sustained Ratio",
                "Stated Bit Rate", "Estimated MP3 Equivalent", "Lossless", "Error"
            ]
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
                        "Peak Frequency (Hz)": f"{r.get('max_frequency', 0.0):.2f}",
                        "Sustained Frequency (Hz)": f"{r.get('sustained_frequency', 0.0):.2f}",
                        "Peak Ratio": f"{r.get('peak_frequency_ratio', 0.0):.3f}",
                        "Sustained Ratio": f"{r.get('sustained_frequency_ratio', 0.0):.3f}",
                        "Stated Bit Rate": stated_bit_rate_val,
                        "Estimated MP3 Equivalent": r.get("estimated_bitrate_numeric", "N/A"),
                        "Lossless": "Yes" if r.get("is_lossless") else "No",
                        "Error": "None"
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


def worker_process_file(file_path, generate_spectrogram, assets_dir):
    audio_file = AudioFile(file_path)
    audio_file.analyze(generate_spectrogram_flag=generate_spectrogram, assets_dir=assets_dir)
    return audio_file.to_dict()


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

        if self.config.multiprocessing and len(self.files_to_process) > 1:
            return self._run_in_parallel()
        else:
            return self._run_serially()

    def _run_serially(self):
        print("Running in single-threaded mode.")
        results_data = []
        gen_spec = not self.config.no_spectrogram and not self.config.csv
        worker_func = partial(worker_process_file, generate_spectrogram=gen_spec, assets_dir=self.assets_dir)
        for file_path in tqdm(self.files_to_process, desc="Processing files (1 thread)"):
            results_data.append(worker_func(file_path))
        return results_data

    def _run_in_parallel(self):
        num_workers = self.config.workers or os.cpu_count()
        print(f"Multiprocessing enabled. Using {num_workers} worker processes.")

        gen_spec = not self.config.no_spectrogram and not self.config.csv

        worker_func = partial(worker_process_file, generate_spectrogram=gen_spec, assets_dir=self.assets_dir)
        results_data = []

        with Pool(processes=num_workers, maxtasksperchild=1) as pool:
            with tqdm(total=len(self.files_to_process), desc=f"Processing files ({num_workers} threads)") as pbar:
                for result in pool.imap_unordered(worker_func, self.files_to_process):
                    results_data.append(result)
                    pbar.update()
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
    util_group.add_argument("-w", "--workers", type=int, help="Set the number of worker processes. Defaults to the number of CPU cores.")
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
