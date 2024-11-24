import os
import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
from datetime import datetime
import random
import string
import subprocess
import json
from tqdm import tqdm

SUPPORTED_FORMATS = ['wav', 'flac', 'mp3', 'aac', 'ogg', 'm4a', 'aiff']

logging.basicConfig(level=logging.INFO, format='%(message)s')

def generate_random_filename(base_name="spectrum", extension=".png"):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    return f"{base_name}_{current_time}_{random_str}{extension}"

def extract_metadata(file_path):
    try:
        ffprobe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=bit_rate,codec_name,sample_rate,channels,bits_per_sample",
            "-of", "json", file_path
        ]
        result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        metadata = json.loads(result.stdout)
        stream = metadata.get("streams", [{}])[0]
        bit_rate = int(stream.get("bit_rate", 0)) // 1000
        codec = stream.get("codec_name", "Unknown")
        sample_rate = int(stream.get("sample_rate", 0))
        channels = int(stream.get("channels", 1))
        bits_per_sample = int(stream.get("bits_per_sample", 0))
        return bit_rate, codec, sample_rate, channels, bits_per_sample
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return None, None, None, None, None

def calculate_lossless_bitrate(sample_rate, bit_depth, channels):
    return (sample_rate * bit_depth * channels) / 1000

def estimate_actual_bitrate(codec, max_frequency, sample_rate=None, channels=None, bit_rate=None, bits_per_sample=None):
    if sample_rate is None or channels is None:
        return "Unknown"
    nyquist = sample_rate / 2
    if nyquist == 0:
        return "Unknown"
    frequency_ratio = max_frequency / nyquist
    if codec.lower() in ["wav", "flac", "aiff", "aif", "pcm_s16le", "pcm_s24le", "pcm_s32le"]:
        if bits_per_sample is not None and bits_per_sample > 0:
            actual_bitrate = (sample_rate * bits_per_sample * channels) / 1000
            if frequency_ratio >= 0.95:
                return f"{actual_bitrate:.2f} kbps"
            else:
                if frequency_ratio >= 0.89:
                    return "320 kbps (Re-encoded)"
                else:
                    return "Low Bitrate (Re-encoded)"
        else:
            return "Unknown"
    elif codec.lower() in ["mp3", "aac", "ogg", "m4a"]:
        if bit_rate is not None and bit_rate > 0:
            return f"{bit_rate} kbps"
        else:
            if frequency_ratio >= 0.89:
                return "320 kbps"
            elif frequency_ratio >= 0.80:
                return "256 kbps"
            elif frequency_ratio >= 0.70:
                return "192 kbps"
            elif frequency_ratio >= 0.55:
                return "128 kbps"
            else:
                return "Very Low Bitrate"
    else:
        return "Unknown Format"

def generate_spectrogram(y, sr, file_path):
    try:
        nyquist = sr / 2
        n_fft = 16384
        hop_length = n_fft // 4
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        S_dB = librosa.amplitude_to_db(S, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            S_dB, sr=sr, x_axis='time', y_axis='linear',
            hop_length=hop_length, cmap='viridis', fmax=nyquist
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram (up to {nyquist:.2f} Hz)')
        plt.tight_layout()
        output_image = generate_random_filename()
        plt.savefig(output_image)
        plt.close()
        print(f"Spectrogram saved as {output_image}")
        return output_image
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        return None

def analyze_audio(y, sr):
    try:
        n_fft = 16384
        hop_length = n_fft // 4
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        avg_spectrum = np.mean(S, axis=1)
        avg_spectrum_dB = librosa.amplitude_to_db(avg_spectrum, ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        threshold = -80
        significant_indices = np.where(avg_spectrum_dB > threshold)[0]
        if significant_indices.size > 0:
            max_frequency = freqs[significant_indices[-1]]
            return max_frequency
        else:
            return 0.0
    except Exception as e:
        print(f"Error analyzing audio data: {e}")
        return 0.0

def is_supported_format(file_path):
    file_extension = os.path.splitext(file_path)[-1].lower()[1:]
    return file_extension in SUPPORTED_FORMATS

def generate_html_report(results, html_filename):
    try:
        with open(html_filename, "w") as html_file:
            html_file.write("<!DOCTYPE html>\n<html lang='en'>\n<head>\n")
            html_file.write("<meta charset='UTF-8'>\n<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n")
            html_file.write("<title>Audio Analysis Report</title>\n")
            html_file.write("<style>\nbody { font-family: Arial, sans-serif; margin: 20px; }\n")
            html_file.write(".result { border: 1px solid #ccc; padding: 10px; margin-bottom: 20px; }\n")
            html_file.write(".result img { max-width: 100%; height: auto; }\n</style>\n</head>\n<body>\n")
            html_file.write("<h1>Audio Analysis Report</h1>\n")
            for result in results:
                html_file.write("<div class='result'>\n")
                html_file.write(f"<h2>File: {os.path.basename(result['file'])}</h2>\n")
                html_file.write(f"<p><strong>Codec:</strong> {result['codec']}</p>\n")
                html_file.write(f"<p><strong>Sample Rate:</strong> {result['sample_rate']} Hz</p>\n")
                html_file.write(f"<p><strong>Max Frequency:</strong> {result['max_frequency']:.2f} Hz</p>\n")
                html_file.write(f"<p><strong>Stated Bitrate:</strong> {result['stated_bitrate']} kbps</p>\n")
                html_file.write(f"<p><strong>Estimated Bitrate:</strong> {result['estimated_bitrate']}</p>\n")
                html_file.write(f"<img src='{result['spectrogram']}' alt='Spectrogram for {os.path.basename(result['file'])}'>\n")
                html_file.write("</div>\n")
            html_file.write("</body>\n</html>\n")
        print(f"\nHTML report successfully saved to {html_filename}")
    except Exception as e:
        print(f"Error generating HTML report: {e}")

def process_file(file_path):
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return None
    if not is_supported_format(file_path):
        print(f"Unsupported format: {file_path}")
        return None
    print(f"Analyzing {file_path}...")
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None
    stated_bitrate, codec, sample_rate, channels, bits_per_sample = extract_metadata(file_path)
    max_freq = analyze_audio(y, sr)
    estimated_bitrate = estimate_actual_bitrate(
        codec, max_freq, sample_rate, channels,
        bit_rate=stated_bitrate, bits_per_sample=bits_per_sample
    )
    frequency_ratio = max_freq / (sample_rate / 2) if sample_rate else 0
    logging.info(f"File: {file_path}")
    logging.info(f"Max Frequency: {max_freq:.2f} Hz")
    logging.info(f"Frequency Ratio: {frequency_ratio:.2f}")
    logging.info(f"Estimated Bitrate: {estimated_bitrate}")
    spectrogram_file = generate_spectrogram(y, sr, file_path)
    result = {
        "file": file_path,
        "codec": codec,
        "sample_rate": sample_rate,
        "max_frequency": max_freq,
        "stated_bitrate": stated_bitrate,
        "estimated_bitrate": estimated_bitrate,
        "spectrogram": spectrogram_file
    }
    return result

def main():
    parser = argparse.ArgumentParser(description="Analyze audio files and generate an HTML report.")
    parser.add_argument(
        "-f", "--file",
        help="Specify the output HTML file name (e.g., results.html)",
        required=True
    )
    parser.add_argument(
        "-m", "--threads",
        type=int,
        default=1,
        help="Specify the number of threads to use (default: 1)"
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Input audio file(s) or pattern(s) (e.g., '*.wav', '*.mp3')"
    )
    args = parser.parse_args()
    matching_files = []
    for pattern in args.input:
        matching_files.extend(glob.glob(pattern))
    if not matching_files:
        print("Error: No matching files found.")
        return
    if args.threads > 1:
        import multiprocessing
        pool = multiprocessing.Pool(processes=args.threads)
        results = list(tqdm(pool.imap_unordered(process_file, matching_files), total=len(matching_files)))
        pool.close()
        pool.join()
    else:
        results = []
        for file_path in matching_files:
            result = process_file(file_path)
            if result is not None:
                results.append(result)
    results = [result for result in results if result is not None]
    generate_html_report(results, args.file)

if __name__ == "__main__":
    main()
