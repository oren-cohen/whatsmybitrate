import os
import glob
import librosa
import subprocess
import json
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from datetime import datetime
import random
import string
import argparse
import multiprocessing
from tqdm import tqdm

SUPPORTED_FORMATS = ['wav', 'flac', 'mp3', 'aac', 'ogg', 'm4a', 'aiff']


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
        bits_per_sample = stream.get("bits_per_sample", 0)
        bits_per_sample = int(bits_per_sample) if bits_per_sample is not None else 0
        return bit_rate, codec, sample_rate, channels, bits_per_sample
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return None, None, None, None, None



def estimate_actual_bitrate(codec, max_frequency, sample_rate=None, channels=None, bit_rate=None, bits_per_sample=None):
    nyquist = sample_rate / 2 if sample_rate else 1
    frequency_ratio = max_frequency / nyquist
    
    print(f"Max Frequency: {max_frequency}, Nyquist: {nyquist}, Frequency Ratio: {frequency_ratio:.2f}")
    
    # If the codec is lossless, adjust estimation accordingly
    if codec.lower() in ["wav", "flac", "aiff", "pcm_s16le", "pcm_s24le", "pcm_s32le"]:
        if frequency_ratio >= 0.95:
            return f"Lossless (Uncompressed)"
        elif frequency_ratio >= 0.75:
            return "320 kbps (Re-encoded)"
        elif frequency_ratio >= 0.65:
            return "256 kbps (Re-encoded)"
        else:
            return "Low-Quality Re-encoded Lossless"
    elif codec.lower() in ["mp3", "aac", "ogg", "m4a"]:
        if frequency_ratio >= 0.84:
            return "320 kbps (High Quality)"
        elif frequency_ratio >= 0.78:
            return "256 kbps (Good Quality)"
        elif frequency_ratio >= 0.70:
            return "128 kbps (Low Quality)"
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
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='linear', hop_length=hop_length, cmap='viridis', fmax=nyquist)
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
        n_fft = 32768  # High-resolution FFT for better frequency resolution
        hop_length = n_fft // 4
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))  # Compute STFT
        
        # Calculate the average spectrum across time frames
        avg_spectrum = np.mean(S, axis=1)
        avg_spectrum_dB = librosa.amplitude_to_db(avg_spectrum, ref=np.max)
        
        # Use a lower threshold for more sensitive frequency detection
        threshold_dB = np.percentile(avg_spectrum_dB, 5)  # Lower threshold for sensitivity
        
        # Get the corresponding frequencies for each bin in the FFT
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Filter frequencies where their average dB is above the threshold
        significant_indices = np.where(avg_spectrum_dB > threshold_dB)[0]
        
        if significant_indices.size == 0:
            return 0.0
        
        # Maximum frequency is the last significant frequency (highest energy frequency)
        max_frequency = freqs[significant_indices[-1]]
        
        # Ensure max_frequency is below Nyquist and doesn't falsely report Nyquist frequency
        nyquist_frequency = sr / 2
        if max_frequency >= nyquist_frequency * 0.95:
            max_frequency = freqs[significant_indices[-2]] if len(significant_indices) > 1 else max_frequency
        
        return max_frequency
    except Exception as e:
        print(f"Error analyzing audio data: {e}")
        return 0.0


def generate_html_report(results, html_filename):
    try:
        with open(html_filename, "w") as html_file:
            html_file.write("<!DOCTYPE html>\n<html lang='en'>\n<head>\n")
            html_file.write("<meta charset='UTF-8'>\n<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n")
            html_file.write("<title>Audio Analysis Report</title>\n")
            html_file.write("<style>\n")
            html_file.write("body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }\n")
            html_file.write(".result { border: 1px solid #ccc; padding: 10px; margin-bottom: 20px; }\n")
            html_file.write(".result img { max-width: 100%; height: auto; display: block; margin: 10px 0; }\n")
            html_file.write("</style>\n</head>\n<body>\n")
            html_file.write("<h1>Audio Analysis Report</h1>\n")

            for result in results:
                html_file.write("<div class='result'>\n")
                html_file.write(f"<h2>File: {os.path.basename(result['file'])}</h2>\n")
                html_file.write(f"<p><strong>Codec:</strong> {result['codec']}</p>\n")
                html_file.write(f"<p><strong>Sample Rate:</strong> {result['sample_rate']} Hz</p>\n")
                html_file.write(f"<p><strong>Max Frequency:</strong> {result['max_frequency']:.2f} Hz</p>\n")
                html_file.write(f"<p><strong>Frequency Ratio:</strong> {result['frequency_ratio']:.2f}</p>\n")
                html_file.write(f"<p><strong>Stated Bitrate:</strong> {result['stated_bitrate']} kbps</p>\n")
                html_file.write(f"<p><strong>Estimated Bitrate:</strong> {result['estimated_bitrate']}</p>\n")
                if result['spectrogram']:
                    html_file.write(f"<img src='{result['spectrogram']}' alt='Spectrogram for {os.path.basename(result['file'])}'>\n")
                html_file.write("</div>\n")

            html_file.write("</body>\n</html>\n")
        print(f"HTML report successfully saved to {html_filename}")
    except Exception as e:
        print(f"Error generating HTML report: {e}")


def process_file(file_path):
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return None
    if not file_path.lower().endswith(tuple(SUPPORTED_FORMATS)):
        print(f"Unsupported format: {file_path}")
        return None
    print(f"Processing {file_path}...")
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

    stated_bitrate, codec, sample_rate, channels, bits_per_sample = extract_metadata(file_path)
    max_freq = analyze_audio(y, sr)
    frequency_ratio = max_freq / (sample_rate / 2) if sample_rate else 0
    estimated_bitrate = estimate_actual_bitrate(codec, max_freq, sample_rate, channels, bit_rate=stated_bitrate, bits_per_sample=bits_per_sample)
    spectrogram_file = generate_spectrogram(y, sr, file_path)
    return {
        "file": file_path,
        "codec": codec,
        "sample_rate": sample_rate,
        "max_frequency": max_freq,
        "frequency_ratio": frequency_ratio,
        "stated_bitrate": stated_bitrate,
        "estimated_bitrate": estimated_bitrate,
        "spectrogram": spectrogram_file
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze audio files and generate an HTML report.")
    parser.add_argument("-f", "--file", help="Output HTML file name", required=True)
    parser.add_argument("input", nargs="+", help="Input audio file(s) or patterns")
    parser.add_argument("-m", "--threads", type=int, default=1, help="Number of threads to use (default: 1)")
    args = parser.parse_args()

    matching_files = []
    for pattern in args.input:
        matching_files.extend(glob.glob(pattern))
    if not matching_files:
        print("No matching files found.")
        return

    results = []
    if args.threads > 1:
        print(f"Processing files using {args.threads} threads...")
        with multiprocessing.Pool(processes=args.threads) as pool:
            results = list(tqdm(pool.imap(process_file, matching_files), total=len(matching_files)))
    else:
        print("Processing files sequentially...")
        for file_path in tqdm(matching_files, desc="Processing files"):
            result = process_file(file_path)
            if result:
                results.append(result)

    results = [result for result in results if result]

    if not results:
        print("No valid results to generate a report.")
        return

    generate_html_report(results, args.file)


if __name__ == "__main__":
    main()