import os
import glob 
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime
import random
import string
import subprocess
import json

SUPPORTED_FORMATS = ['wav', 'flac', 'mp3', 'aac', 'ogg', 'm4a', 'aiff']


def generate_random_filename(base_name="spectrum", extension=".png"):
    """Generate a random filename with a date, timestamp, and random string."""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    return f"{base_name}_{current_time}_{random_str}{extension}"


def extract_metadata(file_path):
    """Extract metadata using ffprobe."""
    try:
        ffprobe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=bit_rate,codec_name,sample_rate,channels",
            "-of", "json", file_path
        ]
        result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        metadata = json.loads(result.stdout)

        stream = metadata.get("streams", [{}])[0]
        bit_rate = int(stream.get("bit_rate", 0)) // 1000  # Convert to kbps
        codec = stream.get("codec_name", "Unknown")
        sample_rate = int(stream.get("sample_rate", 0))
        channels = int(stream.get("channels", 1))
        return bit_rate, codec, sample_rate, channels
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return None, None, None, None


def calculate_lossless_bitrate(sample_rate, bit_depth, channels):
    """Calculate the actual bitrate for lossless formats."""
    return (sample_rate * bit_depth * channels) / 1000  # Convert to kbps


def estimate_actual_bitrate(codec, max_frequency, sample_rate=None, channels=None):
    """Estimate the actual bitrate based on codec, frequency, and audio properties."""
    if codec in ["mp3", "aac", "ogg", "m4a"]:
        if max_frequency >= 20000:
            return "320 kbps"
        elif max_frequency >= 19000:
            return "256 kbps"
        elif max_frequency >= 16000:
            return "192 kbps"
        elif max_frequency >= 15000:
            return "128 kbps"
        elif max_frequency >= 12000:
            return "<128 kbps"
        else:
            return "Very Low (<96 kbps)"
    elif codec in ["wav", "flac", "aiff", "pcm_s16le", "pcm_s24le", "pcm_s32le"]:
        bit_depth = 16 if "16le" in codec else 24 if "24le" in codec else 32
        actual_bitrate = calculate_lossless_bitrate(sample_rate, bit_depth, channels)
        return f"{actual_bitrate:.2f} kbps"
    else:
        return "Unknown Format"


def generate_spectrogram(file_path):
    """Generate a spectrogram and save it as a PNG."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr / 2)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=sr / 2, cmap='coolwarm')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()

        output_image = generate_random_filename()
        plt.savefig(output_image)
        plt.close()

        print(f"Spectrogram saved as {output_image}")
        return output_image
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        return None


def analyze_audio(file_path):
    """Analyze the audio file to determine the maximum significant frequency."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        n_fft = 2048
        hop_length = n_fft // 4
        D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann'))

        freqs = librosa.fft_frequencies(sr=sr)
        avg_magnitude = np.mean(D, axis=1)
        energy_threshold = 0.001 * np.max(avg_magnitude)
        significant_indices = np.where(avg_magnitude > energy_threshold)[0]

        if significant_indices.size > 0:
            max_frequency = freqs[significant_indices[-1]]
        else:
            max_frequency = 0.0

        return max_frequency
    except Exception as e:
        print(f"Error analyzing audio file: {e}")
        return 0.0


def is_supported_format(file_path):
    """Check if the file has a supported audio format."""
    file_extension = os.path.splitext(file_path)[-1].lower()[1:]  # Extract the extension without '.'
    return file_extension in SUPPORTED_FORMATS


def generate_html_report(results, html_filename):
    """Generate an HTML report from the analysis results."""
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


def main():
    parser = argparse.ArgumentParser(description="Analyze audio files and generate an HTML report.")
    parser.add_argument(
        "-f", "--file",
        help="Specify the output HTML file name (e.g., results.html)",
        required=True
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

    results = []

    for file_path in matching_files:
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            continue

        if not is_supported_format(file_path):
            print(f"Unsupported format: {file_path}")
            continue

        print(f"Analyzing {file_path}...")

        stated_bitrate, codec, sample_rate, channels = extract_metadata(file_path)
        max_freq = analyze_audio(file_path)
        estimated_bitrate = estimate_actual_bitrate(codec, max_freq, sample_rate, channels)
        spectrogram_file = generate_spectrogram(file_path)

        results.append({
            "file": file_path,
            "codec": codec,
            "sample_rate": sample_rate,
            "max_frequency": max_freq,
            "stated_bitrate": stated_bitrate,
            "estimated_bitrate": estimated_bitrate,
            "spectrogram": spectrogram_file
        })

    generate_html_report(results, args.file)


if __name__ == "__main__":
    main()
