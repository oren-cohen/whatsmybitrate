import os
import glob
import sys
import argparse
from PIL import Image
import subprocess

def analyze_audio(file_path, output_file=None):
    """Analyze a single audio file and determine its quality."""
    result_lines = []
    result_lines.append(f"\nAnalyzing: {file_path}")

    # Extract metadata using ffprobe
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,bit_rate,sample_rate",
        "-of", "json",
        file_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        error_message = f"Error analyzing file {file_path}: {result.stderr}"
        result_lines.append(error_message)
        output_results(result_lines, output_file)
        return

    metadata = eval(result.stdout)  # Quick and dirty JSON parsing
    if not metadata or "streams" not in metadata or not metadata["streams"]:
        result_lines.append(f"Error: No audio stream found in {file_path}")
        output_results(result_lines, output_file)
        return

    stream = metadata["streams"][0]
    codec = stream.get("codec_name", "Unknown")
    bit_rate = int(stream.get("bit_rate", 0))
    sample_rate = int(stream.get("sample_rate", 0))

    # Generate spectrum image
    spectrum_image = f"{file_path}.png"
    spectrum_cmd = [
        "ffmpeg",
        "-i", file_path,
        "-lavfi", "showspectrumpic=s=1024x512",
        spectrum_image,
        "-y",
    ]
    subprocess.run(spectrum_cmd)

    # Analyze spectrum image
    try:
        with Image.open(spectrum_image) as img:
            max_frequency = analyze_spectrum(img, sample_rate)
    except Exception as e:
        result_lines.append(f"Error analyzing spectrum for {file_path}: {e}")
        output_results(result_lines, output_file)
        return

    # Determine quality
    quality = determine_quality(codec, bit_rate, max_frequency, sample_rate)
    result_lines.append(f"Audio Quality Analysis for {file_path}:")
    result_lines.append(f"Codec: {codec}")
    result_lines.append(f"Bitrate: {bit_rate // 1000} kbps")
    result_lines.append(f"Sample Rate: {sample_rate} Hz")
    result_lines.append(f"Maximum Frequency: {max_frequency} Hz")
    result_lines.append(f"Determined Quality: {quality}")
    output_results(result_lines, output_file)

def analyze_spectrum(image, sample_rate):
    """Analyze a spectrum image to determine the maximum frequency."""
    height = image.height
    threshold = 10
    for y in range(height - 1, -1, -1):
        row = [
            sum(image.getpixel((x, y))) if isinstance(image.getpixel((x, y)), tuple) else image.getpixel((x, y))
            for x in range(image.width)
        ]
        if max(row) > threshold:
            nyquist = sample_rate // 2
            frequency = nyquist * (height - y) // height
            return frequency
    return 0

def determine_quality(codec, bit_rate, max_frequency, sample_rate):
    """Determine the quality of an audio file."""
    if codec in ("mp3", "aac", "mp4a"):
        if bit_rate >= 320000 and max_frequency >= 20000:
            return "High-quality (320 kbps)"
        elif bit_rate >= 256000 and max_frequency >= 19000:
            return "Good-quality (256 kbps)"
        elif bit_rate >= 192000 and max_frequency >= 16000:
            return "Standard quality (192 kbps)"
        elif bit_rate >= 128000 and max_frequency >= 15000:
            return "Low-quality (128 kbps)"
        elif bit_rate < 128000 and max_frequency < 15000:
            return "Very low-quality (<128 kbps)"
        else:
            return f"Detected Quality: {bit_rate // 1000} kbps"
    elif codec in ("flac",):
        return "Lossless FLAC"
    elif codec in ("pcm_s16le", "pcm_s24le", "pcm_s32le"):
        # Uncompressed WAV/AIFF
        if max_frequency >= 20000 and sample_rate >= 44100:
            return "Uncompressed WAV/AIFF (CD-quality, 16-bit)"
        elif max_frequency >= 16000:
            return "Recompressed WAV/AIFF (Original 192 kbps)"
        elif max_frequency >= 15000:
            return "Recompressed WAV/AIFF (Original 128 kbps)"
        else:
            return "Recompressed WAV/AIFF (Original <128 kbps)"
    else:
        return f"Detected Quality: {bit_rate // 1000} kbps"

def output_results(result_lines, output_file):
    """Output results to console or file."""
    output_text = "\n".join(result_lines)
    if output_file:
        with open(output_file, "a") as f:
            f.write(output_text + "\n")
    else:
        print(output_text)

def process_directory(directory, output_file=None):
    """Process all files in a directory."""
    audio_files = glob.glob(os.path.join(directory, "*"))
    if not audio_files:
        print(f"No files found in directory {directory}.")
        return

    for audio_file in audio_files:
        if os.path.isfile(audio_file):
            analyze_audio(audio_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze audio files to determine quality.")
    parser.add_argument("target", nargs="+", help="Audio file(s) or directory to analyze.")
    parser.add_argument("-f", "--file", help="Output results to specified file.", dest="output_file")
    args = parser.parse_args()

    targets = args.target
    output_file = args.output_file

    for target in targets:
        if os.path.isdir(target):
            # Analyze all files in the specified directory
            process_directory(target, output_file)
        elif os.path.isfile(target):
            # Analyze a single file
            analyze_audio(target, output_file)
        else:
            print(f"Invalid input: {target}")
