import os
import sys
import json
import subprocess
from PIL import Image


def analyze_audio_metadata(file_path):
    """Analyze audio metadata using ffprobe."""
    try:
        ffprobe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,bit_rate,bit_rate_mode",
            "-of", "json", file_path
        ]
        result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        metadata = json.loads(result.stdout)

        stream = metadata.get("streams", [{}])[0]
        codec = stream.get("codec_name", "Unknown")
        sample_rate = int(stream.get("sample_rate", 0))
        bit_rate = int(stream.get("bit_rate", 0)) // 1000  # Convert to kbps
        bit_rate_mode = stream.get("bit_rate_mode", "Unknown")  # CBR or VBR

        return {
            "bit_rate": bit_rate,
            "bit_rate_mode": bit_rate_mode,
            "sample_rate": sample_rate,
            "codec": codec
        }
    except Exception as e:
        print(f"Error analyzing metadata: {e}")
        return None


def generate_spectrum_image(file_path, spectrum_image="spectrum.png"):
    """Generate a spectrum image using ffmpeg."""
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", file_path,
                "-lavfi", "showspectrumpic=s=1024x512",
                spectrum_image,
                "-y"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return spectrum_image
    except Exception as e:
        print(f"Error generating spectrum image: {e}")
        return None


def analyze_spectrum_image(spectrum_image, sample_rate):
    """Analyze the spectrum image to estimate frequency cutoff."""
    try:
        image = Image.open(spectrum_image)
        width, height = image.size

        threshold = 5  # Lower intensity threshold to detect subtle frequencies
        max_frequency_bin = height  # Start from the bottom row

        # Debug: Collect intensity data for troubleshooting
        debug_data = []

        for y in range(height - 1, -1, -1):  # Iterate from bottom (highest frequency)
            row = [image.getpixel((x, y))[0] for x in range(width)]
            max_intensity = max(row)
            debug_data.append((y, max_intensity))

            if max_intensity > threshold:
                max_frequency_bin = y
                break

        # Debug: Print detected intensities (row index and max intensity)
        print("\nDebug: Spectrum Intensity Analysis")
        for y, intensity in debug_data:
            print(f"Row {y}: Max Intensity = {intensity}")

        nyquist_frequency = sample_rate / 2
        frequency_cutoff = nyquist_frequency * (height - max_frequency_bin) / height
        return int(frequency_cutoff)
    except Exception as e:
        print(f"Error analyzing spectrum image: {e}")
        return None


def determine_quality(metadata, freq_cutoff):
    """Determine the audio quality based on metadata, frequency cutoff, and bitrate mode."""
    bit_rate = metadata["bit_rate"]
    bit_rate_mode = metadata["bit_rate_mode"]
    sample_rate = metadata["sample_rate"]
    codec = metadata["codec"]

    is_vbr = bit_rate_mode.lower() == "vbr"

    if freq_cutoff is None:
        return "Unknown quality (failed frequency analysis)"

    # MP3 and similar lossy formats
    if codec in ["mp3", "aac", "mp4a"]:
        if bit_rate >= 320 and freq_cutoff < 16000:
            return "Recompressed (Original 128 kbps)"
        elif bit_rate >= 320 and freq_cutoff < 19000:
            return "Recompressed (Original 192 kbps)"
        elif bit_rate >= 320 and freq_cutoff < 20000:
            return "Recompressed (Original 256 kbps)"
        elif bit_rate >= 320:
            return "High-quality (320 kbps)"
        elif bit_rate >= 256 and freq_cutoff < 19000:
            return "Recompressed (Original 192 kbps)"
        elif bit_rate >= 256:
            return "Good-quality (256 kbps)"
        elif bit_rate >= 192:
            return "Standard quality (192 kbps)"
        elif bit_rate >= 128:
            return "Low-quality (128 kbps)"
        else:
            return "Very low-quality (below 128 kbps)"

    # WAV and AIFF (uncompressed formats)
    elif codec in ["pcm_s16le", "pcm_s16be", "pcm_f32le", "pcm_s32le"]:
        if freq_cutoff < 16000:
            return "Recompressed WAV/AIFF (Original 128 kbps)"
        elif freq_cutoff < 19000:
            return "Recompressed WAV/AIFF (Original 192 kbps)"
        elif freq_cutoff < 20000:
            return "Recompressed WAV/AIFF (Original 256 kbps)"
        elif bit_rate >= 1411:
            return "Uncompressed WAV/AIFF (CD-quality, 16-bit)"
        elif bit_rate >= 2304:
            return "High-resolution WAV/AIFF (24-bit, 48 kHz or higher)"
        else:
            return "Unknown WAV/AIFF quality"

    # FLAC (lossless format)
    elif codec == "flac":
        if freq_cutoff < 16000:
            return "Recompressed FLAC (Original 128 kbps)"
        elif freq_cutoff < 19000:
            return "Recompressed FLAC (Original 192 kbps)"
        elif freq_cutoff < 20000:
            return "Recompressed FLAC (Original 256 kbps)"
        else:
            return "Lossless FLAC"

    # Default case
    else:
        return "Unknown or custom quality"


def main(file_path):
    # Step 1: Analyze metadata
    print("Analyzing audio metadata...")
    metadata = analyze_audio_metadata(file_path)
    if not metadata:
        print("Failed to analyze audio metadata.")
        return

    # Step 2: Generate spectrum image
    print("Generating frequency spectrum image...")
    spectrum_image = generate_spectrum_image(file_path)
    if not spectrum_image:
        print("Failed to generate spectrum image.")
        return

    # Step 3: Analyze spectrum image
    print("Analyzing frequency spectrum...")
    freq_cutoff = analyze_spectrum_image(spectrum_image, metadata["sample_rate"])
    if freq_cutoff is None:
        print("Failed to analyze frequency spectrum.")
        freq_cutoff = "N/A"

    # Step 4: Determine audio quality
    print("Determining audio quality...")
    quality = determine_quality(metadata, freq_cutoff)

    # Step 5: Output results
    print("\nAudio Quality Analysis:")
    print(f"Bitrate: {metadata['bit_rate']} kbps")
    print(f"Bitrate Mode: {metadata['bit_rate_mode']}")
    print(f"Sample Rate: {metadata['sample_rate']} Hz")
    print(f"Codec: {metadata['codec']}")
    print(f"Maximum Frequency: {freq_cutoff} Hz")
    print(f"Determined Quality: {quality}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_audio_quality.py <audio_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    main(file_path)
