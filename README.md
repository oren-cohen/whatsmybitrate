
# 🎵 Python v0.1: Audio Quality Analysis Tool (Updated)

This Python script analyzes audio files to determine their quality, detect recompression, and identify their original quality (e.g., 128 kbps, 192 kbps, 256 kbps). It supports popular audio formats like **MP3**, **AAC**, **FLAC**, **WAV**, and **AIFF**.

## ✨ Features

- **Metadata Analysis**: Extracts codec, bitrate, and sample rate using `ffprobe`.
- **Frequency Spectrum Analysis**: Generates a spectrum image using `ffmpeg` and analyzes maximum frequency.
- **Quality Determination**:
  - Detects high-quality, low-quality, and recompressed files.
  - Accurately outputs the actual quality (e.g., `Detected Quality: 224 kbps`).
- **Wildcard Support (`*`)**: Analyze all files in a directory without requiring quotes.
- **Output to File (`-f`)**: Save results to a specified output file.

## 📋 Requirements

- **Python**: Version 3.8 or newer.
- **FFmpeg**: Required for metadata and spectrum analysis.
- **Pillow**: Required for image processing.

## 🛠 Installation

### Install Python

Ensure Python 3.8+ is installed:

```bash
python3 --version
```

If not installed, [download Python](https://www.python.org/downloads/).

---

### Install Required Libraries

Install necessary Python libraries:

```bash
pip install pillow
```

---

### Install FFmpeg

Install FFmpeg using your system’s package manager:

- **macOS**:
  ```bash
  brew install ffmpeg
  ```
- **Ubuntu/Debian**:
  ```bash
  sudo apt install ffmpeg
  ```
- **CentOS/Red Hat**:
  ```bash
  sudo yum install epel-release -y
  sudo yum install ffmpeg -y
  ```
- **Windows**:  
  [Download FFmpeg](https://ffmpeg.org/download.html) and add it to your PATH.

---

## 🚀 Usage

Run the script with an audio file or directory:

### Analyze a Single Audio File:
```bash
python3 audio_quality_v0.1.py example.mp3
```

### Analyze Multiple Files Without Quotes:
```bash
python3 audio_quality_v0.1.py *.mp3
```

### Analyze All Files in a Specific Directory:
```bash
python3 audio_quality_v0.1.py /path/to/directory
```

### Output Results to a File:
```bash
python3 audio_quality_v0.1.py *.mp3 -f results.txt
```

---

## 📝 Example Output

```plaintext
Analyzing: example.mp3
Audio Quality Analysis for example.mp3:
Codec: mp3
Bitrate: 224 kbps
Sample Rate: 44100 Hz
Maximum Frequency: 17000 Hz
Determined Quality: Detected Quality: 224 kbps
```

---

## Key Improvements

1. **Wildcard Support**:
   - No need to wrap `*` in quotes. Just pass `*.mp3` or similar patterns directly.
2. **Accurate Quality Detection**:
   - Outputs specific detected quality (e.g., `Detected Quality: 224 kbps`) instead of "Unknown or recompressed."
3. **Streamlined File Output**:
   - Use the `-f` flag to write results to a specified file.

---

## Notes

- **FFmpeg version 4.0+** is recommended.
- The script can identify recompressed files and estimate their original quality.
- Spectrum images are generated as `.png` files for each analyzed audio file.
