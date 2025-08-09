# WhatsMyBitrate - Audio Quality Analyzer v0.1.0

<p align="center">
  <img src="https://github.com/user-attachments/assets/1c6e089a-b934-41f3-84fb-e07855121b54" alt="Spectrogram Example" width="700">
</p>

## Overview
**WhatsMyBitrate** is a powerful command-line tool for analyzing audio files in bulk. It provides detailed quality metrics, estimates perceptual quality based on spectral analysis, and generates spectrograms for visual inspection. By processing entire directories recursively with multi-threading, it's an efficient solution for managing and verifying large audio collections.

The script supports a wide variety of lossy and lossless audio formats, including MP3, FLAC, WAV, AAC, M4A, Opus, and more.

## Features
- **Detailed Quality Analysis:** Reports codec, sample rate, stated bit rate, and more.
- **Perceptual Quality Estimation:** Estimates the equivalent bitrate (e.g., 128, 192, 256, 320 kbps) based on spectral frequency cutoffs.
- **Spectrogram Generation:** Creates spectrogram images for visual analysis of the audio spectrum.
- **Cross-Platform:** Full support for **Windows, macOS, and Linux**.
- **Broad Format Support:** Analyzes `mp3`, `flac`, `wav`, `aac`, `ogg`, `m4a`, `aiff`, `opus`, and `alac` files.
- **Batch Processing:** Analyze entire directories of audio files at once.
- **Recursive Scanning:** Use the `-r` flag to scan all subdirectories.
- **High-Performance:** Use the `-m` flag to enable multiprocessing and analyze files significantly faster.
- **Flexible Reporting:** Export detailed reports in **HTML** (with spectrograms) or **CSV** format.
- **Automatic Organization:** Reports and assets are saved into a new, uniquely named directory for each run.

## Requirements
- **OS:** Windows, macOS, or Linux
- **Python:** 3.11 or later
- **FFmpeg:** Required for metadata extraction.
- **Python Libraries:** See `requirements.txt`.

## Installation

### Clone the repository
```bash
git clone https://github.com/oren-cohen/whatsmybitrate.git
cd whatsmybitrate
```

### Install Python dependencies
```bash
pip install -r requirements.txt
```

### Install FFmpeg

**This tool depends on `ffprobe`, which is part of the FFmpeg suite.**

#### macOS (with Homebrew)
```bash
brew install ffmpeg
```

#### Linux (Debian/Ubuntu)
```bash
sudo apt update && sudo apt install ffmpeg
```

#### Windows
You have two options:

**Recommended (No PATH editing):**
1. Download an FFmpeg build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or the official FFmpeg site.
2. Extract the zip file.
3. Find `ffprobe.exe` inside the `bin` folder.
4. Copy `ffprobe.exe` into the same `whatsmybitrate` directory as the Python script. The script will find it automatically.

**Alternative (Manual PATH setup):**
Run the script with the `--ffprobe-path` argument:
```bash
python whatsmybitrate.py "C:\path\to\my music" --ffprobe-path "C:\path\to\ffmpeg\bin\ffprobe.exe"
```

To verify your FFmpeg installation:
```bash
ffmpeg -version
ffprobe -version
```

## Usage
The script is invoked from the command line with options and input targets. Input can be one or more files, a directory, or a glob pattern. Reports are automatically saved to a new directory (e.g., `whatsmybitrate_report_20250809_161939_a1b2c3/`).

### View available arguments
```bash
python whatsmybitrate.py -h
```

**Output:**
```
usage: whatsmybitrate.py [-h] [-c] [-t TYPE | -a] [-r] [-m] [-n] [-l] [--ffprobe-path FFPROBE_PATH] [input ...]

Analyzes audio files.

Input & Output Arguments:
  input                 The target for analysis: one or more files, a directory, or a shell glob pattern.
  -c, --csv             Output the report in CSV format instead of the default HTML.

File Scanning & Filtering Arguments:
  -t TYPE, --type TYPE  Scan a directory for a single file TYPE (e.g., 'mp3', 'flac').
  -a, --all             Scan for all supported audio file types.
  -r, --recursive       Scan directories recursively.

Performance & Utility Arguments:
  -m, --multiprocessing Enable multiprocessing using all available CPU cores.
  -n, --no-spectrogram  Disable spectrogram generation in HTML reports.
  -l, --log             Enable verbose logging to a uniquely named log file.
  --ffprobe-path FFPROBE_PATH
                        Specify the full path to the ffprobe executable.
```

## Examples

**Analyze all supported audio files in the current directory:**
```bash
python whatsmybitrate.py . -a
```

**Analyze all `.flac` files in a specific directory and subfolders, using multiprocessing:**
```bash
python whatsmybitrate.py /path/to/music -t flac -r -m
```

**Generate a CSV report for specific MP3 files:**
```bash
python whatsmybitrate.py "song 1.mp3" "another song.mp3" -c
```
*Note: On Windows `cmd`, glob patterns like `*.mp3` are not automatically expanded. Use `-a` or `-t` instead.*

**Analyze a directory recursively and disable spectrograms and logging:**
```bash
python whatsmybitrate.py /path/to/archive -a -r -n
```

## Supported Audio Formats
- WAV
- FLAC
- ALAC
- MP3
- AAC
- M4A
- Opus
- OGG (Vorbis)
- AIFF

## Output
The script generates reports in a new, uniquely named folder:

### HTML Report (`.html`)
A visual report containing:
- Codec & Sample Rate
- Peak Frequency & Frequency Ratios
- Stated & Estimated Bitrate
- Lossless / Transcode status
- Spectrogram image (if generated)

### CSV Report (`.csv`)
Contains the same metrics in a tabular format for spreadsheets, analysis, or integration with other tools.  
*(No spectrogram images in CSV.)*

## Troubleshooting

**FFmpeg Not Found**  
If you see:
```
ERROR: ffprobe executable not found
```
- Copy `ffprobe.exe` (Windows) or `ffprobe` (macOS/Linux) into the same folder as `whatsmybitrate.py`.
- Or ensure its location is in your system's PATH.
- Or use the `--ffprobe-path` argument.

**Errors in Logs**  
Run with the `-l` flag to generate a detailed log file inside the report directory for debugging.

## License
This project is licensed under the **MIT License**.
