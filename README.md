Whatsmybitrate
Audio Quality Analysis Tool

This Python script analyzes audio files to determine their quality, detect recompression, and identify their original quality (e.g., 128 kbps, 192 kbps, 256 kbps). It supports formats like MP3, AAC, FLAC, WAV, and AIFF.

Features

	•	Metadata Analysis: Extracts codec, bitrate, and sample rate using ffprobe.
	•	Frequency Spectrum Analysis: Generates a spectrum image using ffmpeg.
	•	Quality Determination: Detects high-quality, low-quality, and recompressed files.

Requirements

	•	Python: Version 3.8 or newer.
	•	FFmpeg: Required for metadata and spectrum analysis.
	•	Pillow: Required for image processing.

Installation

Install Python

Ensure Python 3.8+ is installed:

python3 --version

Download from Python.org if not installed.

Install Required Libraries

Install necessary Python libraries:

pip install pillow

nstall FFmpeg

Install FFmpeg using your system’s package manager:
	•	macOS: brew install ffmpeg

 	•	Ubuntu/Debian:
  sudo apt install ffmpeg

  CentOS/Red Hat:

  sudo yum install epel-release -y
sudo yum install ffmpeg -y

	•	Windows:
Download from FFmpeg.org and add it to your PATH.
Usage:
Run the script with an audio file:

python3 whatsmybitrate.py <audio_file>

The script outputs:
	•	Bitrate
	•	Sample Rate
	•	Maximum Frequency
	•	Determined Quality

Analyzing audio metadata...
Analyzing waveform data...
Generating frequency spectrum image...

Audio Quality Analysis:
Bitrate: 128 kbps
Sample Rate: 44100 Hz
Codec: mp3
Maximum Frequency: 16000 Hz
Determined Quality: Low-quality MP3 (128 kbps)

	•	FFmpeg version 4.0+ is recommended.
	•	The script can identify recompressed files and estimate their original quality.
