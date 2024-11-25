# Whats my Bitrate? - Audio Quality Analyzer

## Overview
This tool analyzes audio files for quality metrics such as bit rate, frequency, and codec type. It also generates spectrograms for visual representation of the audio spectrum. It supports a variety of audio formats, including MP3, FLAC, WAV, AAC, M4A, and more.

## Features
- Analyze audio files for detailed quality metrics.
- Generate spectrograms for audio visualization.
- Supports multiple file types: `mp3`, `flac`, `wav`, `aac`, `ogg`, `m4a`, `aiff`.
- Batch processing of files in a directory.
- Recursive directory scanning.
- Multi-threaded processing for faster analysis.

## Requirements
- Python 3.7 or later
- FFmpeg (for metadata extraction and file decoding)
- Required Python libraries (see `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/oren-cohen/whatsmybitrate.git
   cd whatsmybitrate
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install FFmpeg:
   - **Linux**:
     ```bash
     sudo apt update
     sudo apt install ffmpeg
     ```
   - **MacOS**:
     ```bash
     brew install ffmpeg
     ```
   - **Windows**:
     1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html).
     2. Extract the archive and add the `bin` folder to your system's PATH.

   To verify FFmpeg installation, run:
   ```bash
   ffmpeg -version
   ```

## Usage
### Basic Command
Analyze all audio files in the current and generate a report:
```bash
python whatsmybitrate.py -f report.html -l -a
```

### Command-Line Arguments
| Argument          | Description                                                                                 |
|--------------------|---------------------------------------------------------------------------------------------|
| `-f <file>`       | Output HTML file name (required).                                                          |
| `-l`              | Enable logging.                                                                            |
| `-a`              | Analyze all supported audio types.                                                         |
| `-t <type>`       | Analyze a specific file type (e.g., `mp3`, `wav`).                                         |
| `-r`              | Scan directories recursively.                                                              |
| `-m <threads>`    | Number of threads to use for analysis (default: 1).                                        |
| `<input>`         | Specify individual files or patterns (e.g., `*.mp3`, `*.wav`).                             |

### Examples
1. Analyze all files in a directory recursively:
   ```bash
   python whatsmybitrate.py -f analysis.html -l -a -r /path/to/directory
   ```

2. Analyze only `mp3` files in a directory recursively:
   ```bash
   python whatsmybitrate.py -f mp3_analysis.html -t mp3 -r /path/to/directory
   ```

3. Analyze specific files:
   ```bash
   python whatsmybitrate.py -f specific_files.html file1.mp3 file2.wav
   ```

![image](https://github.com/user-attachments/assets/1c6e089a-b934-41f3-84fb-e07855121b54)

## Supported Audio Formats
- WAV
- FLAC
- MP3
- AAC
- OGG
- M4A
- AIFF

## Output
- **HTML Report**: Contains detailed metrics for each file, including:
  - Codec
  - Sample Rate
  - Max Frequency
  - Nyquist Frequency
  - Frequency Ratio
  - Stated Bit Rate
  - Estimated Bitrate
  - Spectrogram image (if generated).

## Troubleshooting
### FFmpeg Not Found
Ensure FFmpeg is installed and added to your system's PATH. Test the installation by running:
```bash
ffmpeg -version
```

### Missing Spectrograms
If spectrograms are missing:
- Ensure `matplotlib` and `librosa` are installed:
  ```bash
  pip install matplotlib librosa
  ```
- Check that the audio file is not corrupt or unsupported.

### Errors in Logs
If the script encounters errors, enable logging with the `-l` flag. Review the logs for detailed error messages.


## License
This project is not licensed. Do you want you want with it. :)

---

If you encounter any issues, feel free to open an issue on the GitHub repository.
