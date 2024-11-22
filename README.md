# Whats my Bitrate? - Audio Analysis and HTML Report Generator

This Python script analyzes audio files, generates spectrograms, and creates an HTML report summarizing the analysis. The report includes detailed metadata, frequency analysis, and visual spectrograms for each input audio file.

## Features

- Supports multiple audio formats: `wav`, `flac`, `mp3`, `aac`, `ogg`, `m4a`, `aiff`.
- Extracts metadata such as codec, sample rate, channels, and bitrate using `ffprobe`.
- Generates spectrograms for each audio file.
- Outputs an HTML report with:
  - Audio metadata
  - Visual spectrograms
  - Organized and clean layout

---

## Prerequisites

### Python Version
- **Python 3.8+** is required. For Python 3.6 or 3.7, see [compatibility with older Python versions](#compatibility-with-older-python-versions).

### Dependencies
Install the required Python libraries using:
```bash
pip install -r requirements.txt
```

---

## Usage

### Running the Script

1. **Command Syntax**:
   ```bash
   python script.py -f <output_file.html> <audio_files>
   ```
   Replace `<output_file.html>` with the name of your desired HTML report and `<audio_files>` with the audio file(s) or patterns you want to analyze.

2. **Examples**:
   - Analyze all `.wav` files and save the report as `results.html`:
     ```bash
     python script.py -f results.html "*.wav"
     ```
   - Analyze `.mp3` and `.wav` files together:
     ```bash
     python script.py -f audio_report.html "*.wav" "*.mp3"
     ```

---

## Example Output

After running the script, an HTML file (e.g., `results.html`) will be generated. Open it in any modern web browser to view the following:

- A cleanly formatted table of metadata for each audio file.
- Spectrogram images illustrating the audio's frequency and time distribution.

---

## Compatibility with Older Python Versions

If you're using **Python 3.6 or 3.7**, use the following `requirements.txt`:
```
librosa==0.8.1
matplotlib==3.5.3
numpy==1.21.6
```

Install the dependencies with:
```bash
pip install -r requirements.txt
```

---

## Notes

1. Ensure `ffprobe` (part of `ffmpeg`) is installed and available in your system's PATH.
   - On Ubuntu/Debian:
     ```bash
     sudo apt install ffmpeg
     ```
   - On macOS:
     ```bash
     brew install ffmpeg
     ```

2. Large audio files may take longer to process. Ensure sufficient disk space for spectrogram images.

---

## License

This project is licensed under the MIT License.
