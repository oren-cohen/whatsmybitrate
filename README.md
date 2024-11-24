
# Whats My Bitrate? - Audio Analysis Tool

This tool analyzes audio files to estimate their actual bitrate based on the maximum significant frequency present in the signal. It generates spectrograms for each file and compiles the results into an HTML report.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Multithreading](#multithreading)
- [Examples](#examples)
- [Script Overview](#script-overview)
- [Dependencies](#dependencies)
- [Limitations](#limitations)
- [License](#license)

## Prerequisites

- **Python 3.6 or higher**
- **ffmpeg**: Required for extracting audio metadata.

  Install ffmpeg:

  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your system PATH.
  - **macOS**: Install via Homebrew:

    ```bash
    brew install ffmpeg
    ```

  - **Linux**: Install via package manager:

    ```bash
    sudo apt-get install ffmpeg
    ```

## Installation

1. **Clone the repository or download the script** to your local machine.

2. **Install required Python packages**:

   ```bash
   pip install librosa numpy matplotlib tqdm
   ```

## Usage

### Basic Usage

Run the script from the command line, specifying the output HTML file and the input audio files or patterns.

```bash
python audio_analysis.py -f results.html input_files
```

- `-f results.html`: Specifies the name of the output HTML report.
- `input_files`: One or more audio files or glob patterns (e.g., `*.wav`, `*.mp3`).

**Example:**

```bash
python audio_analysis.py -f report.html *.wav *.mp3
```

### Multithreading

To speed up the analysis of multiple files, you can enable multithreading by specifying the number of threads with the `-m` option.

```bash
python audio_analysis.py -f results.html -m 4 input_files
```

- `-m 4`: Uses 4 threads for processing.

**Note:** The progress bar is only displayed when multithreading is enabled.

## Examples

- **Analyze all WAV and MP3 files in the current directory:**

  ```bash
  python audio_analysis.py -f results.html *.wav *.mp3
  ```

- **Analyze files with multithreading (e.g., using 8 threads):**

  ```bash
  python audio_analysis.py -f results.html -m 8 *.wav *.mp3
  ```

- **Analyze specific files:**

  ```bash
  python audio_analysis.py -f results.html song1.wav song2.mp3
  ```

## Script Overview

The script performs the following tasks:

1. **Extracts Metadata**: Uses `ffprobe` to extract bitrate, codec, sample rate, channels, and bits per sample from the audio files.

2. **Loads Audio Data**: Reads the audio files using `librosa`.

3. **Analyzes Audio**: Determines the maximum significant frequency present in the signal.

4. **Estimates Bitrate**: Estimates the actual bitrate based on the codec and maximum frequency.

5. **Generates Spectrograms**: Creates spectrogram images for each audio file.

6. **Generates HTML Report**: Compiles all the information into an HTML report with embedded spectrograms.

## Dependencies

- **Python Packages**:
  - `librosa`
  - `numpy`
  - `matplotlib`
  - `tqdm`

- **External Tools**:
  - `ffmpeg` (specifically `ffprobe`)

## Limitations

- **Accuracy of Bitrate Estimation**: The estimation is based on frequency analysis and may not be precise for all audio files, especially those with limited high-frequency content due to the nature of the recording or production.

- **Supported Formats**: Only the following audio formats are supported:
  - WAV
  - FLAC
  - MP3
  - AAC
  - OGG
  - M4A
  - AIFF

- **System Resources**: Analyzing large audio files or a large number of files may consume significant CPU and memory resources.

## License

This script is provided as-is without any warranty. Use it at your own risk.
