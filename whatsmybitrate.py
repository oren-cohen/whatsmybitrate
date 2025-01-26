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
import logging
import signal
import soundfile as sf
import audioread
from functools import partial


SUPPORTED_FORMATS = ['wav', 'flac', 'mp3', 'aac', 'ogg', 'm4a', 'aiff']

# Global logger initialized conditionally
logger = None



def setup_logger(enable_logging):
    global logger
    logger = logging.getLogger("audio_analysis")
    logger.setLevel(logging.DEBUG if enable_logging else logging.CRITICAL)

    # Clear existing handlers to avoid duplication
    if logger.handlers:
        logger.handlers.clear()

    # StreamHandler for console logs
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # FileHandler for log file
    file_handler = logging.FileHandler("audio_analysis.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Add the handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False  # Prevent duplicate logs to the console

    if enable_logging:
        logger.info("Logging is enabled and setup complete.")
    else:
        print("Logging is disabled.")

        
def generate_random_filename(base_name="spectrum", extension=".png"):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    return f"{base_name}_{current_time}_{random_str}{extension}"


def analyze_flac_losslessness(file_path, y, sr):
    try:
        # FFT for frequency analysis
        n_fft = 4096
        hop_length = n_fft // 4
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        S_dB = librosa.amplitude_to_db(S, ref=np.max)
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Frequency cutoff detection
        intensity_threshold = -60  # dB threshold for significant frequencies
        max_intensity_per_bin = S_dB.max(axis=1)
        significant_indices = np.where(max_intensity_per_bin > intensity_threshold)[0]
        max_freq = frequencies[significant_indices[-1]] if significant_indices.size > 0 else 0.0
        nyquist = sr / 2
        frequency_ratio = max_freq / nyquist if nyquist else 0

        # Dynamic range analysis
        dynamic_range = np.max(S_dB) - np.min(S_dB)

        # Metadata extraction for bitrate
        bit_rate, codec, sample_rate, channels, bits_per_sample = extract_metadata(file_path)

        # Logging details
        if logger:
            logger.info(f"FLAC Analysis: File: {file_path}")
            logger.info(f"  Max Frequency: {max_freq} Hz")
            logger.info(f"  Nyquist Frequency: {nyquist} Hz")
            logger.info(f"  Frequency Ratio: {frequency_ratio}")
            logger.info(f"  Dynamic Range: {dynamic_range:.2f} dB")
            logger.info(f"  Bitrate: {bit_rate} kbps")

        # Adjusted heuristic for losslessness
        lossless = (
            frequency_ratio > 0.8  # Use 80% of Nyquist
            and dynamic_range > 30  # Dynamic range above 30 dB
            and (bit_rate is None or bit_rate > 800)  # High bitrate or unknown bitrate
        )

        # Detailed analysis results
        analysis_details = {
            "file": file_path,
            "max_frequency": max_freq,
            "nyquist_frequency": nyquist,
            "frequency_ratio": frequency_ratio,
            "dynamic_range": dynamic_range,
            "bit_rate": f"{bit_rate} kbps" if bit_rate else "Unknown",
            "codec": codec,
            "lossless": lossless,  # Renamed field
        }

        return lossless, analysis_details

    except Exception as e:
        if logger:
            logger.error(f"Error analyzing FLAC file: {file_path} - {str(e)}")
        return False, {"error": str(e)}

    except Exception as e:
        if logger:
            logger.error(f"Error analyzing FLAC file: {file_path} - {str(e)}")
        return False, {"error": str(e)}

import os

def extract_metadata(file_path):
    try:
        ffprobe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=bit_rate,codec_name,sample_rate,channels,bits_per_sample,duration",
            "-of", "json", file_path
        ]
        result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        metadata = json.loads(result.stdout)
        stream = metadata.get("streams", [{}])[0]

        # Extract metadata with default fallbacks
        codec = stream.get("codec_name", "Unknown")
        sample_rate = int(stream.get("sample_rate", 0)) if stream.get("sample_rate") else None
        channels = int(stream.get("channels", 1)) if stream.get("channels") else None
        bits_per_sample = int(stream.get("bits_per_sample", 0)) if stream.get("bits_per_sample") else None
        duration = float(stream.get("duration", 0)) if stream.get("duration") else None

        # Extract bit_rate or calculate it if missing
        bit_rate = int(stream.get("bit_rate", 0)) // 1000 if stream.get("bit_rate") else None
        if bit_rate is None and duration:
            file_size = os.path.getsize(file_path)  # File size in bytes
            bit_rate = int((file_size * 8) / duration / 1000)  # Convert to kbps

        if logger:
            logger.info(f"Metadata for {file_path}: {stream}")
            logger.info(f"Calculated Bitrate: {bit_rate} kbps" if bit_rate else "Bitrate unavailable")

        return bit_rate, codec, sample_rate, channels, bits_per_sample
    except Exception as e:
        if logger:
            logger.error(f"Error extracting metadata for {file_path}: {e}")
        return None, "Unknown", None, None, None


def estimate_actual_bitrate(codec, max_frequency, sample_rate=None, channels=None, bit_rate=None, bits_per_sample=None, gap_detected=False):
    """
    Estimate the actual bitrate based on codec, max frequency, and metadata.
    Includes gap detection for .m4a files.
    """
    nyquist = sample_rate / 2 if sample_rate else 1
    frequency_ratio = max_frequency / nyquist if nyquist else 0

    if logger:
        logger.info(f"Max Frequency: {max_frequency:.2f} Hz, Nyquist: {nyquist:.2f} Hz, Frequency Ratio: {frequency_ratio:.2f}")
        logger.info(f"Codec: {codec}, Bitrate: {bit_rate}, Sample Rate: {sample_rate}, Bits per Sample: {bits_per_sample}, Channels: {channels}")

    # Lossless codecs, including AIFF variants
    if codec.lower() in ["wav", "flac", "aiff", "pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_s16be"]:
        if frequency_ratio >= 0.935:
            bitrate = "Lossless (Uncompressed)"
        elif frequency_ratio >= 0.77:
            bitrate = "320 kbps MP3 Equivalent (Re-encoded)"
        elif frequency_ratio >= 0.685:
            bitrate = "128 kbps MP3 Equivalent (Re-encoded)"
        else:
            bitrate = "Low-Quality Re-encoded Lossless"
    elif codec.lower() in ["aac", "m4a"]:
        if frequency_ratio >= 0.86:
            bitrate = "320kbps kbps MP3 Equivalent or better (Good Quality)"
        else:
            bitrate = "Low-Quality AAC/M4A"            
    # Lossy compressed codecs
    elif codec.lower() in ["mp3", "aac", "ogg", "m4a", "vorbis"]:
        if frequency_ratio >= 0.84:
            bitrate = "320 kbps MP3 Equivalent (High Quality)"
        elif frequency_ratio >= 0.78:
            bitrate = "256 kbps MP3 Equivalent (Good Quality)"
        elif frequency_ratio >= 0.685:
            bitrate = "128 kbps MP3 Equivalent (Low Quality)"
        else:
            bitrate = "Very Low Bitrate"
    # Unknown or unhandled codecs
    else:
        bitrate = "Unknown Format"

    return bitrate, nyquist, frequency_ratio

def generate_spectrogram(y, sr, file_path):
    """
    Generates a spectrogram for the given audio data and saves it to a PNG file.
    Handles short signals gracefully.
    """
    try:
        if logger:
            logger.info(f"Starting spectrogram generation for: {file_path}")

        if y is None or len(y) == 0:
            logger.error(f"Audio data is empty or None for: {file_path}")
            return None

        if len(y) < 16384:
            logger.warning(f"Audio signal too short for FFT (length={len(y)}). Skipping spectrogram for {file_path}.")
            return None

        nyquist = sr / 2
        n_fft = 16384
        hop_length = n_fft // 4

        # Compute the spectrogram
        try:
            logger.info(f"Performing STFT for {file_path} with n_fft={n_fft} and hop_length={hop_length}.")
            S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
            if logger:
                logger.debug(f"STFT computed. Shape of S: {S.shape}")
        except Exception as stft_error:
            logger.error(f"STFT computation failed for {file_path}: {stft_error}")
            return None

        try:
            logger.info(f"Converting amplitude to dB for {file_path}.")
            S_dB = librosa.amplitude_to_db(S, ref=np.max)
            if logger:
                logger.debug(f"Amplitude to dB conversion done for {file_path}.")
        except Exception as amplitude_error:
            logger.error(f"Amplitude-to-dB conversion failed for {file_path}: {amplitude_error}")
            return None

        # Sanitize the filename and ensure the directory exists
        base_filename = os.path.basename(file_path)
        safe_filename = base_filename.replace(" ", "_").replace("'", "").replace('"', "")
        spectrogram_dir = os.path.dirname(file_path)
        if not os.path.exists(spectrogram_dir):
            os.makedirs(spectrogram_dir)

        # Generate a valid filename
        spectrogram_file = os.path.join(spectrogram_dir, f"{safe_filename}_spectrogram.png")

        # Plot and save the spectrogram
        try:
            logger.info(f"Plotting spectrogram for {file_path}.")
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(
                S_dB,
                sr=sr,
                x_axis="time",
                y_axis="linear",
                hop_length=hop_length,
                cmap="viridis",
                fmax=nyquist,
            )
            plt.colorbar(format="%+2.0f dB")
            plt.title(f"Spectrogram for {os.path.basename(file_path)} (up to {nyquist:.2f} Hz)")
            plt.tight_layout()

            # Save the spectrogram image
            plt.savefig(spectrogram_file)
            plt.close()

            logger.info(f"Spectrogram saved successfully for {file_path} at {spectrogram_file}.")
            return spectrogram_file
        except Exception as plotting_error:
            logger.error(f"Spectrogram plotting or saving failed for {file_path}: {plotting_error}")
            return None

    except Exception as e:
        logger.error(f"Unexpected error during spectrogram generation for {file_path}: {e}")
        return None

            
            
def load_audio(file_path):
    """
    Load audio using soundfile, with fallback to audioread and librosa for unsupported formats.
    Converts multi-channel audio to mono if necessary.
    """
    try:
        # Use soundfile for primary loading
        y, sr = sf.read(file_path, always_2d=False)
        if logger:
            logger.info(f"Loaded audio using soundfile: {file_path}")
            logger.info(f"Audio data shape: {y.shape if hasattr(y, 'shape') else 'Unknown'}, Sample rate: {sr}")

        # Convert multi-channel audio to mono
        if y.ndim > 1:
            if logger:
                logger.info(f"Converting multi-channel audio to mono for file: {file_path}")
            y = np.mean(y, axis=1)

        return y, sr
    except Exception as e:
        if logger:
            logger.warning(f"soundfile failed for {file_path}: {e}. Falling back to audioread.")

        # Fallback to audioread for unsupported formats
        try:
            with audioread.audio_open(file_path) as input_file:
                sr = input_file.samplerate
                data = np.frombuffer(b"".join(input_file), dtype=np.int16)
                y = data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

                if logger:
                    logger.info(f"Loaded audio using audioread: {file_path}")
                    logger.info(f"Audio data shape: {y.shape if hasattr(y, 'shape') else 'Unknown'}, Sample rate: {sr}")

                # Convert to mono
                if y.ndim > 1:
                    if logger:
                        logger.info(f"Converting multi-channel audio to mono for file: {file_path}")
                    y = np.mean(y, axis=1)

                return y, sr
        except Exception as audioread_error:
            if logger:
                logger.warning(f"audioread failed for {file_path}: {audioread_error}. Falling back to librosa.")

            # Final fallback to librosa
            try:
                y, sr = librosa.load(file_path, sr=None, mono=True)
                if logger:
                    logger.info(f"Loaded audio using librosa: {file_path}")
                    logger.info(f"Audio data shape: {y.shape if hasattr(y, 'shape') else 'Unknown'}, Sample rate: {sr}")
                return y, sr
            except Exception as librosa_error:
                if logger:
                    logger.error(f"Failed to load audio file with any method: {librosa_error}")
                return None, None


def analyze_spectrum(y, sr):
    """
    Analyzes the spectrogram for significant frequencies.
    Returns the maximum frequency with significant energy.
    """
    try:
        n_fft = 4096
        hop_length = n_fft // 4
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        S_dB = librosa.amplitude_to_db(S, ref=np.max)
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Log spectrum shape and frequency range
        if logger:
            logger.info(f"Spectrum shape: {S_dB.shape}, Frequencies: {frequencies}")

        # Detect max frequency with significant energy
        intensity_threshold = -70  # Ignore signals below -70 dB
        max_intensity_per_bin = S_dB.max(axis=1)  # Maximum intensity for each frequency bin
        significant_indices = np.where(max_intensity_per_bin > intensity_threshold)[0]

        if significant_indices.size == 0:
            if logger:
                logger.warning("No significant frequencies detected.")
            return 0.0, False

        # Find the highest frequency with significant energy
        max_freq = frequencies[significant_indices[-1]]

        # Log the detected max frequency
        if logger:
            logger.info(f"Max frequency detected: {max_freq} Hz")

        return max_freq, False  # No gap detection implemented
    except Exception as e:
        if logger:
            logger.error(f"Error during spectrum analysis: {e}")
        return 0.0, False
    
def detect_double_compression(mdct_coefficients):
    """
    Detects double compression in MDCT coefficients.
    """
    try:
        if mdct_coefficients is None or len(mdct_coefficients) == 0:
            return False

        # Example: Analyze variance or patterns in coefficients
        # High variance or low entropy could indicate double compression
        variance = np.var(mdct_coefficients)
        entropy = -np.sum(mdct_coefficients * np.log(np.abs(mdct_coefficients) + 1e-10))
        
        if logger:
            logger.info(f"MDCT Coefficients Analysis - Variance: {variance:.2f}, Entropy: {entropy:.2f}")
        
        # Simple heuristic for double compression (tune thresholds as needed)
        if variance < 0.5 or entropy > 1.0:
            return True
        return False
    except Exception as e:
        if logger:
            logger.error(f"Error analyzing MDCT coefficients for double compression: {e}")
        return False
    
def extract_mdct_coefficients(file_path):
    """
    Extracts MDCT coefficients from the AAC file.
    This uses an external tool (like ffmpeg or aac libraries).
    """
    try:
        # Decode the file into raw PCM using ffmpeg
        decoded_pcm_file = "decoded.raw"
        subprocess.run(
            [
                "ffmpeg",
                "-i", file_path,
                "-f", "s16le",  # Raw PCM format
                "-acodec", "pcm_s16le",
                decoded_pcm_file,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Load PCM data into numpy for further analysis
        pcm_data = np.fromfile(decoded_pcm_file, dtype=np.int16)

        # Perform MDCT (you can implement or use a library for this)
        # Example: librosa.feature.mfcc or another transform
        mdct_coefficients = librosa.feature.mfcc(y=pcm_data.astype(float), sr=44100)

        return mdct_coefficients
    except Exception as e:
        if logger:
            logger.error(f"Error extracting MDCT coefficients: {e}")
        return None
    
def analyze_huffman_codebooks(file_path):
    """
    Decodes the AAC file to access Huffman codebook indices.
    Analyzes their distribution for anomalies.
    """
    try:
        # Use an AAC decoder library to extract Huffman indices
        # Example: Using ffmpeg or fdk-aac for decoding
        result = subprocess.run(
            [
                "ffprobe",
                "-show_entries",
                "frame_tags",
                "-select_streams", "a",
                "-i", file_path,
                "-of", "json",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Parse the JSON output
        huffman_indices = json.loads(result.stdout).get("frames", [])

        # Analyze the distribution
        anomalies = check_huffman_anomalies(huffman_indices)
        return anomalies
    except Exception as e:
        if logger:
            logger.error(f"Error analyzing Huffman codebooks: {e}")
        return False


def check_huffman_anomalies(indices):
    """
    Analyze Huffman codebook indices for anomalies.
    """
    # Example: Check for unusual index distributions
    # This is highly codec-dependent; use a model trained on typical distributions
    if len(indices) > 0:
        # Example logic: Check variance or unusual counts
        distribution = np.bincount(indices)
        return np.var(distribution) > threshold
    return False


def process_file(
    file_path, timeout_duration=60, enable_logging=True, enable_spectrogram=True
):
    # Initialize logger inside the worker to avoid the UnboundLocalError
    logger = setup_logger(enable_logging)
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_duration)
    try:
        if logger:
            logger.info(f"Starting processing for file: {file_path}")

        # Extract metadata
        bit_rate, codec, sample_rate, channels, bits_per_sample = extract_metadata(file_path)

        if logger:
            logger.info(
                f"Metadata extracted for {file_path}: Bitrate: {bit_rate}, Codec: {codec}, "
                f"Sample Rate: {sample_rate}, Channels: {channels}, Bits per Sample: {bits_per_sample}"
            )

        if codec is None:
            logger.error(f"Unable to extract codec information for {file_path}")
            return {"file": file_path, "error": "Unable to extract codec information"}

        # Attempt to load the audio
        if logger:
            logger.info(f"Attempting to load audio file: {file_path}")
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # Log audio data details
        if y is None or len(y) == 0:
            if logger:
                logger.error(f"Audio data is empty for file: {file_path}. Attempting with audioread.")
            y, sr = load_audio(file_path)

        if y is None or len(y) == 0:
            if logger:
                logger.error(f"Failed to load audio data for: {file_path}")
            return {"file": file_path, "error": "Unable to load audio data"}

        # Perform FLAC-specific analysis if applicable
        flac_analysis = {}
        if codec.lower() == "flac":
            is_lossless, flac_analysis = analyze_flac_losslessness(file_path, y, sr)
            if logger:
                logger.info(f"FLAC Analysis Results: {flac_analysis}")
        else:
            is_lossless = None

        # Spectrum analysis
        if logger:
            logger.info(f"Analyzing spectrum for: {file_path}")
        max_freq, gap_detected = analyze_spectrum(y, sr)

        if logger:
            logger.info(f"Spectrum analysis results for {file_path}: Max Frequency: {max_freq}, Gap Detected: {gap_detected}")

        # Estimate bitrate
        estimated_bitrate, nyquist, frequency_ratio = estimate_actual_bitrate(
            codec, max_freq, sample_rate, channels, bit_rate, bits_per_sample, gap_detected
        )

        # Generate spectrogram
        spectrogram_file = None
        if enable_spectrogram:
            if logger:
                logger.info(f"Generating spectrogram for: {file_path}")
            spectrogram_file = generate_spectrogram(y, sr, file_path)
            if logger:
                if spectrogram_file:
                    logger.info(
                        f"Spectrogram successfully generated: {spectrogram_file}"
                    )
                else:
                    logger.error(f"Spectrogram generation failed for: {file_path}")

        # Compile result
        result = {
            "file": file_path,
            "codec": codec,
            "sample_rate": sr,
            "max_frequency": max_freq,
            "nyquist_frequency": nyquist,
            "frequency_ratio": frequency_ratio,
            "bit_rate": f"{bit_rate} kbps" if bit_rate else "Unknown",
            "estimated_bitrate": estimated_bitrate,
            "spectrogram": spectrogram_file,
            "is_lossless": is_lossless,
            **flac_analysis  # Include detailed FLAC analysis
        }

        if logger:
            logger.info(f"Result for file {file_path}: {result}")

        signal.alarm(0)
        return result

    except TimeoutError:
        if logger:
            logger.error(f"Timeout while processing file: {file_path}")
        return {"file": file_path, "error": "Processing timed out"}

    except Exception as e:
        if logger:
            logger.error(f"Error processing file {file_path}: {str(e)}")
        return {"file": file_path, "error": str(e)}
    except TimeoutError:
        if logger:
            logger.error(f"Timeout occurred while processing file: {file_path}")
        return {"file": file_path, "error": "Processing timed out"}
    except (FileNotFoundError, IOError) as e:
        if logger:
            logger.error(f"File I/O error while processing {file_path}: {str(e)}")
        return {"file": file_path, "error": "File I/O error"}
    except (librosa.ParameterError, librosa.Error) as e:
        if logger:
            logger.error(f"Librosa error while processing {file_path}: {str(e)}")
        return {"file": file_path, "error": "Librosa error"}
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error while processing {file_path}: {str(e)}")
    finally:
        signal.alarm(0)
        if logger:
            for handler in logging.getLogger().handlers:
                handler.flush()
                
def timeout_handler(signum, frame):
    if logger:
        logger.error("Timeout occurred during processing.")
    raise TimeoutError("File processing timed out")


def generate_html_report(results, html_filename):
    """
    Generates an HTML report with results including estimated bitrate, Nyquist frequency,
    frequency ratio, FLAC losslessness, and spectrograms.
    """
    try:
        with open(html_filename, "w") as html_file:
            # Start HTML document
            html_file.write("<!DOCTYPE html>\n<html lang='en'>\n<head>\n")
            html_file.write("<meta charset='UTF-8'>\n<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n")
            html_file.write("<title>Audio Analysis Report</title>\n")
            html_file.write("<style>\n")
            html_file.write("body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }\n")
            html_file.write(".result { border: 1px solid #ccc; padding: 10px; margin-bottom: 20px; }\n")
            html_file.write(".result img { max-width: 100%; height: auto; display: block; margin: 10px 0; }\n")
            html_file.write("h1, h2 { color: #333; }\n")
            html_file.write("</style>\n</head>\n<body>\n")
            html_file.write("<h1>Audio Analysis Report</h1>\n")

            # Process each result
            for result in results:
                html_file.write("<div class='result'>\n")

                if 'error' in result:
                    # Display error for failed files
                    html_file.write(f"<p><strong>Error:</strong> {result['error']}</p>\n")
                else:
                    # Display details for successfully processed files
                    html_file.write(f"<h2>File: {os.path.basename(result.get('file', 'Unknown'))}</h2>\n")
                    html_file.write(f"<p><strong>Codec:</strong> {result.get('codec', 'Unknown')}</p>\n")
                    html_file.write(f"<p><strong>Sample Rate:</strong> {result.get('sample_rate', 'Unknown')} Hz</p>\n")
                    html_file.write(f"<p><strong>Max Frequency:</strong> {result.get('max_frequency', 'Unknown')} Hz</p>\n")
                    html_file.write(f"<p><strong>Nyquist Frequency:</strong> {result.get('nyquist_frequency', 'Unknown')} Hz</p>\n")
                    html_file.write(f"<p><strong>Frequency Ratio:</strong> {result.get('frequency_ratio', 'Unknown')}</p>\n")
                    html_file.write(f"<p><strong>Stated Bit Rate:</strong> {result.get('bit_rate', 'Unknown')}</p>\n")
                    html_file.write(f"<p><strong>Estimated Bitrate:</strong> {result.get('estimated_bitrate', 'Unknown')}</p>\n")

                    # FLAC-specific analysis (if applicable)
                    if result.get("codec", "").lower() == "flac":
                        html_file.write(f"<p><strong>Lossless:</strong> {'Yes' if result.get('is_lossless', False) else 'No'}</p>\n")
                        html_file.write(f"<p><strong>Dynamic Range:</strong> {result.get('dynamic_range', 'Unknown')} dB</p>\n")

                    # Include spectrogram if available
                    if result.get('spectrogram'):
                        html_file.write(f"<img src='{result['spectrogram']}' alt='Spectrogram for {os.path.basename(result.get('file', 'Unknown'))}'>\n")

                html_file.write("</div>\n")

            # Close HTML document
            html_file.write("</body>\n</html>\n")

        if logger:
            logger.info(f"HTML report successfully saved to {html_filename}")

    except Exception as e:
        if logger:
            logger.error(f"Error generating HTML report: {e}")


def generate_csv_report(results, csv_filename):
    """
    Generates a CSV report with results including estimated bitrate, Nyquist frequency,
    frequency ratio, and FLAC losslessness (spectrograms excluded as they cannot be represented in CSV).
    """
    import csv

    try:
        with open(csv_filename, "w", newline="") as csv_file:
            # Define CSV headers
            fieldnames = [
                "File",
                "Error",
                "Codec",
                "Sample Rate (Hz)",
                "Max Frequency (Hz)",
                "Nyquist Frequency (Hz)",
                "Frequency Ratio",
                "Stated Bit Rate",
                "Estimated Bitrate",
                "Is Lossless",
                "Dynamic Range (dB)",
            ]

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            # Process each result
            for result in results:
                row = {}
                if "error" in result:
                    # Write error information
                    row = {
                        "File": result.get("file", "Unknown"),
                        "Error": result["error"],
                    }
                else:
                    # Write successful analysis data
                    row = {
                        "File": os.path.basename(result.get("file", "Unknown")),
                        "Error": "",
                        "Codec": result.get("codec", "Unknown"),
                        "Sample Rate (Hz)": result.get("sample_rate", "Unknown"),
                        "Max Frequency (Hz)": result.get("max_frequency", "Unknown"),
                        "Nyquist Frequency (Hz)": result.get(
                            "nyquist_frequency", "Unknown"
                        ),
                        "Frequency Ratio": result.get("frequency_ratio", "Unknown"),
                        "Stated Bit Rate": result.get("bit_rate", "Unknown"),
                        "Estimated Bitrate": result.get("estimated_bitrate", "Unknown"),
                        "Is Lossless": (
                            "Yes" if result.get("is_lossless", False) else "No"
                        ),
                        "Dynamic Range (dB)": result.get("dynamic_range", "Unknown"),
                    }
                writer.writerow(row)

        if logger:
            logger.info(f"CSV report successfully saved to {csv_filename}")

    except Exception as e:
        if logger:
            logger.error(f"Error generating CSV report: {e}")


def output_results(results):
    print("\nSummary of all processed files:")
    for result in results:
        if 'error' in result:
            message = f"Error processing {result['file']}: {result['error']}"
        else:
            message = (f"Processed {result['file']}:\n"
                       f"  Codec: {result.get('codec', 'Unknown')}\n"
                       f"  Sample Rate: {result.get('sample_rate', 'Unknown')} Hz\n"
                       f"  Max Frequency: {result.get('max_frequency', 'Unknown')} Hz\n"
                       f"  Nyquist Frequency: {result.get('nyquist_frequency', 'Unknown')} Hz\n"
                       f"  Frequency Ratio: {result.get('frequency_ratio', 'Unknown')}\n"
                       f"  Bit Rate: {result.get('bit_rate', 'Unknown')}\n"
                       f"  Estimated Bitrate: {result.get('estimated_bitrate', 'Unknown')}\n"
                       f"  Spectrogram: {result.get('spectrogram', 'None')}\n")
        if logger:
            logger.info(message)
        else:
            print(message)

def scan_directory(directory, recursive=False, file_patterns=None, file_type=None):
    """
    Scans the specified directory for audio files matching SUPPORTED_FORMATS or custom patterns.
    Returns a list of matching file paths.

    Args:
        directory (str): Directory to scan.
        recursive (bool): Whether to scan directories recursively.
        file_patterns (list): Optional list of file patterns (e.g., ["*.mp3", "*.wav"]).
        file_type (str): Optional file type to scan for (e.g., "mp3").

    Returns:
        list: List of matching file paths.
    """
    if logger:
        logger.info(f"Scanning directory: {directory} (Recursive: {recursive}, Patterns: {file_patterns}, File type: {file_type})")
    
    matching_files = []
    search_patterns = []

    # Build search patterns based on input
    if file_type:
        search_patterns.append(f"*.{file_type}")
    elif file_patterns:
        search_patterns.extend(file_patterns)
    else:
        search_patterns.extend([f"*{ext}" for ext in SUPPORTED_FORMATS])

    # Perform the search
    for pattern in search_patterns:
        search_path = os.path.join(directory, "**", pattern) if recursive else os.path.join(directory, pattern)
        matching_files.extend(glob.glob(search_path, recursive=recursive))

    if logger:
        logger.info(f"Found {len(matching_files)} matching files.")
    return matching_files


def process_file_wrapper(file_path, enable_logging=False, enable_spectrogram=True):
    return process_file(
        file_path,
        enable_logging=enable_logging,
        enable_spectrogram=enable_spectrogram,
    )


def main():
    global logger
    parser = argparse.ArgumentParser(
        description="Analyze audio files and generate a report in HTML or CSV format."
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Output report file name (use .html or .csv extension)",
        required=True,
    )
    parser.add_argument(
        "input", nargs="*", help="Input audio file(s) or patterns (e.g., *.mp3 *.wav)"
    )
    parser.add_argument(
        "-m",
        "--threads",
        type=int,
        default=1,
        help="Number of threads to use (default: 1)",
    )
    parser.add_argument(
        "-a", "--all", action="store_true", help="Scan all supported audio file types"
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Scan directories recursively"
    )
    parser.add_argument(
        "-t", "--type", help="Specify a single file type to scan (e.g., mp3)"
    )
    parser.add_argument("-l", "--log", action="store_true", help="Enable logging")
    parser.add_argument(
        "--no-spectrogram", action="store_true", help="Disable spectrogram generation"
    )
    args = parser.parse_args()

    # Initialize logger
    setup_logger(args.log)
    if logger:
        logger.info("Logger successfully initialized in main.")
        logger.info(f"Script arguments: {args}")

    # Enforce mutual exclusivity of -a and -t
    if args.all and args.type:
        print("Error: You cannot use -a (all file types) with -t (specific file type).")
        if logger:
            logger.error("Mutually exclusive flags -a and -t used together.")
        return

    # Determine directory
    directory = os.getcwd()  # Default to current directory
    if args.input and args.input[0] not in ["*", ".", "/"]:  # If directory or pattern is specified
        directory = args.input[0]
    elif len(args.input) == 1:
        directory = args.input[0] if os.path.isdir(args.input[0]) else os.getcwd()

    # Determine file patterns
    file_patterns = None
    file_type = args.type

    if args.all:
        # Scan all SUPPORTED_FORMATS if -a is specified
        file_patterns = None
    elif args.input:
        # Use input patterns (e.g., *.mp3)
        file_patterns = args.input
    elif not args.all and not file_type:
        # If neither -a nor -t nor patterns are specified, exit with an error
        print("Error: You must specify -a, -t <type>, or file patterns (e.g., *.mp3).")
        if logger:
            logger.error("No file patterns, -a, or -t specified.")
        return

    # Perform directory scan
    matching_files = scan_directory(directory, recursive=args.recursive, file_patterns=file_patterns, file_type=file_type)

    if not matching_files:
        print("No matching files found.")
        if logger:
            logger.warning("No matching files found.")
        return

    # Process files
    results = []
    if args.threads > 1:
        with multiprocessing.Pool(processes=args.threads) as pool:
            process_func = partial(
                process_file_wrapper,
                enable_logging=args.log,
                enable_spectrogram=not args.no_spectrogram,
            )
            results = list(
                tqdm(
                    pool.imap(process_func, matching_files),
                    total=len(matching_files),
                )
            )
    else:
        for file_path in tqdm(matching_files, desc="Processing files"):
            result = process_file_wrapper(file_path, args.log, not args.no_spectrogram)
            if logger:
                logger.info(f"Finished processing file: {file_path}")
            results.append(result)

    # Output results
    output_results(results)

    # Generate reports based on file extension
    output_file = args.file
    if output_file.lower().endswith(".csv"):
        generate_csv_report(results, output_file)
    else:
        generate_html_report(results, output_file)

    if logger:
        logger.info(f"Report successfully generated: {output_file}")


if __name__ == "__main__":
    main()