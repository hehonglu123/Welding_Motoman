# Microphone Audio Processing
This directory contains a collection of scripts and utilities designed for audio processing, specifically revolving around microphone-based tasks.

## Files Overview
10s_welding.wav: A sample audio file capturing a 10-second welding process.
1000hzfilter.py: Script designed to apply a 1000Hz filter to audio data.
acoustic_feature_capture.py: Captures specific acoustic features from audio inputs.
analysis.py: Provides an in-depth analysis of the given audio data.
microphone_record_save.py: Records audio from a microphone and saves it.
microphone_stream_save.py: Captures streaming audio from a microphone and saves it in real-time.
output.wav: An output audio file, likely generated from one of the processing scripts.
PCA_MFCC.py: Performs Principal Component Analysis (PCA) on Mel Frequency Cepstral Coefficients (MFCC) derived from audio.
play_record.py: Plays and possibly records audio.
unit_test.py: Contains unit tests for the various functionalities.
wav_cutter.py: Cuts or trims WAV audio files.
wavelet_denoise.py: Applies wavelet-based denoising techniques on audio data.
wavmerge.py: Script used for merging multiple WAV audio files.
## Required Libraries
Ensure you have the following libraries installed to run the scripts:

librosa & librosa.display: For audio processing and displaying audio waveforms.
numpy: For numerical operations.
matplotlib & related: For data visualization.
sklearn.decomposition: For decomposition tasks like PCA.
sys: For interacting with the Python runtime environment.
os: For interacting with the operating system.
re: For regular expression operations.
## Usage
To run any script, navigate to the respective file and execute:

python <script_name>.py
Ensure you have the necessary audio files in the expected directories before running.SS