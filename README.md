# AI Transcriber

This project provides a command-line tool to transcribe audio files using a locally-run, GPU-accelerated instance of Whisper.cpp. It is designed for fast and efficient transcription of various audio formats.

## Features

-   **High-Quality Transcription**: Utilizes the Whisper ASR model for accurate speech-to-text conversion.
-   **GPU Acceleration**: Leverages NVIDIA GPUs via CUDA for significantly faster processing.
-   **Audio Pre-processing**: Automatically converts input audio files to the required format (16-bit, 16kHz mono WAV) using `ffmpeg`.
-   **Simple CLI**: Easy-to-use command-line interface for transcribing files.
-   **Local First**: All processing is done locally, ensuring data privacy.
-   **Extensible**: Includes setup for a Llama 3 language model, paving the way for future features like transcript summarization or analysis.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

-   Python 3.8+
-   An NVIDIA GPU with the appropriate CUDA toolkit installed.
-   `ffmpeg`
-   `git`
-   `wget`

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sdmeers/ai_transcriber
    cd ai_transcriber
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Run the setup script:**
    This script will download and build Whisper.cpp, download the required AI models, and install Python dependencies.
    ```bash
    bash scripts/setup.sh
    ```
    This process may take some time, especially the model downloads and the initial build of Whisper.cpp.

## Usage

To transcribe an audio file, run the `main.py` script with the path to your audio file as an argument.

1.  **Place your audio files** in the `audio_files` directory (or any other location).

2.  **Run the transcription:**
    ```bash
    python main.py path/to/your/audio.mp3
    ```
    For example:
    ```bash
    python main.py audio_files/my_meeting.m4a
    ```

3.  **Find the output:**
    The script will process the audio, and the final transcript will be saved as a `.txt` file in the `transcripts` directory.

## Project Structure

```
/
├── audio_files/      # Place your input audio files here
├── models/           # Stores the downloaded AI models (Whisper and LLM)
├── transcripts/      # Output directory for the generated transcripts
├── scripts/
│   ├── setup.sh      # Setup script for dependencies and models
│   └── whisper.cpp/  # Git submodule for the whisper.cpp project
├── main.py           # The main application script
├── requirements.txt  # Python dependencies
└── README.md         # This file
```
