# AI Transcriber with Speaker Diarization

This project provides a robust, containerized command-line tool to transcribe audio files and identify different speakers using `whisperx`. It is designed for fast, accurate, and private transcription on a local machine with an NVIDIA GPU.

The use of Docker is strongly recommended to ensure a consistent and stable environment, avoiding the complex dependency and compatibility issues common in AI/ML projects.

## Features

-   **High-Quality Transcription**: Utilizes the Whisper ASR model for accurate speech-to-text.
-   **Speaker Diarization**: Identifies and assigns speaker labels (e.g., `SPEAKER_01`, `SPEAKER_02`) to the transcript.
-   **GPU Acceleration**: Leverages an NVIDIA GPU via a containerized CUDA environment for fast processing.
-   **Simple & Private**: A straightforward command-line interface runs entirely on your local machine.
-   **Reproducible Environment**: The provided `Dockerfile` guarantees a working setup, regardless of your host system's configuration.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

-   [Docker](https://docs.docker.com/get-docker/)
-   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to allow Docker to access your GPU.
-   An NVIDIA GPU.
-   `git` for cloning the repository.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sdmeers/ai_transcriber
cd ai_transcriber
```

### 2. Hugging Face Setup

This tool requires models from Hugging Face, some of which are "gated" and require you to accept their terms of use.

**A. Get Your Token:**
- Go to [https://hf.co/settings/tokens](https://hf.co/settings/tokens) and create a new access token.
- Grant it the **"Read"** role.
- In your terminal, export this token as an environment variable. **You will need to do this in every new terminal session.**
  ```bash
  export HF_TOKEN="hf_YOUR_TOKEN_HERE"
  ```

**B. Accept Model Terms:**
- You must log in to Hugging Face and accept the terms for the two models used by the diarization pipeline.
  1.  Visit [https://hf.co/pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) and agree to the terms.
  2.  Visit [https://hf.co/pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0) and agree to the terms.

### 3. Build the Docker Image

This command builds the container image, installing all necessary system libraries, Python packages, and AI models. This may take several minutes.

```bash
docker build -t ai-transcriber .
```

### 4. Run the Transcription

You can now run the transcription on an audio file. The command below will start the container, run the script, and automatically clean up afterwards.

Place your audio files in the `audio_files` directory.

```bash
docker run --rm -it --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -v $(pwd):/app \
  ai-transcriber \
  python3 speaker_rec.py audio_files/your_audio_file.mp3
```

- The final transcript will be saved as `.json` and `.txt` files in the `transcripts` directory on your local machine.

**Explanation of the `docker run` command:**
- `--rm`: Automatically removes the container when it exits.
- `-it`: Runs the container in interactive mode to show you the progress.
- `--gpus all`: **(Crucial)** Grants the container access to your NVIDIA GPU.
- `-e HF_TOKEN=$HF_TOKEN`: Securely passes your Hugging Face token into the container.
- `-v $(pwd):/app`: Mounts your current project directory into the container, allowing it to read your audio files and write the transcripts back to your machine.

## Project Structure

```
/
├── audio_files/      # Place your input audio files here
├── transcripts/      # Output directory for the generated transcripts
├── speaker_rec.py    # The main application script for transcription and diarization
├── Dockerfile        # Defines the containerized application environment
├── requirements.txt  # Python dependencies (used within the Dockerfile)
└── README.md         # This file
```