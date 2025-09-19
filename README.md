# AI Transcriber and Diarizer

This project provides a robust, containerized command-line tool to transcribe audio files using `whisperx`. It can be run in two modes: transcription-only, or transcription with speaker diarization.

The use of Docker is strongly recommended to ensure a consistent and stable environment, avoiding the complex dependency and compatibility issues common in AI/ML projects.

## Features

-   **High-Quality Transcription**: Utilizes the Whisper ASR model for accurate speech-to-text with word-level timestamps.
-   **Speaker Diarization (Optional)**: Can identify and assign speaker labels (e.g., `SPEAKER_01`, `SPEAKER_02`) to the transcript.
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

### 2. Hugging Face Setup (For Speaker Diarization Only)

If you only need transcription, you can skip this step. For speaker diarization, the tool requires gated models from Hugging Face.

**A. Get Your Token:**
- Go to [https://hf.co/settings/tokens](https://hf.co/settings/tokens) and create a new access token with the **"Read"** role.
- In your terminal, export this token as an environment variable. **You will need to do this in every new terminal session.**
  ```bash
  export HF_TOKEN="hf_YOUR_TOKEN_HERE"
  ```

**B. Accept Model Terms:**
- Log in to Hugging Face and accept the terms for the two models used by the diarization pipeline:
  1.  [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
  2.  [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0)

### 3. Build the Docker Image

This command builds the container image, installing all necessary system libraries and Python packages. This may take several minutes.

```bash
docker build -t ai-transcriber .
```

### 4. Run the Application

Place your audio files in the `audio_files` directory. The final transcript will be saved as `.json` and `.txt` files in the `transcripts` directory.

#### Option 1: Transcription Only

For a fast transcript with timestamps but no speaker labels. This does not require a Hugging Face token.

```bash
docker run --rm -it --gpus all \
  -v $(pwd):/app \
  ai-transcriber \
  python3 speech_rec.py audio_files/your_audio_file.mp3
```

#### Option 2: Transcription with Speaker Diarization

For a detailed transcript with speaker labels (e.g., `SPEAKER_01`). **Requires the Hugging Face setup from Step 2.**

```bash
docker run --rm -it --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -v $(pwd):/app \
  ai-transcriber \
  python3 speaker_rec.py audio_files/your_audio_file.mp3
```

## Project Structure

```
/
├── audio_files/      # Place your input audio files here
├── transcripts/      # Output directory for the generated transcripts
├── speaker_rec.py    # Script for transcription with speaker diarization
├── speech_rec.py     # Script for transcription only
├── Dockerfile        # Defines the containerized application environment
└── README.md         # This file
```
```