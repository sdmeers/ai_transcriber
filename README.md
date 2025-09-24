# AI Transcriber, Diarizer, and Summarizer

This project provides a robust, containerized command-line tool to transcribe audio files, identify speakers, and generate a concise summary of the conversation. It uses `whisperx` for transcription, `pyannote` for diarization, and a local LLM via `Ollama` for summarization.

The use of Docker is strongly recommended to ensure a consistent and stable environment, avoiding the complex dependency and compatibility issues common in AI/ML projects.

## Features

-   **High-Quality Transcription**: Utilizes the Whisper ASR model for accurate speech-to-text with word-level timestamps.
-   **Speaker Diarization**: Identifies and assigns speaker labels (e.g., `SPEAKER_01`, `SPEAKER_02`) to the transcript.
-   **AI-Powered Summarization**: Uses a local LLM (e.g., Llama 3.1) to generate a structured summary of the transcript, including key topics and action items.
-   **GPU Acceleration**: Leverages an NVIDIA GPU via a containerized CUDA environment for fast processing.
-   **Simple & Private**: A straightforward command-line interface that runs entirely on your local machine.
-   **Reproducible Environment**: The provided `Dockerfile` guarantees a working setup, with all models pre-downloaded for fast execution.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

-   [Docker](https://docs.docker.com/get-docker/) (Docker Desktop is recommended)
-   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to allow Docker to access your GPU.
-   An NVIDIA GPU.
-   `git` for cloning the repository.
-   [Ollama](https://ollama.com/download) installed and running on your host machine.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sdmeers/ai_transcriber
cd ai_transcriber
```

### 2. Hugging Face Setup (For Speaker Diarization)

Speaker diarization requires gated models from Hugging Face.

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

### 3. Ollama Setup (For Summarization)

**A. Configure Ollama Host:**
For Docker Desktop to connect to Ollama, you must configure Ollama to listen on all network interfaces.
```bash
# For Linux with systemd
sudo systemctl edit ollama.service
```
Add the following content, then save and close:
```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
```
Finally, restart the service:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

**B. Pull the Summarization Model:**
Pre-download the LLM to avoid a long delay on the first run.
```bash
ollama pull llama3.1:8b
```

### 4. Build the Docker Image

This command builds the container image, installing all dependencies and pre-downloading the necessary AI models. This may take several minutes.

**Important:** You must provide your Hugging Face token as a build argument.

```bash
docker build --build-arg HF_TOKEN=$HF_TOKEN -t ai-transcriber .
```

### 5. Run the Application

Place your audio files in the `audio_files` directory. The final transcript, diarized text, and summary will be saved in the `transcripts` directory.

The easiest way to run the application is with the provided helper script.

**A. Make the script executable (one-time setup):**
```bash
chmod +x run_transcriber.sh
```

**B. Run the transcriber:**
```bash
./run_transcriber.sh audio_files/your_audio_file.mp3
```
The script handles passing the necessary arguments to the Docker container. It will create three files:
- `your_audio_file.json` (Detailed data)
- `your_audio_file.txt` (Formatted transcript with speakers)
- `your_audio_file_summary.txt` (AI-generated summary)

#### Manual Docker Command

If you prefer to run the `docker` command directly:

```bash
docker run --rm -it \
  -e HF_TOKEN=$HF_TOKEN \
  --gpus all \
  -v $(pwd):/app \
  ai-transcriber \
  python3 speaker_rec.py audio_files/your_audio_file.mp3 --ollama_host http://host.docker.internal
```
**Note:** The `--ollama_host http://host.docker.internal` argument is crucial for users on Docker Desktop (macOS, Windows, or Linux).

## Project Structure

```
/
├── audio_files/      # Place your input audio files here
├── transcripts/      # Output directory for generated files
├── run_transcriber.sh # Helper script to run the application
├── speaker_rec.py    # Script for transcription, diarization, and summarization
├── speech_rec.py     # Script for transcription only (no summarization)
├── Dockerfile        # Defines the containerized application environment
└── README.md         # This file
```
```