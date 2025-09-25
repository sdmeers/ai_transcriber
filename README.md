# AI Transcriber, Diarizer, and Summarizer

This project provides a robust, containerized web application to transcribe audio files, identify speakers, and generate a concise summary of the conversation. It features a simple web interface for uploading files and viewing results.

The application uses `whisperx` for transcription, `pyannote` for diarization, and a local LLM via `Ollama` for summarization. The entire stack runs in a Docker container for a stable, easy-to-manage, and private environment.

## Features

-   **Simple Web Interface**: Upload audio files and view transcripts and summaries directly in your browser.
-   **High-Quality Transcription**: Utilizes Whisper models for accurate speech-to-text with word-level timestamps.
-   **Speaker Diarization**: Identifies and assigns speaker labels (e.g., `SPEAKER_01`, `SPEAKER_02`) to the transcript.
-   **AI-Powered Summarization**: Uses a local LLM (e.g., Llama 3.1) to generate a structured summary.
-   **GPU Acceleration**: Leverages an NVIDIA GPU via a containerized CUDA environment for fast processing.
-   **Private & Local**: All processing happens on your local machine.
-   **Reproducible Environment**: The `Dockerfile` guarantees a working setup, with all models pre-downloaded for fast execution.

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
docker build --build-arg HF_TOKEN -t ai-transcriber .
```

### 5. Run the Application

The easiest way to run the application is with the provided helper script `ai_transcriber.sh`.

**A. Make the script executable (one-time setup):**
```bash
chmod +x ai_transcriber.sh
```

**B. Run the application:**
```bash
./ai_transcriber.sh
```
This script will start the Docker container in the background and automatically open the web interface in your browser at `http://localhost:5000`.

-   To **see the application logs**, run: `docker logs -f ai-transcriber-app`
-   To **stop the application**, run: `docker stop ai-transcriber-app`

### 6. Using the Web Interface

1.  Once the page loads, click the "Choose File" button to select an audio file (`.mp3`, `.wav`, `.m4a`).
2.  Click "Transcribe File".
3.  A loading indicator will appear. Processing can take several minutes depending on the file size and your hardware.
4.  Once complete, the transcription and a summary will appear on the page. The output files (`.txt`) will also be saved to the `transcripts` directory.

## Project Structure

```
/
├── audio_files/      # Temporary storage for uploads
├── transcripts/      # Output directory for generated .txt files
├── static/           # CSS for the web interface
├── templates/        # HTML for the web interface
├── app.py            # The Flask web application
├── ai_transcriber.sh # Helper script to run the application
├── Dockerfile        # Defines the containerized application environment
└── README.md         # This file
```
