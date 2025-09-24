# Use an official NVIDIA CUDA runtime image as a parent image.
# This image comes with CUDA 12.1 and cuDNN 8 pre-installed on Ubuntu 22.04.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables to make the build non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies required for Python, Git, and audio processing
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
ENV VENV_PATH=/opt/venv
RUN python3.11 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support first to ensure compatibility
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install whisperx from its official repository and other dependencies
# This ensures we get the latest version that works with the libraries above
RUN pip install \
    git+https://github.com/m-bain/whisperx.git \
    llama-cpp-python \
    ollama

# Set the working directory inside the container
WORKDIR /app

# --- Pre-download models during build ---
# Set HF_TOKEN as a build argument for pyannote diarization model
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Create a temporary script to download models
RUN python3 -c "import whisperx; import os; from whisperx.diarize import DiarizationPipeline; \
    print('Downloading WhisperX model (medium.en)...'); \
    whisperx.load_model('medium.en', 'cpu', compute_type='float32'); \
    print('Downloading WhisperX alignment model (en)...'); \
    whisperx.load_align_model('en', 'cpu'); \
    print('Downloading Pyannote diarization model...'); \
    DiarizationPipeline(model_name='pyannote/speaker-diarization-2.1', use_auth_token=os.getenv('HF_TOKEN'), device='cpu'); \
    print('All models pre-downloaded.')"

# Copy the rest of your application's code into the container
COPY . .

# The container is now ready. The user will run the script via `docker run`.
