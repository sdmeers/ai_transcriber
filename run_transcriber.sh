#!/bin/bash

# Check if an audio file path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_audio_file>"
  echo "Example: $0 audio_files/my_meeting.mp3"
  exit 1
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN environment variable is not set."
  echo "Please set it before running the script: export HF_TOKEN=your_token"
  exit 1
fi

AUDIO_FILE_PATH="$1"

docker run --rm -it -e HF_TOKEN --gpus all -v "$(pwd):/app" ai-transcriber python3 speaker_rec.py "$AUDIO_FILE_PATH" --ollama_host http://host.docker.internal
