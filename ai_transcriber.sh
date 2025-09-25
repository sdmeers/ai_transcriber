#!/bin/bash

# Script to run the AI Transcriber web application and open it in a browser.

# --- Configuration ---
IMAGE_NAME="ai-transcriber"
CONTAINER_NAME="ai-transcriber-app"
APP_PORT="5000"
OLLAMA_HOST="http://host.docker.internal"

# 1. Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "❌ Error: Hugging Face token is not set."
  echo "Please set it by running: export HF_TOKEN=your_token_here"
  exit 1
fi

echo "▶️ Starting AI Transcriber..."

# 2. Stop and remove any existing container to avoid conflicts
echo "--- (1/3) Cleaning up old containers... ---"
docker stop $CONTAINER_NAME >/dev/null 2>&1 || true
docker rm $CONTAINER_NAME >/dev/null 2>&1 || true

# 3. Run the new container in detached mode
echo "--- (2/3) Running Docker container in background... ---"
docker run \
  --rm \
  -d \
  --gpus all \
  -v "$(pwd):/app" \
  -p "$APP_PORT:$APP_PORT" \
  -e HF_TOKEN \
  -e OLLAMA_HOST="$OLLAMA_HOST" \
  --name $CONTAINER_NAME \
  $IMAGE_NAME

if [ $? -ne 0 ]; then
  echo "❌ Error: Docker container failed to start."
  echo "Make sure you have built the image first, e.g., docker build -t $IMAGE_NAME ."
  exit 1
fi

# 4. Wait for the server to initialize
echo "--- (3/3) Waiting for application to start... ---"
sleep 8 # Time for the server and models to load

APP_URL="http://localhost:$APP_PORT"
echo "✅ Success! Opening application at $APP_URL"

# 5. Open in Google Chrome
cmd.exe /c start "$APP_URL"

echo "
Application is running in the background.
To see logs, run: docker logs -f $CONTAINER_NAME
To stop the application, run: docker stop $CONTAINER_NAME
"
