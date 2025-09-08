import whisperx
import torch
import json
import os
import sys
import time

# ------------------------------
# Configuration
# ------------------------------
AUDIO_FILE = "example.wav"             # Path to your audio file
MODEL_SIZE = "medium.en"               # Whisper model size
HF_TOKEN = os.getenv("HF_TOKEN")       # Hugging Face token from environment
JSON_OUT = "transcript.json"
TXT_OUT = "transcript.txt"

# ------------------------------
# Checks
# ------------------------------
if not os.path.exists(AUDIO_FILE):
    sys.exit(f"❌ Error: Audio file '{AUDIO_FILE}' not found.")

if HF_TOKEN is None:
    sys.exit("❌ Error: Hugging Face token not found. Run `export HF_TOKEN=your_token` before running.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"▶ Using device: {device}")

# ------------------------------
# 1. Load Whisper model
# ------------------------------
print("▶ Loading Whisper model...")
model = whisperx.load_model(MODEL_SIZE, device)

# ------------------------------
# 2. Transcribe audio
# ------------------------------
print(f"▶ Transcribing {AUDIO_FILE}...")
result = model.transcribe(AUDIO_FILE)

# ------------------------------
# 3. Align with WhisperX
# ------------------------------
print("▶ Aligning word-level timestamps...")
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device
)
result_aligned = whisperx.align(
    result["segments"], model_a, metadata, AUDIO_FILE, device
)

# ------------------------------
# 4. Speaker diarization
# ------------------------------
print("▶ Running speaker diarization...")
diarize_model = whisperx.DiarizationPipeline(
    use_auth_token=HF_TOKEN, device=device
)
diarize_segments = diarize_model(AUDIO_FILE)

# ------------------------------
# 5. Combine transcript + diarization
# ------------------------------
print("▶ Assigning speakers to transcript...")
result_diarized = whisperx.assign_word_speakers(diarize_segments, result_aligned)

# ------------------------------
# 6. Build structured output
# ------------------------------
output = {
    "metadata": {
        "audio_file": AUDIO_FILE,
        "model": MODEL_SIZE,
        "device": device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    },
    "transcript": []
}

for seg in result_diarized["segments"]:
    entry = {
        "start": round(seg["start"], 2),
        "end": round(seg["end"], 2),
        "speaker": seg.get("speaker", "Unknown"),
        "text": seg["text"].strip()
    }
    output["transcript"].append(entry)

# ------------------------------
# 7. Save JSON + TXT in one go
# ------------------------------
print("▶ Saving outputs...")
with open(JSON_OUT, "w", encoding="utf-8") as f_json, \
     open(TXT_OUT, "w", encoding="utf-8") as f_txt:

    json.dump(output, f_json, indent=2, ensure_ascii=False)

    for seg in output["transcript"]:
        line = f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['speaker']}: {seg['text']}\n"
        f_txt.write(line)

print(f"✅ Transcript saved to {JSON_OUT} and {TXT_OUT}")