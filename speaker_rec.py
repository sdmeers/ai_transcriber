import whisperx
import torch
import json
import os
import sys
import time
import argparse
import subprocess
import warnings
from pathlib import Path

# Suppress specific UserWarnings
warnings.filterwarnings("ignore", message="torchaudio._backend.list_audio_backends has been deprecated")

# --- Configuration ---
# Define paths relative to the script location
PROJECT_ROOT = Path(__file__).parent
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"

def convert_audio_for_whisper(audio_path: Path) -> Path:
    """
    Converts an audio file to the format required by whisper.cpp
    (16-bit, 16kHz, mono WAV), skipping if it already exists.
    """
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    output_wav_path = TRANSCRIPTS_DIR / f"{audio_path.stem}_converted.wav"

    if output_wav_path.exists():
        print(f"--- Converted file {output_wav_path.name} already exists, skipping conversion. ---")
        return output_wav_path

    print(f"--- Converting {audio_path.name} for transcription ---")
    command = [
        "ffmpeg",
        "-i", str(audio_path),
        "-ar", "16000",      # 16kHz sample rate
        "-ac", "1",          # Mono audio
        "-c:a", "pcm_s16le", # 16-bit PCM audio format
        "-y",                # Overwrite output file if it exists
        str(output_wav_path)
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error during audio conversion with ffmpeg:")
        print(result.stderr)
        sys.exit(1)

    print(f"Successfully converted audio to {output_wav_path}")
    return output_wav_path

def main():
    """Main function to run the transcription and diarization process."""
    parser = argparse.ArgumentParser(description="Transcribe an audio file and perform speaker diarization.")
    parser.add_argument("audio_file", type=Path, help="Path to the audio file to transcribe (e.g., mp3, wav, m4a).")
    parser.add_argument("--model", type=str, default="medium.en", help="Whisper model size (e.g., tiny, base, small, medium, large).")
    args = parser.parse_args()

    # ------------------------------
    # Configuration
    # ------------------------------
    AUDIO_FILE_PATH = args.audio_file
    MODEL_SIZE = args.model
    HF_TOKEN = os.getenv("HF_TOKEN")
    JSON_OUT = TRANSCRIPTS_DIR / f"{AUDIO_FILE_PATH.stem}.json"
    TXT_OUT = TRANSCRIPTS_DIR / f"{AUDIO_FILE_PATH.stem}.txt"

    # ------------------------------
    # Checks
    # ------------------------------
    if not AUDIO_FILE_PATH.is_file():
        sys.exit(f"❌ Error: Audio file not found at {AUDIO_FILE_PATH}")

    if HF_TOKEN is None:
        sys.exit("❌ Error: Hugging Face token not found. Run `export HF_TOKEN=your_token` before running.")

    # ------------------------------
    # Audio Conversion
    # ------------------------------
    converted_wav_path = None
    try:
        # Convert audio to WAV format required by whisperX
        converted_wav_path = convert_audio_for_whisper(AUDIO_FILE_PATH)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"▶ Using device: {device}")

        # ------------------------------
        # 1. Load Whisper model and audio
        # ------------------------------
        print("▶ Loading Whisper model...")
        model = whisperx.load_model(MODEL_SIZE, device)
        print(f"▶ Loading audio: {converted_wav_path.name}...")
        audio = whisperx.load_audio(str(converted_wav_path))


        # ------------------------------
        # 2. Transcribe audio
        # ------------------------------
        print(f"▶ Transcribing...")
        result = model.transcribe(audio)

        # ------------------------------
        # 3. Align with WhisperX
        # ------------------------------
        print("▶ Aligning word-level timestamps...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result_aligned = whisperx.align(
            result["segments"], model_a, metadata, audio, device
        )

        # ------------------------------
        # 4. Speaker diarization
        # ------------------------------
        print("▶ Running speaker diarization...")
        diarize_model = whisperx.diarize.DiarizationPipeline(
            model_name="pyannote/speaker-diarization-2.1",
            use_auth_token=HF_TOKEN, device=device
        )
        diarize_segments = diarize_model(audio)

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
                "audio_file": str(AUDIO_FILE_PATH),
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
        print(f"▶ Saving outputs to {TRANSCRIPTS_DIR}...")
        TRANSCRIPTS_DIR.mkdir(exist_ok=True)
        with open(JSON_OUT, "w", encoding="utf-8") as f_json, \
             open(TXT_OUT, "w", encoding="utf-8") as f_txt:

            json.dump(output, f_json, indent=2, ensure_ascii=False)

            for seg in output["transcript"]:
                line = f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['speaker']}: {seg['text']}\n"
                f_txt.write(line)

        print(f"✅ Transcript saved to {JSON_OUT} and {TXT_OUT}")

    finally:
        # ------------------------------
        # 8. Cleanup
        # ------------------------------
        if converted_wav_path and converted_wav_path.exists():
            print(f"▶ Cleaning up temporary file: {converted_wav_path.name}")
            converted_wav_path.unlink()

if __name__ == "__main__":
    main()
