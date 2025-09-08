import argparse
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
# Define paths relative to the script location
PROJECT_ROOT = Path(__file__).parent
WHISPER_CPP_DIR = PROJECT_ROOT / "scripts" / "whisper.cpp"
WHISPER_CLI_PATH = WHISPER_CPP_DIR / "build" / "bin" / "whisper-cli"
WHISPER_MODEL_PATH = PROJECT_ROOT / "models" / "ggml-base.en.bin"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"

def check_dependencies():
    """Check for required executables and models."""
    if not WHISPER_CLI_PATH.is_file():
        print(f"Error: whisper-cli not found at {WHISPER_CLI_PATH}")
        print("Please ensure you have run the setup.sh script successfully.")
        sys.exit(1)
    if not WHISPER_MODEL_PATH.is_file():
        print(f"Error: Whisper model not found at {WHISPER_MODEL_PATH}")
        print("Please ensure you have run the setup.sh script successfully.")
        sys.exit(1)

def convert_audio_for_whisper(audio_path: Path) -> Path:
    """
    Converts an audio file to the format required by whisper.cpp
    (16-bit, 16kHz, mono WAV).
    """
    print(f"--- Converting {audio_path.name} for transcription ---")
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    output_wav_path = TRANSCRIPTS_DIR / f"{audio_path.stem}_converted.wav"

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

def transcribe_audio(wav_path: Path) -> str:
    """
    Transcribes the given WAV file using whisper.cpp.
    It tells whisper.cpp to output a .txt file and then reads it.
    """
    print(f"--- Transcribing {wav_path.name} with whisper.cpp ---")

    command = [
        str(WHISPER_CLI_PATH),
        "-m", str(WHISPER_MODEL_PATH),
        "-f", str(wav_path),
        "--output-txt"  # Ask whisper.cpp to generate a .txt file
    ]

    # Running from the root of the project
    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"Error during transcription with whisper.cpp:")
        print(result.stderr)
        sys.exit(1)

    # The output file will be named based on the input wav, e.g., "audio.wav.txt"
    # and placed in the directory where the command was run.
    transcript_path = PROJECT_ROOT / wav_path.with_suffix('.wav.txt').name
    
    if not transcript_path.is_file():
        print(f"Error: Transcript file not found at {transcript_path}")
        print("Whisper.cpp may have failed silently. Check its output:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    transcript_text = transcript_path.read_text().strip()

    # --- Cleanup ---
    transcript_path.unlink() # Delete the .txt file
    wav_path.unlink()        # Delete the temporary .wav file

    print("--- Transcription complete ---")
    return transcript_text

def main():
    """Main function to run the transcription process."""
    parser = argparse.ArgumentParser(description="Transcribe an audio file using whisper.cpp.")
    parser.add_argument("audio_file", type=Path, help="Path to the audio file to transcribe (e.g., mp3, wav, m4a).")
    args = parser.parse_args()

    if not args.audio_file.is_file():
        print(f"Error: Audio file not found at {args.audio_file}")
        sys.exit(1)

    check_dependencies()

    try:
        converted_wav = convert_audio_for_whisper(args.audio_file)
        transcript = transcribe_audio(converted_wav)

        print("\n--- TRANSCRIPT ---")
        print(transcript)

        # Save the final transcript to a file
        final_transcript_path = TRANSCRIPTS_DIR / f"{args.audio_file.stem}.txt"
        final_transcript_path.write_text(transcript)
        print(f"\nFull transcript saved to: {final_transcript_path}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
