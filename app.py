from flask import Flask, render_template, request, jsonify
import os
import sys
import subprocess
from pathlib import Path
import warnings
import torch
import whisperx
import ollama
import json
import time

# Suppress specific UserWarnings from torchaudio
warnings.filterwarnings("ignore", message="torchaudio._backend.list_audio_backends has been deprecated")

app = Flask(__name__)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent
UPLOAD_FOLDER = PROJECT_ROOT / "audio_files"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg'}

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# Ensure necessary directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            filename = Path(file.filename).name # Basic sanitization
            filepath = UPLOAD_FOLDER / filename
            file.save(filepath)

            # Process the file and get results
            try:
                transcript_text, summary_text = process_audio_file(filepath)
                return jsonify({
                    "transcript": transcript_text,
                    "summary": summary_text
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    return render_template('index.html')

def process_audio_file(audio_path: Path):
    """
    The core processing logic adapted from speaker_rec.py.
    This function handles conversion, transcription, diarization, and summarization.
    """
    # --- Get config from environment or defaults ---
    model_size = os.getenv("WHISPER_MODEL", "medium.en")
    summarizer_model = os.getenv("SUMMARIZER_MODEL", "llama3.1:8b")
    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1")
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("Hugging Face token (HF_TOKEN) is not set in the environment.")

    converted_wav_path = None
    try:
        # --- 1. Audio Conversion ---
        converted_wav_path = convert_audio_for_whisper(audio_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"▶ Using device: {device}")

        # --- 2. Transcription ---
        print("▶ Loading Whisper model...")
        model = whisperx.load_model(model_size, device)
        audio = whisperx.load_audio(str(converted_wav_path))
        print(f"▶ Transcribing {audio_path.name}...")
        result = model.transcribe(audio)

        # --- 3. Alignment ---
        print("▶ Aligning word-level timestamps...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)

        # --- 4. Diarization ---
        print("▶ Running speaker diarization...")
        diarize_model = whisperx.diarize.DiarizationPipeline(
            model_name="pyannote/speaker-diarization-2.1",
            use_auth_token=hf_token, device=device
        )
        diarize_segments = diarize_model(audio)

        # --- 5. Assign Speakers ---
        print("▶ Assigning speakers to transcript...")
        result_diarized = whisperx.assign_word_speakers(diarize_segments, result_aligned)

        # --- 6. Format Transcript ---
        transcript_text = ""
        for seg in result_diarized["segments"]:
            speaker = seg.get('speaker', 'Unknown')
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            text = seg.get('text', '').strip()
            transcript_text += f"[{start:.2f}–{end:.2f}] {speaker}: {text}\n"

        # --- 7. Summarization ---
        summary = summarize_transcript(transcript_text, summarizer_model, ollama_host)

        # --- 7.5 Save outputs to files ---
        print(f"▶ Saving outputs to {TRANSCRIPTS_DIR}...")
        txt_out = TRANSCRIPTS_DIR / f"{audio_path.stem}.txt"
        summary_out = TRANSCRIPTS_DIR / f"{audio_path.stem}_summary.txt"

        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        print(f"✅ Transcript saved to {txt_out}")

        with open(summary_out, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"✅ Summary saved to {summary_out}")

        return transcript_text, summary

    finally:
        # --- 8. Cleanup ---
        if converted_wav_path and converted_wav_path.exists():
            print(f"▶ Cleaning up temporary file: {converted_wav_path.name}")
            converted_wav_path.unlink()
        # Clean up original upload too
        if audio_path and audio_path.exists():
            audio_path.unlink()


def convert_audio_for_whisper(audio_path: Path) -> Path:
    """Converts audio to 16-bit, 16kHz, mono WAV."""
    output_wav_path = TRANSCRIPTS_DIR / f"{audio_path.stem}_converted.wav"
    print(f"--- Converting {audio_path.name} for transcription ---")
    command = [
        "ffmpeg", "-i", str(audio_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        "-y", str(output_wav_path)
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error during audio conversion: {result.stderr}")
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
    return output_wav_path

def summarize_transcript(transcript_text: str, model: str, ollama_host: str) -> str:
    """Summarizes the transcript using Ollama."""
    print(f"--- Summarizing transcript using {model} at {ollama_host} ---")
    prompt = f"""
You are an expert AI assistant that specializes in summarizing meeting transcripts. Please provide a concise summary of the following text. Structure your summary with these sections:
- A brief, one-paragraph overview of the conversation.
- A bulleted list of the key topics discussed.
- A bulleted list of any action items or decisions that were made.

Here is the transcript:
---
{transcript_text}
---
    """
    try:
        client = ollama.Client(host=f"{ollama_host}:11434")
        response = client.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    except Exception as e:
        print(f"Error during summarization with Ollama: {e}")
        return f"Could not generate summary. Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
