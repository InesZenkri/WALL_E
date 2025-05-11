import imp
import re
import sounddevice as sd
import numpy as np
import requests
import sys
import queue
import time
import whisper
import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()

# Audio parameters
samplerate = 16000
channels = 1
threshold = 0.0003      # silence threshold
silence_duration = 2    # seconds of silence before stopping
chunk_duration = 0.5    # length of each audio chunk (s)

# Load Whisper model
print("Loading Whisper model (this might take a moment)...")
model = whisper.load_model("base")
print("Model loaded. Speak to start recording. Ctrl+C to stop.")

def record_until_silence():
    """Record from mic until `silence_duration` seconds of silence."""
    q_audio = queue.Queue()
    frames = []
    silence_count = 0
    max_silent_chunks = int(silence_duration / chunk_duration)

    def callback(indata, frames_, time, status):
        volume = np.linalg.norm(indata) / len(indata)
        q_audio.put((indata.copy(), volume))

    with sd.InputStream(samplerate=samplerate,
                        channels=channels,
                        dtype='float32',
                        blocksize=int(samplerate * chunk_duration),
                        callback=callback):
        print("Listening…")
        start = time.time() + 5
        while True:
            data, vol = q_audio.get()
            frames.append(data)
            if start < time.time():
                start = time.time() + 5
                break
            if vol < threshold :
                silence_count += 1
                if silence_count >= max_silent_chunks:
                    print("Silence detected.")
                    break
            else:
                silence_count = 0

    if not frames:
        return None
    return np.concatenate(frames, axis=0).flatten()

def get_speech_and_send():
    """Record, transcribe, send transcription if any, and return text."""
    try:
        audio = record_until_silence()
        if audio is None:
            print("No audio captured.")
            return ""

        print("Transcribing…")
        result = model.transcribe(audio, fp16=False)
        text = result["text"].strip()
        print("Transcription:", text)

        if text:
            try:
                print("Sending to server…")
                r = requests.post(
                    "http://localhost:8000/command",
                    json={"message": text},
                    timeout=10
                )
                print(f"Server response: {r.status_code} – {r.text}")
            except requests.exceptions.RequestException as e:
                print("Failed to send:", e)

        return text

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print("Error during recording/transcription:", e)
        sys.exit(1)

def speak(text):
    """Speak the text in German voice (if available)."""
    if not text:
        return
    return

    voices = engine.getProperty('voices')
    for v in voices:
        if "german" in v.name.lower() or "deutsch" in v.name.lower():
            engine.setProperty('voice', v.id)
            break

    engine.setProperty('rate', 175)
    engine.setProperty('volume', 0.9)
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    while True:
        msg = get_speech_and_send()
        speak(msg)
