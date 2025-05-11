import sounddevice as sd
import numpy as np
import requests
import sys
import queue
import whisper
import pyttsx3

engine = pyttsx3.init()

# Parameters
samplerate = 16000
channels = 1
threshold = 0.0003  # Silence threshold (normalized)
silence_duration = 2  # Seconds of silence before stopping
chunk_duration = 0.5  # Duration of each read chunk in seconds

# Load Whisper model
print("Loading Whisper model (this might take a moment)...")
model = whisper.load_model("base")
print("Model loaded successfully!")
print("Speak to start recording. Press Ctrl+C to stop.")


def record_until_silence():
    q_audio = queue.Queue()
    frames = []
    silence_chunk_count = 0
    max_silent_chunks = int(silence_duration / chunk_duration)

    def callback(indata, frames_, time, status):
        volume_norm = np.linalg.norm(indata) / len(indata)
        q_audio.put((indata.copy(), volume_norm))

    with sd.InputStream(samplerate=samplerate, channels=channels, dtype='float32',
                        blocksize=int(samplerate * chunk_duration), callback=callback):
        print("Listening...")
        while True:
            data_chunk, volume = q_audio.get()
            frames.append(data_chunk)

            if volume < threshold:
                silence_chunk_count += 1
                if silence_chunk_count >= max_silent_chunks:
                    print("Silence detected. Stopping recording.")
                    break
            else:
                silence_chunk_count = 0

    audio_data = np.concatenate(frames, axis=0)
    return audio_data.flatten()


def get_speech():
    try:
        audio = record_until_silence()

        print("Transcribing...")
        result = model.transcribe(audio, fp16=False)
        transcription = result["text"]
        print("Transcription:", transcription)

        # Send transcription
        try:
            print(f"Attempting to send command...")
            response = requests.post(
                'http://192.168.148.90:8000/command',
                json={"message": transcription},
                timeout=3000
            )
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.text}")
            # return transcription
        except requests.exceptions.ConnectionError as e:
            print("Connection Error: Could not connect to the server.")
            print(f"Details: {str(e)}")
        except requests.exceptions.Timeout:
            print("Timeout Error: Server took too long to respond.")
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {str(e)}")

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)


def speak(text):
    # Loop through voices to find a German voice
    voices = engine.getProperty('voices')
    for voice in voices:
        if "german" in voice.name.lower() or "deutsch" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break

    # Set properties for better voice quality
    engine.setProperty('rate', 175) # Speed of speech
    engine.setProperty('volume', 0.9) # Volume (0.0 to 1.0)

    # Test speaking in German
    engine.say(text)
    engine.runAndWait()


if __name__ == '__main__':
    speak(get_speech())