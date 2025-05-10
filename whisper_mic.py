import sounddevice as sd
import numpy as np
import whisper
import requests
import sys

# Load Whisper model - using "base" instead of "large"
print("Loading Whisper model (this might take a moment)...")
model = whisper.load_model("base")  
samplerate = 16000
channels = 1

print("Model loaded successfully!")
print("Press Ctrl+C to stop.")

try:
    while True:
        duration = float(input("\nHow many seconds to record? (e.g., 5): "))
        print(f"Recording for {duration} seconds... Speak now!")
        audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=channels, dtype='int16')
        sd.wait()
        print("Transcribing...")
        audio = audio.flatten().astype(np.float32) / 32768.0
        result = model.transcribe(audio, fp16=False)
        transcription = result["text"]
        print("Transcription:", transcription)
        
        # Send transcription as JSON with the correct format
        try:
            print(f"Attempting to send to http://192.168.1.104:8000/command...")
            response = requests.post(
                'http://192.168.1.104:8000/command',
                json={"message": transcription},  # Format as required by FastAPI
                timeout=3000
            )
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.text}")
        except requests.exceptions.ConnectionError as e:
            print(f"Connection Error: Could not connect to the server. Please check if the server is running at 192.168.1.104:8000")
            print(f"Detailed error: {str(e)}")
        except requests.exceptions.Timeout as e:
            print(f"Timeout Error: The server took too long to respond")
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {str(e)}")
            
except KeyboardInterrupt:
    print("\nStopped.")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
    sys.exit(1)