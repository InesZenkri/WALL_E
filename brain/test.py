import os

import requests

# Set OPENROUTER_API_KEY in your environment (see .env.example). Never commit the key.
api_key = os.environ["OPENROUTER_API_KEY"]

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}
data = {
    "model": "openai/gpt-3.5-turbo",
    "messages": [
        {"role": "user", "content": "Hello! Can you tell me a fun fact about space?"}
    ],
    "temperature": 0.7,
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
