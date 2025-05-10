import requests

url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json",
}
data = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "user", "content": "Hello! Can you tell me a fun fact about space?"}
    ],
    "temperature": 0.7,
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
