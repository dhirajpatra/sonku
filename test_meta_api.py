import os
import requests

# Retrieve API Key from .env / environment
api_key = os.environ.get("MODEL_API_KEY")
url = "https://api.meta.ai/v1/responses"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

payload = {
    "model": "muse-spark-1.3-contributor",
    "input": [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Hello! Confirm your connection to Ritchie Mini."
                }
            ]
        }
    ],
    "stream": False
}

try:
    response = requests.post(url, headers=headers, json=payload, timeout=(5, 12))
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(response.json())
except Exception as e:
    print(f"Connection Error: {e}")