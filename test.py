import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")

res = requests.post(f"{API_URL}/chat/", json={"query": "Tell me about Nikhil"})
print(res.json())
