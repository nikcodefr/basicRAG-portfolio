import requests

res = requests.post("http://localhost:8000/chat/", json={"query": "Tell me about Nikhil"})
print(res.json())
