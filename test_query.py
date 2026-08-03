import requests

API_URL = "http://localhost:8000/api/v1/query"
response = requests.post(API_URL, json={"query": "explain me the architecture of transformer", "session_id": "test_123"})
print(response.status_code)
print(response.json())
