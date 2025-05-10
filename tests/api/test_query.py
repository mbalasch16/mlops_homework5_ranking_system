from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

# 
def test_query_endpoint():
     # response = client.post("/similar_responses", json={"question": "What is the capital of France?"})
     # assert response.status_code == 200
     # assert response.json() == {"answers": ["These are test responses"]}
     response = client.post("/similar_responses", json={"question": "hindi movie"})
     assert response.status_code == 200
     json_data = response.json()
     assert "answers" in json_data
     assert isinstance(json_data["answers"], list)
     assert all("excerpt" in ans and "score" in ans for ans in json_data["answers"])
    # json_data = response.json()
    # assert "answers" in json_data
    # assert isinstance(json_data["answers"], list)
    # assert all("excerpt" in ans and "score" in ans for ans in json_data["answers"])
