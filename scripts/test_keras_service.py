import time
import json
import requests
import os

URL = os.environ.get("KERAS_SERVICE_URL", "http://127.0.0.1:8001")

def build_payload():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    feature_path = os.path.join(base, "feature_names.json")
    if os.path.exists(feature_path):
        with open(feature_path, "r") as f:
            features = json.load(f)
        # create a simple payload with average-like values (zeros)
        sample = {name: 0.0 for name in features}
    else:
        sample = {"feature_1": 0.0}
    return {"features": sample}


def main():
    payload = build_payload()
    health = requests.get(f"{URL}/health", timeout=5)
    print("health:", health.status_code, health.json())

    resp = requests.post(f"{URL}/predict", json=payload, timeout=10)
    print("predict:", resp.status_code, resp.json())


if __name__ == "__main__":
    main()
