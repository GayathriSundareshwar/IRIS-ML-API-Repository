from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import json

app = FastAPI()

# Paths
BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")

# Load model
model = joblib.load(MODEL_PATH)

# Load metadata
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

target_names = metadata["target_names"]

# Fake in-memory prediction history
prediction_history = []

# -----------------------------
# Request Schemas
# -----------------------------

class IrisRequest(BaseModel):
    features: list[float]

class MetadataUpdateRequest(BaseModel):
    model_owner: str
    model_version: str

# -----------------------------
# GET
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Iris Prediction API Running"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/history")
def get_history(class_name: str = None):

    if class_name:
        filtered = [
            item for item in prediction_history
            if item["class_name"] == class_name
        ]

        return {
            "filtered_history": filtered
        }

    return {
        "prediction_history": prediction_history
    }

# -----------------------------
# POST
# -----------------------------

@app.post("/predict")
def predict(request: IrisRequest):

    prediction = model.predict([request.features])[0]

    class_name = target_names[prediction]

    result = {
        "features": request.features,
        "prediction": int(prediction),
        "class_name": class_name
    }

    prediction_history.append(result)
    print("Prediction history:", prediction_history)
    return result

# -----------------------------
# PUT
# -----------------------------

@app.put("/metadata")
def update_metadata(request: MetadataUpdateRequest):

    metadata["model_owner"] = request.model_owner
    metadata["model_version"] = request.model_version

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "message": "Metadata updated",
        "metadata": metadata
    }

# -----------------------------
# DELETE
# -----------------------------

@app.delete("/history")
def delete_history():

    prediction_history.clear()

    return {
        "message": "Prediction history deleted"
    }   