# Iris Classification API

A machine learning inference API built using:
- FastAPI
- Scikit-learn
- Docker
- Kubernetes

This project trains an Iris flower classification model and serves predictions through a REST API.

---

# Project Structure

```text
.
├── app.py
├── train.py
├── requirements.txt
├── Dockerfile
├── deployment.yaml
├── service.yaml
├── README.md
└── artifacts/
    ├── model.pkl
    ├── metadata.json
    └── metrics.json
```

---

# Train Model

Run:

```bash
python train.py
```

This will:
- train Logistic Regression model
- save model artifact
- save metrics
- save metadata

---

# Run FastAPI Locally

Start API server:

```bash
uvicorn app:app --reload
```

Swagger docs:

```text
http://127.0.0.1:8000/docs
```

---

# Example Prediction Request

POST `/predict`

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Example response:

```json
{
  "prediction": 0,
  "class_name": "setosa"
}
```

---

# Build Docker Image

```bash
docker build -t iris-api .
```

---

# Run Docker Container

```bash
docker run -p 8000:8000 iris-api
```

---

# Kubernetes Deployment

Apply deployment:

```bash
kubectl apply -f deployment.yaml
```

Apply service:

```bash
kubectl apply -f service.yaml
```

---

# Technologies Used

- Python
- FastAPI
- Scikit-learn
- Docker
- Kubernetes
- Uvicorn