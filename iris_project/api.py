"""
api.py
======
This is the BACKEND. Its only job is to load our trained model and answer
questions like "given these 4 measurements, what species is this flower?"

It's a small web server built with FastAPI. The Streamlit app (app.py)
will send it requests over the network (even if both run on your own
laptop, they still talk over HTTP -- that's the "API" part).

Run it like this (after you've run train_model.py at least once):
    uvicorn api:app --reload --port 8000

Then open http://localhost:8000/docs in your browser -- FastAPI
automatically generates an interactive page where you can try the API
without writing any code.

New to APIs? Two endpoints (URLs) are defined below:
- GET  /health   -> "are you alive, and is the model loaded?"
- POST /predict  -> "here are 4 numbers, tell me the species"
"""

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Create the FastAPI application. "app" is the object uvicorn looks for
# when we run `uvicorn api:app`.
app = FastAPI(title="Iris Prediction API")

# ---- Load the trained model when the server starts ----
# We do this once, up front, instead of on every request, because loading
# a model from disk is relatively slow -- no need to repeat it every time.
try:
    bundle = joblib.load("model.pkl")
    model = bundle["model"]
    target_names = bundle["target_names"]
except FileNotFoundError:
    # If someone forgot to run train_model.py first, we don't want the
    # whole server to crash -- we just remember that we have no model,
    # and return a clear error message later if /predict is called.
    model = None
    target_names = None


# ---- Define what a valid request looks like ----
# Pydantic models describe the "shape" of the data we expect. FastAPI uses
# this to automatically validate incoming requests and reject bad data
# (e.g. missing fields, wrong types) with a helpful error message.
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")
    # `...` means "this field is required"
    # `gt=0` means "must be greater than 0" (a flower can't have negative length!)


# ---- Define what our response looks like ----
class PredictionResponse(BaseModel):
    predicted_class: str          # e.g. "setosa"
    class_index: int              # e.g. 0
    probabilities: dict           # e.g. {"setosa": 0.95, "versicolor": 0.05, ...}


@app.get("/health")
def health():
    """
    A simple 'is this thing on?' check.
    Visiting http://localhost:8000/health in a browser should show you
    whether the server is running and whether the model loaded correctly.
    """
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    """
    The main endpoint. Takes 4 flower measurements and returns:
    - the predicted species name
    - the predicted species as a number (0, 1, or 2)
    - the model's confidence for each of the 3 possible species
    """
    if model is None:
        # This happens if model.pkl doesn't exist yet.
        raise HTTPException(
            status_code=500,
            detail="Model not found. Run train_model.py first to create model.pkl",
        )

    # scikit-learn expects a 2D array: a list of rows, where each row is
    # one flower's 4 measurements. We only have 1 flower, so it's a list
    # containing a single row.
    X = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])

    # model.predict() gives us the predicted class as a number (0, 1, or 2)
    pred_idx = int(model.predict(X)[0])

    # model.predict_proba() gives us the model's confidence for EACH class,
    # e.g. [0.95, 0.03, 0.02] -- these always add up to 1.0 (100%)
    probs = model.predict_proba(X)[0]

    return PredictionResponse(
        predicted_class=target_names[pred_idx],
        class_index=pred_idx,
        probabilities={name: float(p) for name, p in zip(target_names, probs)},
    )
