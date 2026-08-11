"""
app.py
======
This is the FRONTEND -- the part the user actually sees and interacts
with, built entirely with Streamlit.

IMPORTANT: this file does NOT contain any machine learning code. It just
draws sliders, and when you click "Predict", it sends those numbers to
our API (api.py) over the network and displays whatever the API sends
back. This separation (frontend vs. backend) is a very common pattern in
real-world apps.

Before running this, make sure the API is already running in another
terminal:
    uvicorn api:app --reload --port 8000

Then start this app:
    streamlit run app.py

Streamlit will open a page in your browser automatically.
"""

import streamlit as st
import requests   # lets us make HTTP calls to our API, just like a browser would
import pandas as pd

# The address where our FastAPI backend is running.
# If you deploy the API somewhere else later, just change this one line.
API_URL = "http://localhost:8000"

# Basic page setup -- title shown in the browser tab, and a flower emoji icon.
st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")

st.title("🌸 Iris Species Classifier")
st.write(
    "Move the sliders to describe a flower's measurements, then click "
    "**Predict** to see which Iris species it most likely is."
)

# ---------------------------------------------------------------------
# Sidebar: check whether the API is reachable before the user even tries
# to predict anything. This gives a helpful hint if something's wrong.
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Backend status")
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=3)
        health = health_response.json()
        if health.get("model_loaded"):
            st.success("API connected, model loaded ✅")
        else:
            st.error("API is running, but model.pkl was not found.\n\nRun `python train_model.py` first.")
    except requests.exceptions.RequestException:
        # This happens if the API isn't running at all
        st.error("Cannot reach the API.\n\nStart it with:\n\n`uvicorn api:app --reload --port 8000`")

    st.markdown("---")
    st.caption(f"API URL: `{API_URL}`")

# ---------------------------------------------------------------------
# Input sliders -- these let the user describe a flower.
# Each slider returns a number that we store in a variable.
# ---------------------------------------------------------------------
st.subheader("Flower measurements (in centimeters)")

col1, col2 = st.columns(2)  # split the sliders into two side-by-side columns
with col1:
    sepal_length = st.slider("Sepal length", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
    petal_length = st.slider("Petal length", min_value=1.0, max_value=7.0, value=4.3, step=0.1)
with col2:
    sepal_width = st.slider("Sepal width", min_value=2.0, max_value=4.5, value=3.0, step=0.1)
    petal_width = st.slider("Petal width", min_value=0.1, max_value=2.5, value=1.3, step=0.1)

st.markdown("---")

# ---------------------------------------------------------------------
# Predict button -- when clicked, everything inside this "if" block runs.
# ---------------------------------------------------------------------
if st.button("🔍 Predict", type="primary", use_container_width=True):

    # Package the slider values into a dictionary matching what the API
    # expects (see IrisFeatures in api.py).
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    try:
        # Send the data to the API and wait for a response.
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        response.raise_for_status()  # raises an error if the API returned a failure status code
        result = response.json()

        # Show the headline result
        st.success(f"Predicted species: **{result['predicted_class'].capitalize()}**")

        # Show a bar chart of how confident the model was for each species
        probs = result["probabilities"]
        chart_data = pd.DataFrame(
            {"Species": list(probs.keys()), "Probability": list(probs.values())}
        ).set_index("Species")

        st.bar_chart(chart_data)
        st.dataframe(
            chart_data.style.format({"Probability": "{:.1%}"}),
            use_container_width=True,
        )

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Make sure it's running on port 8000.")
    except requests.exceptions.HTTPError as e:
        st.error(f"The API returned an error: {e}")
    except Exception as e:
        st.error(f"Something unexpected went wrong: {e}")

# ---------------------------------------------------------------------
# A friendly explanation for anyone curious how the pieces fit together.
# ---------------------------------------------------------------------
st.markdown("---")
with st.expander("ℹ️ How this project works"):
    st.write(
        "**1. train_model.py** trains a RandomForest classifier on the "
        "classic Iris dataset (150 example flowers) and saves it as "
        "`model.pkl`.\n\n"
        "**2. api.py** is a small FastAPI web server. It loads `model.pkl` "
        "once when it starts, and exposes a `/predict` endpoint that "
        "anyone (or anything) can send flower measurements to.\n\n"
        "**3. app.py** (this file) is the Streamlit interface. It never "
        "touches the model directly -- when you click Predict, it sends "
        "your slider values to the API over HTTP and displays whatever "
        "comes back. This keeps the ML logic and the user interface "
        "cleanly separated, just like in many real production apps."
    )
