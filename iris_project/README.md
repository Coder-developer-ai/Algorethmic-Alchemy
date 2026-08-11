# 🌸 Iris Classifier — Streamlit + FastAPI (Beginner-Friendly)

A small but complete ML web app, split into two clean pieces:

- **A backend** (FastAPI) that runs the machine learning model
- **A frontend** (Streamlit) that lets a user interact with it in a browser

This mirrors how real-world ML products are usually built — the model
lives on a server, and different apps (web, mobile, etc.) talk to it
over an API instead of each having their own copy of the model.

## 📁 Files

| File | What it does |
|---|---|
| `train_model.py` | Trains the ML model on the Iris flower dataset, saves it as `model.pkl` |
| `api.py` | FastAPI server — loads `model.pkl`, exposes a `/predict` endpoint |
| `app.py` | Streamlit app — sliders + button, calls the API, shows the result |
| `requirements.txt` | List of Python packages this project needs |

Every file is heavily commented — open them up and read along if you
want to understand *why*, not just *what*.

## ✅ Setup (do this once)

Make sure you have Python 3.9+ installed, then from inside this folder:

```bash
pip install -r requirements.txt
```

## ▶️ Running the app (3 steps, in order)

**Step 1 — Train the model.** This creates the `model.pkl` file.
```bash
python train_model.py
```
You should see something like `Test accuracy: 90.00%` printed out.

**Step 2 — Start the API**, in its own terminal window:
```bash
uvicorn api:app --reload --port 8000
```
Leave this terminal running. Visit http://localhost:8000/docs to see
an interactive page where you can test the API by hand.

**Step 3 — Start the Streamlit app**, in a *second* terminal window:
```bash
streamlit run app.py
```
This opens http://localhost:8501 in your browser automatically.

You now have two things running at once — that's normal! The Streamlit
app (frontend) and the FastAPI server (backend) are two separate
programs talking to each other.

## 🧠 Quick glossary

- **Model** — the trained "brain" that takes flower measurements and
  guesses the species. Created by `train_model.py`, saved as `model.pkl`.
- **API (Application Programming Interface)** — a way for programs to
  talk to each other over the network using URLs, similar to how your
  browser talks to a website. Our API has one main "endpoint":
  `/predict`.
- **Backend** — the part that does the actual work (here: `api.py`
  running the model). Usually has no visual interface.
- **Frontend** — the part the user sees and clicks on (here: `app.py`,
  the Streamlit page).
- **Endpoint** — a specific URL on an API that does one job, e.g.
  `POST /predict`.

## 🛠️ Troubleshooting

- **Streamlit sidebar says "Cannot reach the API"** → make sure the
  `uvicorn` terminal from Step 2 is still running.
- **"model.pkl was not found"** → run `python train_model.py` again, and
  make sure it's in the same folder as `api.py`.
- **Port already in use** → another program is using port 8000 or 8501.
  Either stop that program, or run with a different port, e.g.
  `uvicorn api:app --port 8001` (and update `API_URL` in `app.py` to match).

## 🚀 Ideas to extend this project

- Swap `RandomForestClassifier` for another scikit-learn model and
  compare accuracy.
- Add a file-upload option so users can classify many flowers from a CSV.
- Deploy the API to a cloud service, then update `API_URL` in `app.py`
  to point to it — now anyone can use your Streamlit app without
  running the API themselves.
