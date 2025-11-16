from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.text import tokenizer_from_json


MODEL_PATH = "models/suicide_model.keras"
TOKENIZER_PATH = "models/tokenizer.json"
MAX_SEQUENCE_LENGTH = 200


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev_secret_change_me")


def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")

    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "r") as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    return model, tokenizer


def preprocess_texts(texts, tokenizer):
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    return padded



model, tokenizer = load_resources()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text_input", "").strip()
        if not text:
            flash("Please enter some text to analyze.")
            return redirect(url_for("index.html"))


        seq = preprocess_texts([text], tokenizer)
        prob = model.predict(seq)[0][0]
        label = "Suicide" if prob >= 0.5 else "Non-suicide"
        confidence = float(prob) if label == "Suicide" else float(1 - prob)


        return render_template("result.html", label=label, confidence=confidence, text=text)

    perf_exists = {
    'performance': os.path.exists('static/metrics.png'),
    'training': os.path.exists('static/training_curves.png'),
    'confusion': os.path.exists('static/confusion_matrix.png')
    }
    return render_template("index.html", perf_exists=perf_exists)


@app.route("/about")
def about():
    return render_template("about.html")


app.run(host='0.0.0.0', port=5000, debug=True)