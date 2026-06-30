"""
app.py
-------
The web app. Run this AFTER you've trained a model with train_model.py.

HOW TO RUN
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import json
import os
import uuid

import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "fruit_model.pth")
CLASS_NAMES_PATH = os.path.join("model", "class_names.json")
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Friendly display names for common fruits. Add to this if your dataset
# has fruits not listed here -- anything missing just falls back to the
# raw folder name with a capital letter.
FRUIT_DISPLAY_NAMES = {
    "apples": "Apple",
    "apple": "Apple",
    "banana": "Banana",
    "bananas": "Banana",
    "oranges": "Orange",
    "orange": "Orange",
    "grapes": "Grapes",
    "tomato": "Tomato",
    "tomatoes": "Tomato",
    "guava": "Guava",
    "mango": "Mango",
    "strawberry": "Strawberry",
}


def parse_class_name(class_name):
    """
    Turns a folder name like "freshapples" or "rotten_banana" into
    ("Fresh", "Apple"). Assumes every class name starts with "fresh" or
    "rotten" (this is how the standard dataset, and this whole app, is set up).
    """
    lower = class_name.lower().replace("-", "").replace("_", "")
    if lower.startswith("fresh"):
        freshness = "Fresh"
        fruit_key = lower[len("fresh"):]
    elif lower.startswith("rotten"):
        freshness = "Rotten"
        fruit_key = lower[len("rotten"):]
    else:
        # Doesn't match the fresh/rotten naming convention -- show as-is.
        return "Unknown", class_name.capitalize()

    fruit_name = FRUIT_DISPLAY_NAMES.get(fruit_key, fruit_key.capitalize() if fruit_key else "Fruit")
    return freshness, fruit_name


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        return None, None

    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, class_names


# Load once at startup so every request is fast.
model, class_names = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html", model_ready=model is not None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "No trained model found. Run train_model.py first."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image was sent."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Please upload a PNG, JPG, JPEG, or WEBP image."}), 400

    # Save with a random name so uploads never collide or overwrite each other.
    ext = file.filename.rsplit(".", 1)[1].lower()
    saved_name = f"{uuid.uuid4().hex}.{ext}"
    saved_path = os.path.join(UPLOAD_FOLDER, saved_name)
    file.save(saved_path)

    try:
        image = Image.open(saved_path).convert("RGB")
    except Exception:
        return jsonify({"error": "That file doesn't look like a valid image."}), 400

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = class_names[predicted_idx.item()]
    freshness, fruit = parse_class_name(predicted_class)

    # Also return the full probability breakdown -- nice for the UI to show
    # "how sure" the model is across all classes, not just the top pick.
    all_scores = [
        {"label": class_names[i], "confidence": round(probabilities[i].item() * 100, 1)}
        for i in range(len(class_names))
    ]
    all_scores.sort(key=lambda x: x["confidence"], reverse=True)

    return jsonify({
        "freshness": freshness,
        "fruit": fruit,
        "confidence": round(confidence.item() * 100, 1),
        "image_url": f"/{saved_path}",
        "all_scores": all_scores,
    })


if __name__ == "__main__":
    if model is None:
        print("\nWARNING: No trained model found at model/fruit_model.pth")
        print("The site will load, but predictions will fail until you run train_model.py.\n")
    app.run(debug=True)
