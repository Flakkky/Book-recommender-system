from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import pickle

model = pickle.load(open("cnn_model.pkl", "rb"))

# Define the image size and the classes
IMG_SIZE = 224
CLASSES = ["cat", "dog", "bird"]

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files["image"]
    image = Image.open(image_file)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = CLASSES[index]
    return f"The image is classified as {class_name}"
