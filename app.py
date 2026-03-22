from flask import Flask, request, jsonify
from PIL import Image

import tensorflow as tf
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return "Hello World!"


model = tf.keras.models.load_model("models/model.h5")
class_names = ["Kirmizi_Pistachio", "Siirt_Pistachio"]

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Invalid image", "detail": str(e)}), 400

    img = img.resize((64, 64))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    prediction_class = np.argmax(prediction, axis=1)[0]

    return jsonify({
        "class": class_names[prediction_class],
        "probabilities": prediction.tolist()
    })


if __name__ == "__main__":
    app.run(debug=True)
