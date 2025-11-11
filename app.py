from flask import Flask, request, jsonify
import numpy as np
import joblib, os

app = Flask(__name__)

# Model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model.pkl")

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model not found. Run train_model.py first!")
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return jsonify({"message": "✅ Disease Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array([list(data.values())], dtype=float)
        prediction = model.predict(features)[0]
        return jsonify({"input": data, "prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
