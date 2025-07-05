from flask import Flask, request, jsonify
from flask_cors import CORS
import fasttext

app = Flask(__name__)
CORS(app)

# Load the single-label trained model ONCE
model = fasttext.load_model("movie_genre_model.ftz")  # or .bin

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        description = data.get("description", "").strip()

        if not description:
            return jsonify({"error": "Missing or empty 'description' field"}), 400

        # Use k=1 for single-label classification
        labels, probabilities = model.predict(description, k=1)

        # Ensure compatibility with NumPy 2.x
        if hasattr(probabilities, 'tolist'):
            probabilities = probabilities.tolist()

        genre = labels[0].replace("__label__", "")

        return jsonify({
            "genre": genre,
            "probability": probabilities[0] if probabilities else None
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
