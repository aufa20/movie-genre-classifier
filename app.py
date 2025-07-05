from flask import Flask, request, jsonify
from flask_cors import CORS
import fasttext

app = Flask(__name__)
CORS(app)

# Load model once
model = fasttext.load_model("movie_genre_model.ftz")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        description = data.get("description", "").strip()

        if not description:
            return jsonify({"error": "Missing or empty 'description' field"}), 400

        # Predict with FastText
        labels, probabilities = model.predict(description, k=3)

        # Fix for NumPy 2.0: ensure probabilities is a plain list
        if hasattr(probabilities, 'tolist'):
            probabilities = probabilities.tolist()

        genres = [label.replace("__label__", "") for label in labels]
        joined_genres = "+".join(genres)

        return jsonify({
            "genre": joined_genres,
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
