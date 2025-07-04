from flask import Flask, request, jsonify
import fasttext

app = Flask(__name__)

# Load FastText model
model = fasttext.load_model("movie_genre_model.ftz")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        data = request.get_json()
        description = data.get("description", "").strip()

        # Validate input
        if not description:
            return jsonify({"error": "Missing or empty 'description' field"}), 400

        # Predict top 3 genres
        labels, probabilities = model.predict(description, k=3)

        # Clean labels
        genres = [label.replace("__label__", "") for label in labels]
        joined_genres = "+".join(genres)

        # Return result
        return jsonify({"genre": joined_genres})

    except Exception as e:
        # Handle any failure
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
