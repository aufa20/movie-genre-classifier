import fasttext

# Path to the training file (make sure train.txt is in the same folder)
train_file = "train.txt"

# Train the FastText model
model = fasttext.train_supervised(
    input=train_file,
    epoch=25,
    lr=1.0,
    wordNgrams=2,
    loss='ova',  # One-vs-All loss function for multi-label classification
    verbose=2
)

# Save the trained model
model.save_model("movie_genre_model.ftz")
print("✅ Model saved as movie_genre_model.ftz")
