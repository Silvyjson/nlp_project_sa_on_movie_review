import joblib
import re
import os
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Folder where models are stored
MODEL_DIR = "trained_models"

def preprocess_single_review(review, tfidf):
    """
    Preprocess a single review and transform it into TF-IDF vector.
    """
    review = review.lower()
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    stop_words = set(stopwords.words('english'))
    review = ' '.join([w for w in review.split() if w not in stop_words])
    tokens = nltk.word_tokenize(review)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    clean_text = ' '.join(tokens)

    return tfidf.transform([clean_text]).toarray()


def clean_model_name(filename: str) -> str:
    """
    Convert 'logistic_regression_(imdb_(large)).pkl'
    into 'Logistic Regression (IMDb - Large)'.
    """
    name = filename.replace(".pkl", "")
    name = name.replace("_(", " (").replace(")_", ") ").replace("_", " ")

    if "(" in name and ")" in name:
        algo, dataset = name.split("(", 1)
        algo = algo.strip().title()
        dataset = dataset.strip(")").replace("(", "").replace(")", "")
        dataset = dataset.replace("  ", " ").title()
        dataset = dataset.replace("Imdb", "IMDb")
        dataset = dataset.replace("Rt", "Rotten Tomatoes")
        dataset = dataset.replace("Nltk Movie Reviews", "NLTK Movie Reviews")
        return f"{algo} ({dataset})"
    else:
        return name.title()


def predict_with_all_models(review):
    """
    Loop through all saved models in the trained_models folder
    and predict sentiment with each, including confidence score.
    """
    results = {}

    # Loop through each folder in trained_models
    for folder in os.listdir(MODEL_DIR):
        folder_path = os.path.join(MODEL_DIR, folder)
        if os.path.isdir(folder_path):
            model_file = None
            vectorizer_file = None
            # Find model and vectorizer files in the folder
            for file in os.listdir(folder_path):
                if file.endswith('.pkl') and file != 'tfidf_vectorizer.pkl':
                    model_file = os.path.join(folder_path, file)
                elif file == 'tfidf_vectorizer.pkl':
                    vectorizer_file = os.path.join(folder_path, file)

            if model_file and vectorizer_file:
                try:
                    model = joblib.load(model_file)
                    tfidf = joblib.load(vectorizer_file)
                    vectorized_review = preprocess_single_review(review, tfidf)

                    # Prediction + probability
                    pred = model.predict(vectorized_review)[0]
                    if hasattr(model, "predict_proba"):
                        proba = np.max(model.predict_proba(vectorized_review))
                    else:
                        proba = 1.0

                    sentiment = "Positive" if pred == 1 else "Negative"
                    results[clean_model_name(os.path.basename(model_file))] = f"{sentiment} ({proba:.2f})"

                except Exception as e:
                    results[clean_model_name(os.path.basename(model_file))] = f"Error: {str(e)}"
            else:
                folder_display = folder.replace('_', ' ').title()
                results[folder_display] = "Error: Model or vectorizer file missing."

    return results


if __name__ == "__main__":
    print("=== Sentiment Prediction (All Models) ===\n")

    while True:
        review = input("\nEnter a movie review (or type 'exit' to quit): ")
        if review.lower() == "exit":
            break

        predictions = predict_with_all_models(review)
        print("\nResults:")
        for model_name, sentiment in predictions.items():
            print(f"{model_name}: {sentiment}")
