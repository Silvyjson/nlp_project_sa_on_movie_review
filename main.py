import nltk
import joblib
import os
from preprocessing import preprocess_data
from train import train_and_evaluate
from models.logistic_regression import get_model as get_lr_model
from models.naive_bayes import get_model as get_nb_model

# Download resources if not already installed
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt_tab')
# nltk.download('omw-1.4')

def main():
    datasets = {
        "Rotten Tomatoes (Small)": "Datasets/data_rt.csv",
        "NLTK Movie Reviews (Medium)": "Datasets/movie_review.csv",
        "IMDb (Large)": "Datasets/IMDB Dataset.csv"
    }

    for dataset_name, filepath in datasets.items():
        print(f"\n\n=== Running on {dataset_name} ===")

        # Preprocess
        print("Preprocessing data...")
        X, y, tfidf = preprocess_data(filepath)

        results = {}

        # List of models and their names
        models = [
            (get_lr_model(), "Logistic Regression"),
            (get_nb_model(), "Naive Bayes")
        ]

        for model, model_type in models:
            # Create folder name: model_type + dataset_name, cleaned
            folder_name = f"{model_type}_{dataset_name}".lower().replace(" ", "_").replace("(", "").replace(")", "")
            model_dir = os.path.join("trained_models", folder_name)
            os.makedirs(model_dir, exist_ok=True)

            # Save vectorizer in the same folder
            vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
            joblib.dump(tfidf, vectorizer_path)
            print(f"TF-IDF vectorizer saved at {vectorizer_path}")

            # Save model using train_and_evaluate
            model_name = f"{model_type} ({dataset_name})"
            results[model_type] = train_and_evaluate(
                model, X, y, model_name
            )

        # Comparison summary
        print(f"\n=== {dataset_name} - Model Comparison Summary ===")
        for model_name, metrics in results.items():
            print(f"{model_name}: "
                  f"Acc={metrics['accuracy']:.4f}, "
                  f"Prec={metrics['precision']:.4f}, "
                  f"Rec={metrics['recall']:.4f}, "
                  f"F1={metrics['f1']:.4f}, "
                  f"Time={metrics['time']:.2f}s")

if __name__ == "__main__":
    main()
