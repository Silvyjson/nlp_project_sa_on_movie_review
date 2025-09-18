import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib

def train_and_evaluate(model, X, y, model_name, test_size=0.2, random_state=42, save_model=True):
    """
    Train and evaluate a model with given data.
    """
    start = time.time()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    end = time.time()
    duration = end - start

    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    print(f"Training + Evaluation Time: {duration:.2f} seconds")

    if save_model:
        # Clean up folder name to remove illegal characters
        folder_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_dir = os.path.join("trained_models", folder_name)

        # Create folder if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save model
        filename = os.path.join(save_dir, f"{folder_name}.pkl")
        joblib.dump(model, filename)
        print(f"{model_name} model saved at {filename}!")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
        "time": duration
    }
