import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# === Dataset and Results ===
datasets = ["Rotten Tomatoes (Small)", "NLTK (Medium)", "IMDb (Large)"]

log_reg_acc = [0.7482, 0.6702, 0.8847]
naive_bayes_acc = [0.7529, 0.6772, 0.8522]

log_reg_f1 = [0.7451, 0.6771, 0.8869]
naive_bayes_f1 = [0.7506, 0.6872, 0.8538]

log_reg_time = [5.02, 58.04, 15.01]
naive_bayes_time = [0.85, 25.31, 5.16]

# === Confusion Matrices (from your results) ===
conf_matrices = {
    "Figure 4a: RT Small - Logistic Regression": [[811, 251], [286, 785]],
    "Figure 4b: RT Small - Naive Bayes": [[813, 249], [278, 793]],
    "Figure 4c: NLTK Medium - Logistic Regression": [[4200, 2171], [2098, 4475]],
    "Figure 4d: NLTK Medium - Naive Bayes": [[4177, 2194], [1984, 4589]],
    "Figure 4e: IMDb Large - Logistic Regression": [[4324, 637], [516, 4523]],
    "Figure 4f: IMDb Large - Naive Bayes": [[4206, 755], [723, 4316]],
}

# === CSV test results (20 reviews) - aggregated ===
csv_models = [
    "LR IMDb Large", "LR NLTK Medium", "LR RT Small",
    "NB IMDb Large", "NB NLTK Medium", "NB RT Small"
]
csv_positive = [65, 60, 55, 60, 55, 50]
csv_negative = [35, 40, 45, 40, 45, 50]

# === Accuracy Comparison ===
x = np.arange(len(datasets))
plt.figure(figsize=(8,5))
plt.bar(x - 0.2, log_reg_acc, width=0.4, label="Logistic Regression")
plt.bar(x + 0.2, naive_bayes_acc, width=0.4, label="Naive Bayes")
plt.xticks(x, datasets, rotation=15)
plt.ylabel("Accuracy")
plt.title("Figure 1: Accuracy Comparison Across Datasets")
plt.legend()
plt.tight_layout()
plt.show()

# === F1-score Comparison ===
plt.figure(figsize=(8,5))
plt.plot(datasets, log_reg_f1, marker="o", label="Logistic Regression")
plt.plot(datasets, naive_bayes_f1, marker="o", label="Naive Bayes")
plt.ylabel("F1-score")
plt.title("Figure 2: F1-Score Comparison Across Datasets")
plt.legend()
plt.tight_layout()
plt.show()

# === Training Time Comparison ===
plt.figure(figsize=(8,5))
plt.plot(datasets, log_reg_time, marker="o", label="Logistic Regression")
plt.plot(datasets, naive_bayes_time, marker="o", label="Naive Bayes")
plt.ylabel("Time (seconds)")
plt.title("Figure 3: Training Time vs Dataset Size")
plt.legend()
plt.tight_layout()
plt.show()

# === Confusion Matrices for all models and datasets ===
for title, matrix in conf_matrices.items():
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(matrix),
                                  display_labels=["Negative", "Positive"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.show()

# === CSV Test Distribution ===
bar_width = 0.35
index = np.arange(len(csv_models))

plt.figure(figsize=(10,6))
plt.bar(index, csv_positive, bar_width, label="Positive")
plt.bar(index, csv_negative, bar_width, bottom=csv_positive, label="Negative")
plt.xticks(index, csv_models, rotation=30)
plt.ylabel("Percentage of Predictions (%)")
plt.title("Figure 5: Distribution of Predictions on CSV Test Set (20 Reviews)")
plt.legend()
plt.tight_layout()
plt.show()
