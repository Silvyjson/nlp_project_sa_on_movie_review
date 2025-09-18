## 🎬 Sentiment Analysis on Movie Reviews (NLP Project)

This project implements a **sentiment analysis system** that classifies movie reviews as **Positive** or **Negative** using Natural Language Processing (NLP) and Machine Learning techniques.  
It follows the requirements of the IU project task for **DLBAIPNLP01 – NLP**.

---

## 🚀 Features
- Preprocessing pipeline:
  - Lowercasing
  - Removing stopwords
  - Lemmatization
  - TF-IDF encoding
- Multiple models:
  - Logistic Regression
  - Naive Bayes
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
  - Training Time
- Interactive prediction: enter your own reviews and get instant predictions.
- Modular structure for easy extension (add new models like SVM, LSTM, BERT).

---

## 📂 Project Structure
```

nlp_project_sa_on_movie_review/
│
├── preprocessing.py
├── train.py
├── main.py
├── predict.py
├── predict_csv.py
│
├── models/
│   ├── logistic_regression.py
│   ├── naive_bayes.py
│
├── trained_models/
│   ├── logistic_regression_imdb_large/
│   ├── logistic_regression_nltk_movie_reviews_medium/
│   ├── logistic_regression_rotten_tomatoes_small/
│   ├── naive_bayes_imdb_large/
│   ├── naive_bayes_nltk_movie_reviews_medium/
│   ├── naive_bayes_rotten_tomatoes_small/
│
├── Datasets/
│   ├── IMDB Dataset.csv
│   ├── movie_review.csv
│   ├── data_rt.csv
│
├── test/
│   └── test_reviews.csv
│
├── output/
│   └── predicted_reviews.csv
│
├── requirements.txt
└── README.md

````

---

## 📊 Datasets Used
We progressively test the system on **different dataset sizes**:

1. **Small (2K reviews)** → NLTK Movie Reviews (`nltk.corpus.movie_reviews`)  
2. **Medium (10K reviews)** → Rotten Tomatoes (via Hugging Face: `rotten_tomatoes`)  
3. **Large (50K reviews)** → IMDb Large Movie Review Dataset (CSV file)  

---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-nlp.git
   cd sentiment-analysis-nlp
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download datasets:

   * IMDb: [Kaggle or AI Stanford dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
   * Rotten Tomatoes: [Hugging Face Datasets](https://huggingface.co/datasets/rotten_tomatoes)
   * NLTK Movie Reviews: comes with `nltk`.

   uncomment on the main.py if needed
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    nltk.download('omw-1.4')

---

## 🏃 How to Run

### Train Models

Run training and comparison:

```bash
python main.py
```

### Predict with a Model

Run interactive mode:

```bash
python predict.py
```

Type your own review and choose a model (`logistic_regression` / `naive_bayes`).

---

## 📈 Example Results (IMDb dataset)

```
=== Model Comparison Summary ===
Logistic Regression: Acc=0.89, Prec=0.89, Rec=0.89, F1=0.89, Time=15.2s
Naive Bayes: Acc=0.84, Prec=0.84, Rec=0.84, F1=0.84, Time=3.4s
```

---

## 📝 Report Notes

In the project report, you should:

* Explain preprocessing pipeline (stopword removal, lemmatization, TF-IDF).
* Describe models used (Logistic Regression, Naive Bayes).
* Show evaluation results across datasets (small → medium → large).
* Compare models (accuracy, time, etc.).
* Conclude with insights and possible improvements (e.g., SVM, LSTM, Transformers).

---

## 📜 License

This project is for educational purposes under IU’s NLP project task. Please do not redistribute without permission.

```

---

👉 Do you want me to also generate a **requirements.txt** file for your repo so everything runs smoothly on another machine?
```
