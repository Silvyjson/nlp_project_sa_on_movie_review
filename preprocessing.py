# preprocessing.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(filepath, max_features=5000):
    """
    Preprocess text data from a CSV file and return TF-IDF features, labels, and fitted vectorizer.
    """
    df = pd.read_csv(filepath)

    df.rename(columns=lambda x: x.strip().lower(), inplace=True)

    # Standardize column names
    col_map = {
        'reviews': 'review',
        'review': 'review',
        'text': 'review',
        'sentence': 'review',

        'labels': 'sentiment',
        'label': 'sentiment',
        'sentiment': 'sentiment',
        'tag': 'sentiment'
    }

    # Rename only if present
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Validate required columns
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError(f"Expected columns 'review' and 'sentiment' not found in {filepath}. Found: {df.columns.tolist()}")

    # Lowercase
    df['review'] = df['review'].astype(str).str.lower()
   
    # Remove punctuation & special characters
    df['review'] = df['review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['review'] = df['review'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))
    
    # Tokenization
    df['tokens'] = df['review'].apply(nltk.word_tokenize)
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    df['lemmatized'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(w) for w in x])
    
    # Clean text
    df['clean_text'] = df['lemmatized'].apply(lambda x: ' '.join(x))

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=max_features)
    X = tfidf.fit_transform(df['clean_text']).toarray()

    # Normalize sentiment labels
    df['sentiment'] = df['sentiment'].astype(str).str.lower()
    y = df['sentiment'].map({
        'positive': 1,
        'pos': 1,
        'negative': 0,
        'neg': 0,
        '1': 1,
        '0': 0
    })

     # Final check
    if y.isnull().any():
        raise ValueError(f"Some sentiment labels in {filepath} could not be mapped to 0/1.")
    
    return X, y, tfidf
