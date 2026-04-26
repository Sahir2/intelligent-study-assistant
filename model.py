"""
model.py
---------
Trains and evaluates the two ML classifiers used by the Intelligent
Study Assistant: Multinomial Naive Bayes and Logistic Regression.

This module:
  1. Loads the dataset from data/dataset.csv
  2. Cleans + preprocesses the question text
  3. Splits the data into training and testing sets
  4. Vectorizes text with CountVectorizer (Bag-of-Words)
  5. Trains both classifiers
  6. Reports accuracy, classification report, and confusion matrices
  7. Saves the trained vectorizer and models so main.py can reuse them

Run directly with:  python model.py
"""

import os
import re
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_PATH = os.path.join(RESULTS_DIR, "trained_models.pkl")
os.makedirs(RESULTS_DIR, exist_ok=True)


# Preprocessing
def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV dataset and apply text cleaning."""
    df = pd.read_csv(path)
    df = df.dropna()
    df["question_clean"] = df["question"].apply(clean_text)
    return df


# Training
def train_models(random_state: int = 42):
    """Train both classifiers and return trained objects + evaluation."""
    df = load_data()

    X = df["question_clean"].values
    y = df["category"].values

    # 80/20 train-test split, stratified to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    # Bag-of-Words vectorization (fit ONLY on training data)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model 1: Multinomial Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    nb_preds = nb_model.predict(X_test_vec)

    # Model 2: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=random_state)
    lr_model.fit(X_train_vec, y_train)
    lr_preds = lr_model.predict(X_test_vec)

    #  Evaluation 
    nb_acc = accuracy_score(y_test, nb_preds)
    lr_acc = accuracy_score(y_test, lr_preds)

    labels = sorted(np.unique(y))
    nb_cm = confusion_matrix(y_test, nb_preds, labels=labels)
    lr_cm = confusion_matrix(y_test, lr_preds, labels=labels)

    print("=" * 60)
    print("INTELLIGENT STUDY ASSISTANT - MODEL TRAINING")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(X_train)}  |  Testing samples: {len(X_test)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print()

    print("--- Multinomial Naive Bayes ---")
    print(f"Accuracy: {nb_acc:.4f}")
    print(classification_report(y_test, nb_preds, labels=labels, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(nb_cm, index=labels, columns=labels))
    print()

    print("--- Logistic Regression ---")
    print(f"Accuracy: {lr_acc:.4f}")
    print(classification_report(y_test, lr_preds, labels=labels, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(lr_cm, index=labels, columns=labels))
    print()

    # Persist everything so main.py can load it without retraining
    with open(MODELS_PATH, "wb") as f:
        pickle.dump(
            {
                "vectorizer": vectorizer,
                "nb_model": nb_model,
                "lr_model": lr_model,
                "labels": labels,
                "metrics": {
                    "nb_accuracy": nb_acc,
                    "lr_accuracy": lr_acc,
                    "nb_confusion_matrix": nb_cm.tolist(),
                    "lr_confusion_matrix": lr_cm.tolist(),
                },
                "test_data": {
                    "X_test": X_test,
                    "y_test": y_test,
                    "nb_preds": nb_preds,
                    "lr_preds": lr_preds,
                },
            },
            f,
        )
    print(f"Saved trained models to: {MODELS_PATH}")

    return vectorizer, nb_model, lr_model, labels


def load_trained_models():
    """Load the pickled bundle. Trains first if it doesn't exist."""
    if not os.path.exists(MODELS_PATH):
        train_models()
    with open(MODELS_PATH, "rb") as f:
        return pickle.load(f)


def classify_question(question: str):
    """Return (nb_label, lr_label) for a new user question."""
    bundle = load_trained_models()
    vectorizer = bundle["vectorizer"]
    nb_model = bundle["nb_model"]
    lr_model = bundle["lr_model"]

    cleaned = clean_text(question)
    vec = vectorizer.transform([cleaned])
    return nb_model.predict(vec)[0], lr_model.predict(vec)[0]


if __name__ == "__main__":
    train_models()
