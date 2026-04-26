"""
visualize.py
-------------
Generates all visualizations needed for the project report:

    results/class_distribution.png    -> EDA: bar chart of class balance
    results/question_length.png       -> EDA: histogram of question lengths
    results/top_words.png             -> EDA: most common words per class
    results/accuracy_comparison.png   -> Results: NB vs LR accuracy
    results/confusion_matrices.png    -> Results: side-by-side heatmaps

Run:  python visualize.py
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from model import load_data, load_trained_models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_style("whitegrid")


# 1. EDA: Class distribution
def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["category"].value_counts()
    plt.figure(figsize=(7, 5))
    bars = plt.bar(counts.index, counts.values,
                   color=["#4C72B0", "#DD8452"], edgecolor="black")
    plt.title("Class Distribution in the Dataset", fontsize=14, fontweight="bold")
    plt.xlabel("Category")
    plt.ylabel("Number of Samples")
    for bar, value in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.5,
                 str(value), ha="center", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "class_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# 2. EDA: Question length distribution per class
def plot_question_length(df: pd.DataFrame) -> None:
    df = df.copy()
    df["word_count"] = df["question_clean"].str.split().str.len()

    plt.figure(figsize=(8, 5))
    for cat, color in zip(["STEM", "HUMANITIES"], ["#4C72B0", "#DD8452"]):
        subset = df[df["category"] == cat]["word_count"]
        plt.hist(subset, bins=range(1, 16), alpha=0.65,
                 label=cat, color=color, edgecolor="black")
    plt.title("Question Length Distribution by Category",
              fontsize=14, fontweight="bold")
    plt.xlabel("Number of Words per Question")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "question_length.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# 3. EDA: Top words per class
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "is",
    "are", "was", "were", "be", "by", "for", "with", "as", "it",
    "this", "that", "what", "how", "do", "does", "did", "you",
    "your", "i", "we", "they", "their", "its", "from", "at",
    "about", "into", "between", "than",
}


def plot_top_words(df: pd.DataFrame, top_n: int = 10) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, cat, color in zip(axes, ["STEM", "HUMANITIES"],
                              ["#4C72B0", "#DD8452"]):
        words = " ".join(df[df["category"] == cat]["question_clean"]).split()
        words = [w for w in words if w not in STOPWORDS and len(w) > 2]
        common = Counter(words).most_common(top_n)
        if not common:
            continue
        labels, values = zip(*common)
        ax.barh(labels[::-1], values[::-1], color=color, edgecolor="black")
        ax.set_title(f"Top {top_n} Words in {cat} Questions",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Frequency")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "top_words.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# 4. Results: Accuracy comparison
def plot_accuracy_comparison(metrics: dict) -> None:
    models = ["Naive Bayes", "Logistic Regression"]
    accuracies = [metrics["nb_accuracy"], metrics["lr_accuracy"]]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(models, accuracies,
                   color=["#55A868", "#C44E52"], edgecolor="black")
    plt.ylim(0, 1.05)
    plt.title("Model Accuracy Comparison on Test Set",
              fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy")
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                 f"{acc:.2%}", ha="center", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "accuracy_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# 5. Results: Confusion matrices side by side
def plot_confusion_matrices(metrics: dict, labels: list) -> None:
    nb_cm = np.array(metrics["nb_confusion_matrix"])
    lr_cm = np.array(metrics["lr_confusion_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, title, cmap in zip(
        axes,
        [nb_cm, lr_cm],
        ["Naive Bayes Confusion Matrix", "Logistic Regression Confusion Matrix"],
        ["Blues", "Oranges"],
    ):
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                    xticklabels=labels, yticklabels=labels,
                    cbar=False, ax=ax, annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# Main
def main() -> None:
    df = load_data()
    bundle = load_trained_models()

    plot_class_distribution(df)
    plot_question_length(df)
    plot_top_words(df)
    plot_accuracy_comparison(bundle["metrics"])
    plot_confusion_matrices(bundle["metrics"], bundle["labels"])

    print("\nAll visualizations saved to the 'results/' directory.")


if __name__ == "__main__":
    main()
