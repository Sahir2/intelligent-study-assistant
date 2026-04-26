# Intelligent Study Assistant

ITCS 3156 — Spring 2026 — Final Project
**Author:** Sahir

An intelligent study assistant that combines two classical machine learning classifiers
(Multinomial Naive Bayes and Logistic Regression) with the Mistral AI language model.
The system first predicts whether a student's question belongs to **STEM** or
**HUMANITIES**, then asks Mistral for a beginner-friendly explanation of the question.

---

## Project Structure

```
intelligent-study-assistant/
├── main.py                 # Interactive command-line app (classification + Mistral)
├── model.py                # Trains and evaluates Naive Bayes + Logistic Regression
├── visualize.py            # Generates all EDA + results graphs
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/
│   └── dataset.csv         # 80 labeled study questions (40 STEM / 40 HUMANITIES)
└── results/
    ├── class_distribution.png
    ├── question_length.png
    ├── top_words.png
    ├── accuracy_comparison.png
    ├── confusion_matrices.png
    └── trained_models.pkl  # Saved vectorizer + trained models (auto-generated)
```

---

## Setup

1. Clone this repository.
2. (Recommended) create a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set your Mistral API key as an environment variable:
   - macOS / Linux: `export MISTRAL_API_KEY="your_key_here"`
   - Windows (cmd): `setx MISTRAL_API_KEY "your_key_here"`

   You can get a free API key at <https://console.mistral.ai/>.

---

## How to Run

### 1. Train the models and print evaluation metrics
```bash
python model.py
```
Trains both classifiers, prints accuracy + confusion matrices, and saves the
trained models to `results/trained_models.pkl`.

### 2. Generate all visualizations
```bash
python visualize.py
```
Saves five PNG figures to `results/`:
- `class_distribution.png` — class balance bar chart
- `question_length.png` — histogram of question lengths per category
- `top_words.png` — most frequent words per category
- `accuracy_comparison.png` — Naive Bayes vs Logistic Regression accuracy
- `confusion_matrices.png` — side-by-side confusion matrix heatmaps

### 3. Launch the interactive assistant
```bash
python main.py
```
Type a question, see both ML predictions, and get a Mistral-generated explanation.

---

## Models Used

| Model | Library | Purpose |
|---|---|---|
| Multinomial Naive Bayes | scikit-learn | Probabilistic baseline classifier |
| Logistic Regression | scikit-learn | Discriminative linear classifier |

Both are trained on a Bag-of-Words (`CountVectorizer`) representation of the cleaned
question text, using a stratified 80/20 train/test split.

---

## Results Summary

| Model | Test Accuracy |
|---|---|
| Naive Bayes | 87.50% |
| Logistic Regression | 87.50% |

Both models tie on accuracy, but Logistic Regression has a more balanced confusion
matrix (1 error per class) while Naive Bayes has a small bias toward the
HUMANITIES class. Full discussion is in the project report.

---

## License

This project was developed for academic coursework (ITCS 3156).
