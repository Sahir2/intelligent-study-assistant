# Intelligent Study Assistant

**ITCS 3156 — Spring 2026 — Final Project**
**Author:** Sahir
**University of North Carolina at Charlotte**

An intelligent study assistant that combines two classical machine learning
classifiers (**Multinomial Naive Bayes** and **Logistic Regression**) with the
**Mistral AI** language model. The system first predicts whether a student's
question belongs to **STEM** or **HUMANITIES**, then asks Mistral for a
beginner-friendly explanation of the question.

---

## Demo

```
Your question: what is the integral of cos x

[Naive Bayes prediction]        STEM
[Logistic Regression prediction] STEM

[Mistral explanation]
The integral of cos(x) is sin(x) + C, where C is the constant of integration...
------------------------------------------------------------
Your question: discuss the impact of the cold war

[Naive Bayes prediction]        HUMANITIES
[Logistic Regression prediction] HUMANITIES

[Mistral explanation]
The Cold War (1947–1991) shaped global politics, economics, and culture...
```

---

## Project Structure

```
intelligent-study-assistant/
├── main.py                 # Interactive CLI app (classification + Mistral)
├── model.py                # Trains and evaluates Naive Bayes + Logistic Regression
├── visualize.py            # Generates all EDA + results graphs
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Excludes cache, .pkl, .env
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

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Sahir2/intelligent-study-assistant.git
cd intelligent-study-assistant
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

> **Tip:** use `python -m pip` (not just `pip`) to make sure the packages are installed
> into the same Python interpreter that will run the scripts. If you have multiple
> Pythons (e.g. system Python + Anaconda), this avoids the classic
> "ModuleNotFoundError: No module named 'numpy'" mismatch.

### 3. Get a free Mistral API key

1. Go to <https://console.mistral.ai/>
2. Sign up / log in (Google or email)
3. Verify your phone number (required for free tier)
4. Click **API Keys** → **Create new key**
5. Name it `study-assistant` and copy the key immediately
   (you only see it once)

### 4. Set the API key as an environment variable

**macOS / Linux (bash / zsh):**

```bash
export MISTRAL_API_KEY="your_key_here"
```

**Windows (PowerShell):**

```powershell
$env:MISTRAL_API_KEY="your_key_here"
```

> The key only persists for the current terminal session. If you open a new
> terminal, run the export command again, or add it to your `~/.zshrc` /
> `~/.bashrc` to make it permanent. **Never commit your API key to git** —
> the included `.gitignore` already excludes `.env` files.

### 5. Train the models

```bash
python model.py
```

Trains both classifiers, prints accuracy + classification reports + confusion
matrices, and saves the trained bundle to `results/trained_models.pkl`.

### 6. Generate all visualizations

```bash
python visualize.py
```

Saves five PNG figures to `results/`:

| File                      | What it shows                                        |
| ------------------------- | ---------------------------------------------------- |
| `class_distribution.png`  | Bar chart confirming the dataset is balanced (40/40) |
| `question_length.png`     | Histogram of question lengths per category           |
| `top_words.png`           | Most frequent content words in each category         |
| `accuracy_comparison.png` | Naive Bayes vs Logistic Regression accuracy          |
| `confusion_matrices.png`  | Side-by-side confusion matrix heatmaps               |

### 7. Launch the interactive assistant

```bash
python main.py
```

Type a study question, see both ML predictions, and read a Mistral-generated
explanation. Type `exit` or `quit` to leave.

---

## Models Used

| Model                   | Library      | Why this model                                                                                                        |
| ----------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------- |
| Multinomial Naive Bayes | scikit-learn | Strong, fast probabilistic baseline for text data; works well with small datasets and word-count features             |
| Logistic Regression     | scikit-learn | Discriminative complement to Naive Bayes; does not assume feature independence, so it handles correlated words better |

Both classifiers are trained on a Bag-of-Words representation
(`CountVectorizer`) of the cleaned question text, using a stratified 80/20
train/test split with `random_state=42` for reproducibility.

---

## Results

| Model               | Test Accuracy | Notes                                                         |
| ------------------- | ------------- | ------------------------------------------------------------- |
| Naive Bayes         | **87.50%**    | Perfect HUMANITIES recall, but misclassified 2 STEM questions |
| Logistic Regression | **87.50%**    | Symmetric errors (1 in each direction) — more balanced        |

Both models tie on accuracy but behave differently. Naive Bayes leans toward
HUMANITIES because the independence assumption gives extra weight to common
words like _explain_ and _describe_. Logistic Regression learns joint feature
weights, so it can downweight ambiguous shared words and produces more balanced
predictions. See the project report for the full discussion.

---

## Tech Stack

- **Python** 3.10+
- **scikit-learn** — Naive Bayes, Logistic Regression, vectorization, evaluation
- **pandas** / **numpy** — data loading and manipulation
- **matplotlib** / **seaborn** — visualizations
- **requests** — calls to the Mistral API
- **Mistral AI API** — generative explanations (model: `mistral-small-latest`)

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'numpy'`**
You probably installed packages into a different Python than the one running
the script. Run `python -m pip install -r requirements.txt` (note the
`python -m pip`) — this guarantees pip installs into your current Python.

**`{"detail": "Unauthorized"}` when running main.py**
Your `MISTRAL_API_KEY` is missing or wrong. Re-run the `export` command from
step 4 with the correct key, then verify with `echo $MISTRAL_API_KEY`.

**The classifier still works without an API key**
If you don't set `MISTRAL_API_KEY`, `main.py` prints a warning and skips the
Mistral call but still shows the ML predictions, so you can demo the
classifier piece on its own.

---

## License

This project was developed for academic coursework (ITCS 3156, Spring 2026).
