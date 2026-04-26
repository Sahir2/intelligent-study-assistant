"""
main.py
--------
Interactive command-line app for the Intelligent Study Assistant.

Workflow per question:
  1. Classify the question with both ML models (Naive Bayes + Logistic Regression)
  2. Send the question to the Mistral API for a natural-language explanation
  3. Show the predicted subject area together with the AI's explanation

Setup:
For this project to execute properly you will first need to:

  - create a free account on Mistral.ai and obtain an API key: https://mistral.ai/signup
  - Sign up / log in (Google or email both work)
  - They may ask you to verify your phone number — this is required for free tier access, just go through it
  - Once logged in, on the left sidebar click "API Keys" (sometimes called "API" → "Keys")
  - Click the button that says "Create new key" (or "Create API Key")
  - Give it a name like study-assistant and click create

Now in Terminal execute the following commands in the project directory:

  - Install dependencies:  pip install -r requirements.txt
  - Set your Mistral API key as an environment variable:
        export MISTRAL_API_KEY="your_key_here"          (macOS / Linux)
        setx MISTRAL_API_KEY "your_key_here"            (Windows)
NOTE: This API key will only be used within this terminal session. If you open a new terminal, you will need to set the environment variable again.
  - First run will trigger model training automatically.

Run:  python model.py
Run:  python visualize.py

finally run this main app with: python main.py
"""

import os
import sys

import requests

from model import classify_question, load_trained_models

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"


def ask_mistral(question: str, api_key: str) -> str:
    """Send the question to Mistral and return the assistant's reply."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful, beginner-friendly study assistant. "
                    "Explain concepts clearly with short steps and examples."
                ),
            },
            {"role": "user", "content": question},
        ],
        "temperature": 0.4,
    }
    try:
        response = requests.post(MISTRAL_URL, headers=headers,
                                 json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as exc:
        return f"[Error reaching Mistral API: {exc}]"
    except (KeyError, IndexError, ValueError) as exc:
        return f"[Error parsing Mistral response: {exc}]"


def banner() -> None:
    print("=" * 60)
    print("        INTELLIGENT STUDY ASSISTANT")
    print("   ML Classification (Naive Bayes + Logistic Regression)")
    print("                + Mistral AI Explanations")
    print("=" * 60)
    print("Type a study question and press Enter.")
    print("Type 'exit' or 'quit' to leave.\n")


def main() -> None:
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        print("WARNING: MISTRAL_API_KEY environment variable is not set.")
        print("         The classifier will still work, but Mistral replies "
              "will be skipped.\n")

    # Make sure models are loaded (and trained if first run)
    load_trained_models()

    banner()

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        nb_label, lr_label = classify_question(question)
        print(f"\n[Naive Bayes prediction]        {nb_label}")
        print(f"[Logistic Regression prediction] {lr_label}")

        if api_key:
            print("\n[Mistral explanation]")
            print(ask_mistral(question, api_key))
        else:
            print("\n[Mistral explanation skipped — set MISTRAL_API_KEY to enable.]")
        print("-" * 60)


if __name__ == "__main__":
    main()
