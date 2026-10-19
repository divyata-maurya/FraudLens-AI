
import os

import streamlit as st
from groq import Groq


DEFAULT_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """
You are an AI assistant inside a Credit Card Fraud Detection project.

Your responsibilities:
- Explain credit card fraud detection in simple language.
- Explain fraud probability, classification thresholds, model predictions,
  false positives, false negatives, precision, and recall.
- Help users understand how machine learning fraud detection works.
- Answer questions about the project's functionality and safe usage.
- Be friendly, concise, and technically accurate.

Important limitations:
- A model prediction is an estimate, not proof that a transaction is fraudulent.
- Never claim that a transaction is definitely safe or definitely fraudulent
  based only on a model prediction.
- Do not request credit card numbers, CVV, PINs, passwords, or other sensitive
  banking information.
- If a question is unrelated to fraud detection or this project, politely
  redirect the user to the project's purpose.
"""


def get_groq_client():
    """Create a Groq client using Streamlit secrets or an environment variable."""
    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except (FileNotFoundError, KeyError):
        pass

    api_key = api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "Groq API key not found. Configure GROQ_API_KEY in "
            ".streamlit/secrets.toml."
        )

    return Groq(api_key=api_key)


def get_chat_response(messages):
    """Send conversation history to Groq and return the assistant's reply."""
    client = get_groq_client()

    model = DEFAULT_MODEL
    try:
        model = st.secrets.get("GROQ_MODEL", DEFAULT_MODEL)
    except (FileNotFoundError, KeyError):
        pass

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *messages,
        ],
        temperature=0.4,
        max_completion_tokens=1024,
    )

    return response.choices[0].message.content or (
        "Sorry, I couldn't generate a response. Please try again."
    )