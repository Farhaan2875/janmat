"""One function the dashboard calls: predict(texts, model_name)."""
import importlib.util
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import (BASELINE_PATH, BERT_DIR, LABELS, LSTM_PATH, MODEL_BASELINE, MODEL_BERT,
                        MODEL_DIR,
                        MODEL_ENSEMBLE, MODEL_LSTM, MODEL_PRETRAINED, PRETRAINED_ID)
from src.preprocess import clean_text, light_clean

_CACHE = {}  # models are loaded once and reused
_NAME_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}   # our column order = LABELS


def _transformers_installed():
    return importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("torch") is not None


def single_models():
    """Individual models that can be used right now."""
    found = []
    if BASELINE_PATH.exists():
        found.append(MODEL_BASELINE)
    if LSTM_PATH.exists():
        found.append(MODEL_LSTM)
    if BERT_DIR.exists() and _transformers_installed():
        found.append(MODEL_BERT)
    if _transformers_installed():
        found.append(MODEL_PRETRAINED)   # downloads ~500 MB from HuggingFace on first use
    return found


def available_models():
    """What the dashboard offers: single models, plus an ensemble when there are at least two."""
    models = single_models()
    return models + [MODEL_ENSEMBLE] if len(models) >= 2 else models


@lru_cache(maxsize=1)
def _comparison_accuracy():
    path = MODEL_DIR / "model_comparison.csv"
    if not path.exists():
        return {}
    comparison = pd.read_csv(path)
    if "model" not in comparison.columns or "overall_acc" not in comparison.columns:
        return {}
    return dict(zip(comparison["model"], comparison["overall_acc"]))


def model_accuracy(model_name):
    """Return the evaluated overall accuracy for a model, when it is available."""
    return _comparison_accuracy().get(model_name)


def _transformer_probs(source, texts):
    """Class probabilities from a HuggingFace model, columns re-ordered to negative/neutral/positive."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    key = str(source)
    if key not in _CACHE:
        tok = AutoTokenizer.from_pretrained(source)
        mdl = AutoModelForSequenceClassification.from_pretrained(source).eval()
        # The model's own config says which column is which class - use it, don't assume.
        order = [_NAME_TO_ID[mdl.config.id2label[i].lower()] for i in range(3)]
        _CACHE[key] = (tok, mdl, np.argsort(order))
    tok, mdl, reorder = _CACHE[key]
    out = []
    for i in range(0, len(texts), 32):   # small batches keep memory low
        enc = tok(texts[i:i + 32], truncation=True, max_length=160, padding=True, return_tensors="pt")
        with torch.no_grad():
            out.append(torch.softmax(mdl(**enc).logits, dim=1).numpy())
    return np.vstack(out)[:, reorder]


def _probabilities(texts, model_name):
    """texts are the ORIGINAL comments; each model applies the cleaning it was trained with."""
    if model_name == MODEL_BASELINE:
        if model_name not in _CACHE:
            _CACHE[model_name] = joblib.load(BASELINE_PATH)
        return _CACHE[model_name].predict_proba([clean_text(t) for t in texts])

    if model_name == MODEL_LSTM:
        if model_name not in _CACHE:
            from tensorflow import keras
            _CACHE[model_name] = keras.models.load_model(LSTM_PATH)
        import tensorflow as tf
        data = tf.constant([clean_text(t) for t in texts], dtype=tf.string)[:, tf.newaxis]
        return _CACHE[model_name].predict(data, verbose=0)

    if model_name == MODEL_BERT:
        return _transformer_probs(BERT_DIR, [light_clean(t) for t in texts])

    if model_name == MODEL_PRETRAINED:
        return _transformer_probs(PRETRAINED_ID, [light_clean(t) for t in texts])

    if model_name == MODEL_ENSEMBLE:   # simple average: errors of different models partly cancel out
        return np.mean([_probabilities(texts, m) for m in single_models()], axis=0)

    raise ValueError(f"Unknown model: {model_name}")


def predict(texts, model_name) -> pd.DataFrame:
    """Return a DataFrame with columns: sentiment, confidence."""
    texts = [str(t) for t in texts]
    proba = _probabilities(texts, model_name)
    return pd.DataFrame({
        "sentiment": [LABELS[i] for i in proba.argmax(axis=1)],
        "confidence": proba.max(axis=1).round(3),
    })
