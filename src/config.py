"""Central place for paths and constants, so every script agrees."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Label order MUST match the tweet_eval dataset: 0 = negative, 1 = neutral, 2 = positive
LABELS = ["negative", "neutral", "positive"]

SAMPLE_CSV = DATA_DIR / "sample_econsultation_comments.csv"

BASELINE_PATH = MODEL_DIR / "baseline_tfidf_lr.joblib"
LSTM_PATH = MODEL_DIR / "bilstm.keras"
BERT_DIR = MODEL_DIR / "bert"

# Display names used in the dashboard
MODEL_BASELINE = "TF-IDF + Logistic Regression"
MODEL_LSTM = "Bi-LSTM"
MODEL_BERT = "DistilBERT (fine-tuned)"

# ---- Training data sources (see src/data_loader.py) ----
# Remove a name from this list to leave that dataset out.
SOURCES = ["tweet_eval", "twitter_fin", "sst5", "fin_phrasebank", "yelp", "imdb"]
MAX_PER_SOURCE = 30_000     # cap per dataset so one big dataset cannot drown the others
EXTRA_DIR = DATA_DIR / "extra_labeled"   # drop your own labelled CSVs here (text,label)
EXTRA_REPEAT = 3            # your own comments are copied this many times in training

MODEL_PRETRAINED = "Pretrained RoBERTa (no training needed)"
MODEL_ENSEMBLE = "Ensemble (average of all available models)"
PRETRAINED_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
