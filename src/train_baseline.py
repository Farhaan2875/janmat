"""Step 1 - Baseline: TF-IDF + Logistic Regression.

Run:  python -m src.train_baseline
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import BASELINE_PATH, MODEL_DIR
from src.data_loader import load_all, load_sample_comments
from src.evaluate import report, report_by_source, save_metrics


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    train, val, test = load_all()      # several datasets combined (see src/data_loader.py)
    print("Total:", len(train), "train /", len(val), "val /", len(test), "test")

    # TF-IDF turns each text into a vector of weighted word / word-pair counts.
    # Logistic Regression then learns which words push towards each class.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced")),
    ])
    model.fit(train["clean"], train["label"])

    pred = model.predict(test["clean"])
    results = {"combined_test": report(test["label"], pred, "Baseline", "combined test set"),
               "per_source": report_by_source(test, pred, "Baseline")}

    # Domain check: how does the model do on (simulated) policy comments?
    sample = load_sample_comments()
    sample_y = sample["true_label"].map({"negative": 0, "neutral": 1, "positive": 2})
    results["econsultation_sample"] = report(sample_y, model.predict(sample["clean"]), "Baseline", "e-consultation sample")

    joblib.dump(model, BASELINE_PATH)
    save_metrics("baseline", results)
    print("\nSaved model to", BASELINE_PATH)


if __name__ == "__main__":
    main()
