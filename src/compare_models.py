"""Compare every available model on every dataset's test split + the e-consultation sample.

Run:  python -m src.compare_models            (500 test rows per dataset - fast)
      python -m src.compare_models --n 2000   (slower, more precise)
Writes models/model_comparison.csv - copy this table into your report.
"""
import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import MODEL_DIR
from src.data_loader import load_all, load_sample_comments
from src.sentiment import available_models, predict


def main(n_per_source: int):
    _, _, test = load_all()
    test = pd.concat([d.sample(min(len(d), n_per_source), random_state=1) for _, d in test.groupby("source")])
    sample = load_sample_comments()
    sample_y = sample["true_label"].map({"negative": 0, "neutral": 1, "positive": 2})
    ids = {"negative": 0, "neutral": 1, "positive": 2}

    rows = []
    for model in available_models():
        print(f"Evaluating {model} ...")
        pred = predict(test["text"], model)["sentiment"].map(ids).values
        row = {"model": model,
               "overall_acc": accuracy_score(test["label"], pred),
               "overall_macro_f1": f1_score(test["label"], pred, average="macro")}
        for source, part in test.assign(pred=pred).groupby("source"):
            row[f"{source}_acc"] = accuracy_score(part["label"], part["pred"])
        sp = predict(sample["comment"], model)["sentiment"].map(ids).values
        row["econsult_sample_acc"] = accuracy_score(sample_y, sp)
        rows.append(row)

    table = pd.DataFrame(rows).set_index("model").round(3)
    print("\n", table.T.to_string())
    MODEL_DIR.mkdir(exist_ok=True)
    table.to_csv(MODEL_DIR / "model_comparison.csv")
    print("\nSaved models/model_comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    main(parser.parse_args().n)
