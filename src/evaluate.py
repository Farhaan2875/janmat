"""Small helper to print and save evaluation results in one format."""
import json

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import LABELS, MODEL_DIR


def report(y_true, y_pred, name: str, split: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n=== {name} on {split} ===")
    print(f"Accuracy: {acc:.4f}   Macro-F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_metrics(name: str, results: dict):
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / f"metrics_{name}.json", "w") as f:
        json.dump(results, f, indent=2)


def report_by_source(test_df, y_pred, name: str) -> dict:
    """Accuracy / macro-F1 for each dataset separately, so we can see where a model is weak."""
    out = {}
    print(f"\n--- {name}: accuracy per dataset ---")
    for source, part in test_df.assign(pred=y_pred).groupby("source"):
        acc = accuracy_score(part["label"], part["pred"])
        f1 = f1_score(part["label"], part["pred"], average="macro")
        print(f"{source:16s} n={len(part):5d}  acc={acc:.3f}  macro-F1={f1:.3f}")
        out[source] = {"n": int(len(part)), "accuracy": round(float(acc), 4), "macro_f1": round(float(f1), 4)}
    return out
