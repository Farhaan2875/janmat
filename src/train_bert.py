"""Step 3 (stretch goal) - fine-tune a transformer on ALL the combined datasets.

Run:  python -m src.train_bert --quick     CPU smoke test (4,000 rows, 1 epoch)
      python -m src.train_bert             full run - use a GPU (Google Colab is free)
      python -m src.train_bert --base cardiffnlp/twitter-roberta-base-sentiment-latest
          ^ starts from a model that already understands sentiment: best accuracy, needs a GPU
Needs:  pip install transformers torch accelerate
"""
import argparse

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

from src.config import BERT_DIR, MODEL_DIR
from src.data_loader import load_all
from src.evaluate import report, report_by_source, save_metrics
from src.preprocess import light_clean

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def main(base: str, quick: bool):
    MODEL_DIR.mkdir(exist_ok=True)
    train, val, test = load_all()
    for df in (train, val, test):
        df["input"] = df["text"].apply(light_clean)   # same cleaning the dashboard uses
    if quick:
        train = train.sample(4000, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(base)

    def to_dataset(df):
        ds = Dataset.from_pandas(df[["input", "label"]].reset_index(drop=True))
        return ds.map(lambda b: tokenizer(b["input"], truncation=True, max_length=128), batched=True)

    train_ds, val_ds, test_ds = to_dataset(train), to_dataset(val), to_dataset(test)

    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=3, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()},
        ignore_mismatched_sizes=True)

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"accuracy": accuracy_score(labels, preds), "macro_f1": f1_score(labels, preds, average="macro")}

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "bert_runs"), num_train_epochs=1 if quick else 2,
        per_device_train_batch_size=32, per_device_eval_batch_size=64, learning_rate=2e-5,
        eval_strategy="epoch", save_strategy="no", report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)
    trainer.train()

    pred = trainer.predict(test_ds).predictions.argmax(axis=1)
    results = {"combined_test": report(test["label"], pred, "Transformer", "combined test set"),
               "per_source": report_by_source(test, pred, "Transformer")}

    trainer.save_model(str(BERT_DIR))
    tokenizer.save_pretrained(str(BERT_DIR))
    save_metrics("bert", results)
    print("\nSaved model to", BERT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="distilbert-base-uncased")
    parser.add_argument("--quick", action="store_true")
    a = parser.parse_args()
    main(a.base, a.quick)
