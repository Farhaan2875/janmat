"""Step 2 - Bi-LSTM sentiment classifier (TensorFlow / Keras).

Run:  python -m src.train_lstm
The text-to-number step (TextVectorization) lives INSIDE the model, so the
saved model accepts raw strings - no separate tokenizer file to manage.
"""
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from src.config import LSTM_PATH, MODEL_DIR
from src.data_loader import load_all, load_sample_comments
from src.evaluate import report, report_by_source, save_metrics

VOCAB_SIZE = 20_000   # how many distinct words the model knows
SEQ_LEN = 80          # comments are cut / padded to this many words
EPOCHS = 6
BATCH = 64


def as_input(texts):
    """Keras wants shape (n, 1) for string input."""
    # Keras 3 rejects NumPy object/string arrays during tree conversion.
    return tf.constant(list(texts), dtype=tf.string)[:, tf.newaxis]


def build_model(train_texts):
    # 1) words -> integer ids
    vectorizer = layers.TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=SEQ_LEN)
    vectorizer.adapt(as_input(train_texts))

    # 2) ids -> embeddings -> Bi-LSTM -> 3-way softmax
    inputs = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(inputs)
    x = layers.Embedding(VOCAB_SIZE, 128, mask_zero=True)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)   # reads the text forwards AND backwards
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    train, val, test = load_all()      # several datasets combined (see src/data_loader.py)

    model = build_model(train["clean"])
    model.summary()

    # Classes are unevenly spread across the datasets, so we weight them to keep it fair
    weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=train["label"])
    class_weight = {i: w for i, w in enumerate(weights)}

    model.fit(
        as_input(train["clean"]), train["label"].values,
        validation_data=(as_input(val["clean"]), val["label"].values),
        epochs=EPOCHS, batch_size=BATCH, class_weight=class_weight,
        callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
    )

    pred = model.predict(as_input(test["clean"]), verbose=0).argmax(axis=1)
    results = {"combined_test": report(test["label"], pred, "Bi-LSTM", "combined test set"),
               "per_source": report_by_source(test, pred, "Bi-LSTM")}

    sample = load_sample_comments()
    sample_y = sample["true_label"].map({"negative": 0, "neutral": 1, "positive": 2})
    pred_s = model.predict(as_input(sample["clean"]), verbose=0).argmax(axis=1)
    results["econsultation_sample"] = report(sample_y, pred_s, "Bi-LSTM", "e-consultation sample")

    model.save(LSTM_PATH)
    save_metrics("bilstm", results)
    print("\nSaved model to", LSTM_PATH)


if __name__ == "__main__":
    main()
