"""JanMat browser application.

Run with: python web_app.py
The API keeps model inference in Python while the interface is plain HTML, CSS
and JavaScript under web/.
"""
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from src.config import LABELS, SAMPLE_CSV
from src.preprocess import clean_text
from src.sentiment import available_models, predict
from src.summarizer import summarize
from src.topics import cluster_topics

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")


@app.after_request
def allow_local_browser_requests(response):
    """Allow the standalone HTML file to call the local model server."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/api/models")
def models():
    return jsonify({"models": available_models()})


def _read_comments():
    upload = request.files.get("file")
    if upload is None:
        raw = pd.read_csv(SAMPLE_CSV)
        column = "comment"
    else:
        raw = pd.read_csv(upload)
        text_columns = [c for c in raw.columns if pd.api.types.is_string_dtype(raw[c])]
        if not text_columns:
            raise ValueError("The CSV must contain a text column.")
        column = request.form.get("column", "comment")
        if column not in text_columns:
            column = text_columns[0]

    raw = raw.dropna(subset=[column]).reset_index(drop=True)
    if len(raw) < 12:
        raise ValueError("At least 12 comments are needed to identify themes.")
    return raw, column


@app.post("/api/analyze")
def analyze():
    try:
        raw, column = _read_comments()
        model_name = request.form.get("model", "")
        if model_name not in available_models():
            return jsonify({"error": "Select an available model."}), 400
        requested_topics = request.form.get("topics", "auto")
        k = None if requested_topics == "auto" else int(requested_topics)
        comments = raw[column].astype(str).tolist()
        result = predict(comments, model_name)
        cleaned = [clean_text(comment) for comment in comments]
        topic_ids, topic_names = cluster_topics(cleaned, k)

        rows = []
        for i, comment in enumerate(comments):
            rows.append({
                "comment": comment,
                "sentiment": result.iloc[i]["sentiment"],
                "confidence": float(result.iloc[i]["confidence"]),
                "topic": topic_names[topic_ids[i]],
            })

        counts = {label: sum(row["sentiment"] == label for row in rows) for label in LABELS}
        topic_data = []
        for topic in topic_names:
            subset = [row for row in rows if row["topic"] == topic]
            topic_data.append({
                "name": topic,
                "total": len(subset),
                "negative": sum(row["sentiment"] == "negative" for row in subset),
                "neutral": sum(row["sentiment"] == "neutral" for row in subset),
                "positive": sum(row["sentiment"] == "positive" for row in subset),
            })

        return jsonify({
            "model": model_name,
            "total": len(rows),
            "average_confidence": sum(row["confidence"] for row in rows) / len(rows),
            "counts": counts,
            "topics": sorted(topic_data, key=lambda item: item["total"], reverse=True),
            "summary": {
                "concerns": summarize([r["comment"] for r in rows if r["sentiment"] == "negative"]),
                "support": summarize([r["comment"] for r in rows if r["sentiment"] == "positive"]),
            },
            "rows": rows,
        })
    except (ValueError, KeyError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Analysis failed")
        return jsonify({"error": "Analysis failed. Check the model and CSV, then try again."}), 500


@app.post("/api/predict")
def predict_one():
    payload = request.get_json(silent=True) or {}
    comment = str(payload.get("comment", "")).strip()
    model_name = payload.get("model", "")
    if not comment:
        return jsonify({"error": "Write a comment first."}), 400
    if model_name not in available_models():
        return jsonify({"error": "Select an available model."}), 400
    try:
        return jsonify(predict([comment], model_name).iloc[0].to_dict())
    except Exception:
        app.logger.exception("Single comment prediction failed")
        return jsonify({"error": "The comment could not be classified."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
