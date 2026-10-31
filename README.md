# JanMat (जनमत) - Sentiment analysis of public e-consultation comments

Inspired by SIH25035 (MoCA eConsultation). JanMat reads thousands of public comments on a draft
policy, labels each as **positive / neutral / negative**, groups them by **topic**, and shows a
dashboard with a **written summary** and **word clouds** for policy officers.

## Project layout
```
janmat/
  app.py                    Streamlit dashboard
  make_sample_data.py       builds the demo comments (data/sample_econsultation_comments.csv)
  requirements.txt
  src/
    config.py               paths, label order, model names
    preprocess.py           text cleaning
    data_loader.py          downloads and combines 6 real datasets (+ your own CSVs)
    train_baseline.py       TF-IDF + Logistic Regression
    train_lstm.py           Bi-LSTM (TensorFlow/Keras)
    train_bert.py           DistilBERT fine-tuning (stretch goal)
    sentiment.py            predict() used by the dashboard
    topics.py               k-means topic clustering
    summarizer.py           extractive auto-summary
    evaluate.py             metrics printing / saving
    compare_models.py       one table comparing all models on all datasets
  models/                   trained models + metrics_*.json appear here
  docs/PROJECT_REPORT.md    formal report for submission
```

## Setup (Python 3.10 - 3.12)
```bash
python -m venv venv
venv\Scripts\activate          # Windows      (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Run it (in this order)
```bash
python -m src.train_baseline      # ~5 min.  Downloads the datasets on first run (a few hundred MB, mostly Yelp)
python -m src.train_lstm          # ~20-40 min on CPU (about 100k training rows)
python -m src.compare_models      # builds the results table for your report
streamlit run app.py              # opens http://localhost:8501
```

### Run the HTML / CSS / JavaScript interface

JanMat also includes a browser-native frontend in `web/`. It uses a small Flask
API to call the same Python sentiment, topic, and summary functions, so the
model remains in Python while the interface is plain HTML, CSS, and JavaScript.

For Windows, double-click `run_janmat.bat`. It starts the Python model server
and opens the application automatically. Keep the command window open while
using JanMat.

```bash
python web_app.py                 # opens http://127.0.0.1:5000
```

You can also open `web/index.html` directly after starting the server. A
browser cannot launch Python or load the trained models from an HTML file
alone, so `run_janmat.bat` is the required one-click launcher.

The browser interface supports the sample consultation, CSV uploads, model
selection, theme analysis, comment search, CSV export, and single-comment
classification. Streamlit remains available as the original Python dashboard.

### Browser-only mode

The files in `web/` also include an exported browser model:

```text
web/browser_model.js
web/browser_sample.js
```

Open `web/index.html` directly to run sentiment classification without Python,
Flask, or a server. This mode uses the exported TF-IDF + Logistic Regression
model. The original Bi-LSTM, DistilBERT, RoBERTa, and ensemble models require
their Python/TensorFlow/PyTorch runtimes and remain available through
`web_app.py`.
Optional transformer fine-tuning (use Google Colab with a GPU for the full run):
```bash
python -m src.train_bert --quick                                                  # CPU smoke test
python -m src.train_bert --base cardiffnlp/twitter-roberta-base-sentiment-latest  # best accuracy, needs GPU
```
The dashboard offers the **Pretrained RoBERTa** model with no training at all (downloads ~500 MB the first time),
and an **Ensemble** that averages every available model. Train nothing and you still get a strong model;
train everything and the ensemble is usually the most accurate.

## Datasets (all real, all downloaded automatically from the HuggingFace Hub)
| Source | What it adds | Size used |
|---|---|---|
| TweetEval Sentiment (`cardiffnlp/tweet_eval`) | Real tweets, 3 classes | 30k train |
| Twitter Financial News (`zeroshot/twitter-financial-news-sentiment`) | Formal-ish news text, bearish/bullish/neutral | ~9.5k |
| Financial PhraseBank (`takala/financial_phrasebank`, all-agree) | Formal sentences, high-quality labels | ~2.3k |
| SST-5 (`SetFit/sst5`) | Stanford Sentiment Treebank, 5 levels folded to 3 | ~8.5k |
| Yelp reviews (`Yelp/yelp_review_full`) | Longer opinion text, 3 stars = neutral | 30k |
| IMDB (`stanfordnlp/imdb`) | Long reviews (no neutral class) | 8k |
| **Your own comments** (`data/extra_labeled/*.csv`) | The real consultation language - see below | any |

Each dataset is optional: if one cannot be downloaded (for example Financial PhraseBank on some library versions),
it is skipped with a message and training continues. Edit `SOURCES` in `src/config.py` to leave datasets out.
Label orders differ between datasets, so labels are read from each dataset's own metadata, not assumed.

**Why this helps:** tweets teach slang, but consultation comments are formal and long. Mixing in financial news,
reviews and treebank sentences exposes the models to formal, longer text. Per-dataset accuracy is printed so you
can see exactly where each model is strong or weak.

**No open, labelled dataset of policy-consultation comments exists** (I searched). The biggest single accuracy gain
therefore comes from labelling a few hundred real comments yourself:
put a CSV with `text,label` (negative / neutral / positive) in `data/extra_labeled/`
(see the README.txt there) and re-run the training scripts. They are weighted 3x in training.

`data/sample_econsultation_comments.csv` (360 comments) is **simulated** and is never used for training. It is only
a held-out check of how well the models transfer to policy language. To analyse real comments, upload your own CSV
in the dashboard (one comment per row). A `true_label` column adds an accuracy readout.

## Honest limitations
* Even with six datasets, none is policy-consultation text, so accuracy on real consultation comments will be
  lower than the in-domain scores. Add your own labelled comments (above) to close the gap.
* The demo comments are simulated, so scores on them show transfer behaviour, not real-world accuracy.
* Summaries are extractive (existing sentences are selected, nothing is rewritten).
