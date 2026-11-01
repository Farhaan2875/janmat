# JanMat (जनमत)

JanMat is a sentiment and topic-analysis application for public
e-consultation comments. It helps a policy team understand whether comments
are **negative, neutral, or positive**, which themes appear repeatedly, and
which sentences best represent the main concerns and support.

The project has two interfaces:

- A Streamlit dashboard in `app.py`.
- A plain HTML/CSS/JavaScript interface in `web/`, backed by the Flask API in
  `web_app.py`.

The models run in Python. The browser interface sends comments to the Python
API and renders the returned results on the HTML page.

---

## 1. Model and application architecture

### 1.1 High-level architecture

```text
User
  |
  v
HTML/CSS/JavaScript interface (web/)
  |
  | HTTP requests: /api/models, /api/analyze, /api/predict
  v
Flask API (web_app.py)
  |
  +--> Text preprocessing (src/preprocess.py)
  +--> Sentiment model (src/sentiment.py)
  +--> Topic clustering (src/topics.py)
  +--> Extractive summary (src/summarizer.py)
  |
  v
JSON results returned to the page
```

The application keeps model inference on the Python side because the trained
`.joblib` and `.keras` files are Python/scikit-learn and TensorFlow model
formats. JavaScript is responsible for controls, requests, validation, and
displaying results.

### 1.2 Sentiment labels

Every comment receives one of three labels:

```text
negative
neutral
positive
```

The shared label order is defined in `src/config.py` and is used consistently
by the training scripts, prediction code, and dashboards.

### 1.3 Available models

#### TF-IDF + Logistic Regression

This is the lightweight baseline model.

1. The comment is cleaned, converted to lowercase, and common chat slang is
   expanded (for example, `wtf` becomes `unacceptable` and `lit`/`fire` become
   `excellent`).
2. `TfidfVectorizer` converts words and word pairs into numerical features.
3. Logistic Regression calculates a score for each sentiment class.
4. The class with the highest probability becomes the prediction.

This model is stored at:

```text
models/baseline_tfidf_lr.joblib
```

#### Bi-LSTM

The Bi-LSTM is a neural-network sentiment model built with TensorFlow/Keras.
It reads the sequence of words in both directions:

```text
comment
  -> text vectorization
  -> embedding
  -> bidirectional LSTM
  -> dense classification layer
  -> negative / neutral / positive probabilities
```

The trained model is stored at:

```text
models/bilstm.keras
```

It usually takes longer than the baseline model on CPU, especially the first
time it is loaded. The HTML page displays a loading message while inference is
running.

#### DistilBERT fine-tuned model

If a fine-tuned model has been created in `models/bert/` and the required
packages are installed, JanMat exposes it as:

```text
DistilBERT (fine-tuned)
```

The model tokenizes each comment, creates contextual token representations,
and predicts the sentiment class using its classification head.

#### Pretrained RoBERTa

JanMat can use the pretrained model:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

It does not require local training, but it downloads model files from
Hugging Face the first time it is used. It requires `transformers` and
`torch`.

#### Ensemble

The ensemble averages the probability outputs of every available individual
model:

```text
ensemble probabilities =
    average(baseline, Bi-LSTM, BERT models that are available)
```

The final sentiment is the class with the largest averaged probability. The
ensemble is available when at least two individual models are available.

### 1.4 Model loading and caching

`src/sentiment.py` contains the common `predict(texts, model_name)` function.
Models are loaded lazily and stored in an in-memory cache. This prevents the
application from loading the same large model again for every request.

The main prediction flow is:

```text
web_app.py
  -> available_models()
  -> predict(comments, selected_model)
  -> sentiment and confidence DataFrame
```

### 1.5 Topic analysis

Sentiment and topic analysis are separate operations. After sentiment
prediction:

1. Comments are cleaned.
2. TF-IDF vectors are created.
3. K-means groups similar comments.
4. The best number of clusters is selected automatically using silhouette
   score, unless the user chooses a fixed number.
5. The three highest-weighted terms in each cluster become its display name.

For example, a cluster may be named:

```text
penalties, penalty, breaches
```

The implementation is in `src/topics.py`.

### 1.6 Extractive summaries

The summary is not generated from new text. `src/summarizer.py` selects
sentences that already exist in the comments:

1. Split comments into sentences.
2. Remove very short sentences and duplicates.
3. Convert sentences into TF-IDF vectors.
4. Score sentences by similarity to the group centroid.
5. Select diverse representative sentences.
6. Restore their original order.

This produces separate summaries for negative comments and positive comments.

---

## 2. Code and repository guide

```text
janmat/
├── app.py                         Streamlit dashboard
├── web_app.py                     Flask API and HTML server
├── run_janmat.bat                Windows launcher
├── make_sample_data.py            Creates demo consultation data
├── requirements.txt               Python dependencies
├── data/
│   ├── sample_econsultation_comments.csv
│   └── extra_labeled/             Optional labelled training comments
├── models/
│   ├── baseline_tfidf_lr.joblib
│   ├── bilstm.keras
│   └── metrics and comparison files
├── src/
│   ├── config.py                  Paths, labels, model names
│   ├── preprocess.py              Text cleaning
│   ├── data_loader.py             Dataset loading and normalization
│   ├── sentiment.py               Model discovery and prediction
│   ├── topics.py                  TF-IDF + K-means topics
│   ├── summarizer.py              Extractive summaries
│   ├── train_baseline.py          Baseline training
│   ├── train_lstm.py              Bi-LSTM training
│   ├── train_bert.py              Transformer fine-tuning
│   ├── evaluate.py                Evaluation metrics
│   └── compare_models.py          Model comparison table
├── web/
│   ├── index.html                 Dashboard markup
│   ├── styles.css                 Dashboard styling
│   ├── app.js                     Browser behavior and API calls
│   ├── browser_model.js           Exported browser baseline model
│   └── browser_sample.js          Browser sample data
└── docs/
    └── PROJECT_REPORT.md          Project report
```

### 2.1 `web_app.py`

The Flask server provides:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serves `web/index.html` |
| `/api/models` | GET | Lists models currently available |
| `/api/analyze` | POST | Analyzes sample data or an uploaded CSV |
| `/api/predict` | POST | Classifies one comment |

`/api/analyze` returns:

- Total comments.
- Sentiment counts.
- Average confidence.
- Topic totals and sentiment breakdown.
- Extractive concern and support summaries.
- A row for every comment.

### 2.2 `web/app.js`

The JavaScript frontend:

- Loads available models from `/api/models`.
- Sends selected CSV files and settings to `/api/analyze`.
- Sends one comment to `/api/predict`.
- Renders the mood bar, facts, summaries, topics, and table.
- Supports sentiment filtering and comment search.
- Exports the displayed results as CSV.

When the Python server is unavailable, the page can fall back to the exported
baseline model in `browser_model.js`. The browser-only fallback does not run
the Bi-LSTM or BERT models.

### 2.3 `app.py`

`app.py` is the Streamlit version of the same application. It calls the same
functions from `src/` as the Flask API, but renders controls and charts using
Streamlit components.

### 2.4 Training and evaluation scripts

- `src/train_baseline.py` trains the TF-IDF baseline.
- `src/train_lstm.py` trains the TensorFlow Bi-LSTM.
- `src/train_bert.py` optionally fine-tunes DistilBERT or another transformer.
- `src/evaluate.py` calculates metrics and writes model reports.
- `src/compare_models.py` compares available models across datasets.

### 2.5 Training datasets

The sentiment models are trained from several public datasets combined into
one three-class dataset. The loader maps every source to the shared label
order `0 = negative`, `1 = neutral`, `2 = positive`:

| Dataset | Purpose | Label mapping |
|---|---|---|
| `cardiffnlp/tweet_eval` (`sentiment`) | Real-world social-media posts and slang | Negative, neutral, positive |
| `stanfordnlp/sentiment140` | Large Twitter corpus with informal language, abbreviations, and slang | 0 → negative, 2 → neutral, 4 → positive |
| `zeroshot/twitter-financial-news-sentiment` | Short financial and market-related posts | Bearish → negative, neutral → neutral, bullish → positive |
| `SetFit/sst5` | Stanford Sentiment Treebank sentences | Very negative/negative → negative, neutral → neutral, positive/very positive → positive |
| `takala/financial_phrasebank` | Formal financial-news sentences | Negative, neutral, positive |
| `Yelp/yelp_review_full` | Longer customer reviews | 1–2 stars → negative, 3 stars → neutral, 4–5 stars → positive |
| `stanfordnlp/imdb` | Longer movie reviews | Negative → negative, positive → positive |

Each source is capped during loading so a large dataset does not dominate
training. Dataset downloads are optional: if a source cannot be downloaded,
the loader skips it and continues with the remaining sources. In the latest
training run, `financial_phrasebank` was skipped because the installed
Hugging Face datasets version no longer supports its dataset script.

Sentiment140 uses distant supervision: its original labels are generated from
tweet emoticons rather than human annotation. It improves exposure to informal
language, but its labels can be noisy, so TweetEval and the consultation
sample remain important evaluation sources.

Additional labelled comments can be added under `data/extra_labeled/`; these
are included as `own_data` and repeated during training to emphasize the
consultation domain.

---

## 3. Installation

Use Python 3.10–3.13 on Windows. From the project folder:

```powershell
cd C:\Users\farha\Downloads\janmat
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Verify the environment:

```powershell
python -c "import pandas, flask, numpy, sklearn, joblib; print('core dependencies installed')"
python -c "import tensorflow, transformers, torch; print('AI dependencies installed')"
```

---

## 4. Running the application

### Flask HTML interface

In an activated terminal:

```powershell
python web_app.py
```

Open the application at:

```text
http://127.0.0.1:5000
```

Then select a model and click **Open the record**. Bi-LSTM inference can take
20–30 seconds on CPU the first time it loads.

On Windows, the shortcut is:

```text
run_janmat.bat
```

Double-clicking this file checks the available Python environments, starts the
Flask server, waits for `/api/models` to respond, and opens the application.

### Streamlit interface

```powershell
streamlit run app.py
```

Then open the address shown by Streamlit, normally:

```text
http://localhost:8501
```

### VS Code

1. Open `C:\Users\farha\Downloads\janmat` as the VS Code folder.
2. Open **Terminal → New Terminal**.
3. Activate the environment:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

4. Start the server:

   ```powershell
   python web_app.py
   ```

5. Open `http://127.0.0.1:5000` in a browser or VS Code integrated browser.

Keep the terminal running while using the page.

---

## 5. Training models

The existing model files can be used without retraining. To retrain:

```powershell
python -m src.train_baseline
python -m src.train_lstm
python -m src.compare_models
```

Transformer training is optional:

```powershell
python -m src.train_bert --quick
python -m src.train_bert --base cardiffnlp/twitter-roberta-base-sentiment-latest
```

Full transformer training is better suited to a GPU or Google Colab.

### Adding labelled consultation comments

Place CSV files in:

```text
data/extra_labeled/
```

The expected columns are:

```text
text,label
```

Valid labels are `negative`, `neutral`, and `positive`. Re-run the training
scripts after adding data.

---

## 6. Input CSV requirements

For uploaded consultation data:

- Use one comment per row.
- Include a text column such as `comment`, `text`, `content`, or `feedback`.
- The Flask API requires at least 12 comments for topic analysis.
- A `true_label` column may be included for evaluation workflows.

The included sample file contains 360 simulated consultation comments. It is
for demonstration and transfer testing, not model training.

---

## 7. Limitations and responsible interpretation

- The training datasets are not a perfect representation of policy
  consultations.
- Sentiment confidence is a model probability estimate, not certainty.
- The sample consultation comments are simulated.
- Topic names are generated from high-weight words and may need human review.
- Summaries are extractive and may not capture every important concern.
- BERT/RoBERTa downloads can be large and require an internet connection the
  first time.
- Bi-LSTM and transformer models can be slow on CPU.
- Human judgement remains necessary before using results for policy decisions.

JanMat is a reading aid for consultation data, not an automated decision-maker.
