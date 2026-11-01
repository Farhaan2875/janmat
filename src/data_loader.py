"""Loads and combines several real-world sentiment datasets into ONE 3-class training set.

Why several? Tweets alone teach a model slang, not formal language. Public consultation comments are
formal and often long, so we also train on formal / longer text. All labels are mapped to
0 = negative, 1 = neutral, 2 = positive.

  source          what it is                                  classes mapped from
  tweet_eval      real tweets (SemEval-2017 Task 4)           negative/neutral/positive
  sentiment140    large Twitter corpus with distant labels    0 -> negative, 2 -> neutral, 4 -> positive
  twitter_fin     finance news tweets (zeroshot)              bearish -> neg, bullish -> pos, neutral
  sst5            Stanford Sentiment Treebank (movie text)    very neg + neg -> neg, neutral, pos + very pos -> pos
  fin_phrasebank  formal financial news sentences             negative/neutral/positive
  yelp            Yelp reviews (longer text)                  1-2 stars neg, 3 stars neutral, 4-5 pos
  imdb            IMDB movie reviews                          negative / positive (no neutral)
  own_data        your own labelled CSVs in data/extra_labeled   negative/neutral/positive

Every dataset is optional: if one cannot be downloaded it is skipped with a message.
"""
import pandas as pd

from src.config import EXTRA_DIR, EXTRA_REPEAT, MAX_PER_SOURCE, SAMPLE_CSV, SOURCES
from src.preprocess import clean_text

NAME_TO_ID = {"negative": 0, "bearish": 0, "neutral": 1, "positive": 2, "bullish": 2}
STAR_TO_ID = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}          # 5-level scales -> 3 classes
EVAL_CAP = 3000                                       # max validation / test rows per source


# ------------------------------------------------------------------ small helpers
def _df(texts, labels, source):
    return pd.DataFrame({"text": [str(t) for t in texts], "label": list(labels), "source": source})


def _names_to_ids(split, default_names):
    """Read the label names stored in the dataset and map them to our 0/1/2.

    Datasets disagree about label order (e.g. bearish/bullish/neutral), so we never hard-code
    the order if the dataset tells us what it is.
    """
    names = getattr(split.features.get("label"), "names", None)
    if not names or not all(n.lower() in NAME_TO_ID for n in names):
        names = default_names
    print(f"      label order used: {names}")
    return [NAME_TO_ID[names[i].lower()] for i in split["label"]]


def _train_val(df, seed=42):
    """Hold out 10% of a training set as validation."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    cut = int(len(df) * 0.9)
    return df[:cut], df[cut:]


def _split_all(df, seed=42):
    """80 / 10 / 10 split for datasets that ship as a single block."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    a, b = int(len(df) * 0.8), int(len(df) * 0.9)
    return {"train": df[:a], "val": df[a:b], "test": df[b:]}


def _subsample(ds_split, n, seed=42):
    """Random subset of a big HuggingFace split (so Yelp does not download-then-process 650k rows)."""
    n = min(n, len(ds_split))
    return ds_split.shuffle(seed=seed).select(range(n))


# ------------------------------------------------------------------ one function per dataset
def _tweet_eval():
    from datasets import load_dataset
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    g = lambda s: _df(ds[s]["text"], ds[s]["label"], "tweet_eval")      # 0 neg, 1 neu, 2 pos
    return {"train": g("train"), "val": g("validation"), "test": g("test")}


def _sentiment140():
    """Load the large, emoticon-labelled Twitter corpus.

    Sentiment140 commonly stores labels as 0, 2, and 4. Some mirrors expose
    names instead, so support both representations.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("stanfordnlp/sentiment140")
    except Exception as err:
        print(f"      Hugging Face mirror unavailable ({type(err).__name__}); using the original CSV archive.")
        return _sentiment140_csv()

    def label_ids(split):
        names = getattr(split.features.get("sentiment"), "names", None)
        if names:
            mapping = {}
            for index, name in enumerate(names):
                value = str(name).lower()
                if value in NAME_TO_ID:
                    mapping[index] = NAME_TO_ID[value]
                elif value in {"0", "2", "4"}:
                    mapping[index] = {"0": 0, "2": 1, "4": 2}[value]
                else:
                    mapping[index] = None
            if any(value is None for value in mapping.values()):
                raise ValueError(f"Unsupported Sentiment140 labels: {names}")
            return [mapping[int(value)] for value in split["sentiment"]]
        raw = split["sentiment"]
        return [{0: 0, 2: 1, 4: 2}[int(value)] for value in raw]

    def g(name):
        split = ds[name]
        text_column = "text" if "text" in split.column_names else "sentences"
        return _df(split[text_column], label_ids(split), "sentiment140")

    train_name = "train" if "train" in ds else list(ds.keys())[0]
    test_name = "test" if "test" in ds else ("validation" if "validation" in ds else train_name)
    train = g(train_name)
    test = g(test_name)
    if train_name == test_name:
        return _split_all(train)
    tr, va = _train_val(train)
    return {"train": tr, "val": va, "test": test}


def _sentiment140_csv():
    """Fallback for datasets versions that no longer execute HF dataset scripts."""
    import io
    import urllib.request
    import zipfile

    url = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    columns = ["sentiment", "id", "date", "query", "user", "text"]
    archive = urllib.request.urlopen(url, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        train_name = next(name for name in zipped.namelist() if "training" in name.lower())
        train = pd.read_csv(zipped.open(train_name), header=None, names=columns,
                            usecols=["sentiment", "text"], encoding="latin-1")
    train = _df(train["text"], train["sentiment"].map({0: 0, 2: 1, 4: 2}), "sentiment140")
    train = train.sample(n=min(len(train), MAX_PER_SOURCE * 3), random_state=42)
    tr, va = _train_val(train)
    return {"train": tr, "val": va, "test": va.copy()}


def _twitter_fin():
    from datasets import load_dataset
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    default = ["bearish", "bullish", "neutral"]
    train = _df(ds["train"]["text"], _names_to_ids(ds["train"], default), "twitter_fin")
    test = _df(ds["validation"]["text"], _names_to_ids(ds["validation"], default), "twitter_fin")
    tr, va = _train_val(train)
    return {"train": tr, "val": va, "test": test}


def _sst5():
    from datasets import load_dataset
    ds = load_dataset("SetFit/sst5")
    g = lambda s: _df(ds[s]["text"], [STAR_TO_ID[i] for i in ds[s]["label"]], "sst5")
    return {"train": g("train"), "val": g("validation"), "test": g("test")}


def _fin_phrasebank():
    from datasets import load_dataset
    ds = load_dataset("takala/financial_phrasebank", "sentences_allagree")["train"]
    labels = _names_to_ids(ds, ["negative", "neutral", "positive"])
    return _split_all(_df(ds["sentence"], labels, "fin_phrasebank"))


def _yelp():
    from datasets import load_dataset
    ds = load_dataset("Yelp/yelp_review_full")
    def g(split, n):
        sub = _subsample(ds[split], n)
        return _df([t[:600] for t in sub["text"]], [STAR_TO_ID[i] for i in sub["label"]], "yelp")
    tr, va = _train_val(g("train", MAX_PER_SOURCE))
    return {"train": tr, "val": va, "test": g("test", EVAL_CAP)}


def _imdb():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb")
    def g(split, n):
        sub = _subsample(ds[split], n)
        return _df([t[:600] for t in sub["text"]], [0 if i == 0 else 2 for i in sub["label"]], "imdb")
    tr, va = _train_val(g("train", 8000))
    return {"train": tr, "val": va, "test": g("test", 1500)}


def _own_data():
    """Your own labelled comments: CSV files in data/extra_labeled with columns text,label."""
    frames = []
    for path in sorted(EXTRA_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        df["label"] = df["label"].astype(str).str.lower().map(NAME_TO_ID)
        frames.append(_df(df["text"], df["label"], "own_data").dropna())
    if not frames:
        raise FileNotFoundError("no CSV files in data/extra_labeled (optional)")
    parts = _split_all(pd.concat(frames, ignore_index=True))
    parts["train"] = pd.concat([parts["train"]] * EXTRA_REPEAT, ignore_index=True)   # emphasise your domain
    return parts


LOADERS = {"tweet_eval": _tweet_eval, "sentiment140": _sentiment140, "twitter_fin": _twitter_fin, "sst5": _sst5,
           "fin_phrasebank": _fin_phrasebank, "yelp": _yelp, "imdb": _imdb, "own_data": _own_data}


# ------------------------------------------------------------------ public functions
def load_all(sources=None, max_per_source=MAX_PER_SOURCE):
    """Returns (train, val, test) DataFrames with columns: text, label, source, clean."""
    names = list(sources or SOURCES)
    if EXTRA_DIR.exists() and any(EXTRA_DIR.glob("*.csv")) and "own_data" not in names:
        names.append("own_data")

    parts = {"train": [], "val": [], "test": []}
    for name in names:
        print(f"Loading {name} ...")
        try:
            data = LOADERS[name]()
        except Exception as err:   # one broken download must not stop the whole project
            print(f"   [skipped] {name}: {type(err).__name__}: {str(err)[:120]}")
            continue
        for split in parts:
            d = data[split]
            cap = max_per_source if split == "train" else EVAL_CAP
            if len(d) > cap and name != "own_data":
                d = d.sample(n=cap, random_state=42)
            parts[split].append(d)
        print(f"   ok: {len(data['train'])} train / {len(data['val'])} val / {len(data['test'])} test")

    if not parts["train"]:
        raise RuntimeError("No dataset could be loaded. Check your internet connection and `pip install datasets`.")

    out = []
    for split in ["train", "val", "test"]:
        df = pd.concat(parts[split], ignore_index=True)
        df["clean"] = df["text"].apply(clean_text)
        df = df[df["clean"].str.len() > 0]
        if split == "train":   # drop repeated texts, but keep the deliberate copies of your own data
            own = df[df["source"] == "own_data"]
            df = pd.concat([df[df["source"] != "own_data"].drop_duplicates(subset="clean"), own])
        else:
            df = df.drop_duplicates(subset=["clean", "source"])
        df["label"] = df["label"].astype(int)
        if split == "train":
            df = df.sample(frac=1, random_state=42)       # mix the sources together
        out.append(df.reset_index(drop=True))

    train, val, test = out
    print("\nTraining rows by source and class (0 neg, 1 neu, 2 pos):")
    print(pd.crosstab(train["source"], train["label"]).to_string())
    return train, val, test


def load_tweet_eval():
    """Kept for backwards compatibility: tweets only."""
    return load_all(sources=["tweet_eval"])


def load_sample_comments():
    """The e-consultation demo comments (with true labels for evaluation)."""
    df = pd.read_csv(SAMPLE_CSV)
    df["clean"] = df["comment"].apply(clean_text)
    return df
