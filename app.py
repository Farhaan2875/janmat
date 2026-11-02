"""JanMat dashboard.   Run:  streamlit run app.py"""
import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from wordcloud import STOPWORDS, WordCloud

from src.config import LABELS, MODEL_DIR, SAMPLE_CSV
from src.preprocess import clean_text
from src.sentiment import available_models, model_accuracy, predict
from src.summarizer import summarize
from src.topics import STOP_WORDS, cluster_topics

st.set_page_config(page_title="JanMat | Public record", page_icon="✦", layout="wide")

# Colours used for each sentiment everywhere on the page
COLORS = {"negative": "#9B4938", "neutral": "#8A7966", "positive": "#526B4B"}
CLOUD_PALETTES = {
    "all": ["#2F241E", "#60483A", "#896B53", "#B28D6C"],
    "negative": ["#9B4938", "#B96850", "#78372D", "#D28B70"],
    "positive": ["#526B4B", "#718966", "#3D5139", "#9CAF88"],
}

# ----------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'DM Sans', 'Segoe UI', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1240px; }
.stApp { background: #F4EFE6; color: #2F241E; }
.stApp > header { background: transparent; }
.block-container:before { content: ''; position: fixed; inset: 0; pointer-events: none; opacity: .14;
  background-image: radial-gradient(#8A7966 .55px, transparent .55px); background-size: 7px 7px; }

.masthead { margin: 0 -4rem 2.6rem; padding: 2.7rem 4rem 2.3rem; background: #2F241E; color: #F8F3EA;
  border-bottom: 5px solid #B8895D; position: relative; overflow: hidden; }
.masthead:after { content: 'PUBLIC RECORD / 01'; position: absolute; right: 4rem; bottom: 1.6rem;
  font: 500 .68rem 'DM Mono', monospace; letter-spacing: .14em; color: #BFA98D; }
.masthead .eyebrow { font: 500 .7rem 'DM Mono', monospace; color: #D0A778; letter-spacing: .16em; text-transform: uppercase; }
.masthead .name { font: 800 3.8rem 'Playfair Display', Georgia, serif; letter-spacing: -1.5px; line-height: 1; }
.masthead .name span { color: #D0A778; font-size: 2.2rem; margin-left: .55rem; }
.masthead .sub { margin-top: .7rem; color: #D9CDBD; font-size: 1rem; max-width: 560px; }
.masthead .rule { width: 52px; height: 3px; background: #B8895D; margin: 1.2rem 0 .8rem; }

.section { font: 700 1.7rem 'Playfair Display', Georgia, serif; color: #2F241E; margin: 2.5rem 0 .45rem;
  letter-spacing: -.3px; }
.section:before { content: '— '; color: #B8895D; }
.note { color: #786957; font-size: .88rem; margin-bottom: .8rem; }

.tally { display: flex; height: 70px; border: 1px solid #CDBEAB; border-radius: 2px; overflow: hidden; margin: .7rem 0 .5rem;
  box-shadow: 0 3px 0 #E3D8C9; }
.tally div { display: flex; align-items: center; padding: 0 1rem; color: #fff; font-weight: 600; font-size: 1.05rem; white-space: nowrap; overflow: hidden; }
.tally small { font-weight: 400; opacity: .85; margin-left: .5rem; font-size: .85rem; }

.facts { display: flex; gap: 0; margin: 1.4rem 0 0; border-top: 1px solid #CDBEAB; }
.fact { flex: 1; padding: .9rem 1rem .4rem 0; }
.fact + .fact { padding-left: 1rem; border-left: 1px solid #CDBEAB; }
.fact b { display: block; font: 700 1.65rem 'Playfair Display', Georgia, serif; color: #2F241E; }
.fact span { color: #786957; font-size: .78rem; line-height: 1.25; display: block; }

.quote { background: #FBF8F1; border: 1px solid #D9CCBA; border-left: 5px solid #2F241E; padding: 1rem 1.2rem; margin: .4rem 0 1rem;
  line-height: 1.6; font: .94rem Georgia, serif; box-shadow: 3px 3px 0 #E3D8C9; }
.quote.negative { border-left-color: #9B4938; }
.quote.positive { border-left-color: #526B4B; }
.quote .who { display: block; font: 600 .68rem 'DM Mono', monospace; letter-spacing: .08em; text-transform: uppercase;
  margin-bottom: .45rem; color: #786957; }

section[data-testid="stSidebar"] { background: #E8DECF; border-right: 1px solid #CDBEAB; }
section[data-testid="stSidebar"] h3 { font: 700 1.35rem 'Playfair Display', Georgia, serif; color: #2F241E; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #786957; }
div[data-testid="stExpander"] { background: #FBF8F1; border: 1px solid #D1C2AF; border-radius: 2px; }
.stButton > button { border-radius: 2px; background: #2F241E; color: #F8F3EA; border: none; font-weight: 600; }
.stButton > button:hover { background: #5A4031; color: #fff; }
.stDownloadButton > button { border-radius: 2px; border-color: #8F6B4E; color: #5A4031; background: #F8F3EA; }
div[data-testid="stDataFrame"] { border: 1px solid #CDBEAB; }
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, textarea { background: #FBF8F1; border-color: #CDBEAB !important; border-radius: 2px !important; }
label, .stCheckbox p, .stRadio p { color: #5A4031 !important; }
@media (max-width: 760px) { .facts { flex-direction: column; } .fact + .fact { border-left: none; padding-left: 0; } }
</style>
<div class="masthead">
  <div class="eyebrow">JanMat / जनमत &nbsp; · &nbsp; Consultation intelligence</div>
  <div class="name">JanMat<span>जनमत</span></div>
  <div class="rule"></div>
  <div class="sub">A living public record of what people said about a draft policy — read by mood, theme, and evidence.</div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- helpers
@st.cache_data(show_spinner="Reading and grouping the comments...")
def analyse(comments: tuple, model_name: str, k: int) -> pd.DataFrame:
    """Sentiment for every comment, then topic clusters. k = 0 means 'choose automatically'."""
    df = pd.DataFrame({"comment": list(comments)})
    df = pd.concat([df, predict(df["comment"], model_name)], axis=1)
    cleaned = [clean_text(c) for c in df["comment"]]
    labels, names = cluster_topics(cleaned, None if k == 0 else k)
    df["topic"] = [names[i] for i in labels]
    return df


def make_cloud(texts, palette):
    """Word cloud image (numpy array) or None if there is nothing to draw."""
    text = " ".join(clean_text(t) for t in texts)
    if len(text.split()) < 5:
        return None
    stop = set(STOPWORDS) | set(STOP_WORDS)
    colour = lambda *args, **kwargs: random.Random(args[0]).choice(palette)  # same word, same colour
    cloud = WordCloud(width=1000, height=380, background_color="white", stopwords=stop, max_words=70,
                      collocations=False, color_func=colour, prefer_horizontal=0.95)
    return cloud.generate(text).to_array()


def quote_box(title, text, kind=""):
    st.markdown(f'<div class="quote {kind}"><span class="who">{title}</span>{text or "Not enough comments to summarise."}</div>',
                unsafe_allow_html=True)


# ----------------------------------------------------------------- sidebar
models = available_models()
with st.sidebar:
    st.markdown("### The reading room")
    st.caption("Choose a record to examine. JanMat keeps the original comments visible alongside every interpretation.")
    source = st.radio("Record", ["Sample consultation", "Upload a CSV file"], label_visibility="collapsed")
    uploaded = None
    if source == "Upload a CSV file":
        uploaded = st.file_uploader("Bring a CSV record", type="csv")

    st.markdown("### Lens")
    model_name = st.selectbox("Model", models, index=len(models) - 1) if models else None
    st.caption("The ensemble averages every available model and is usually the most accurate.")
    auto = st.checkbox("Let JanMat find the themes", value=True)
    k = 0 if auto else st.slider("Number of topics", 3, 8, 5)
    n_sent = st.slider("Sentences in each brief", 2, 5, 3)
    st.markdown("---")
    st.markdown('<div style="font:500 .68rem \'DM Mono\',monospace;letter-spacing:.08em;color:#786957">FIELD NOTE</div>'
                '<div style="font:.83rem Georgia,serif;color:#5A4031;line-height:1.45;margin-top:.35rem">'
                'This is an aid for reading public feedback, not a replacement for human judgement.</div>',
                unsafe_allow_html=True)

if not models:
    st.error("No trained model found yet.")
    st.markdown("Train at least the baseline first, from the project folder:")
    st.code("python -m src.train_baseline", language="bash")
    st.stop()

# ----------------------------------------------------------------- load data
if source == "Sample consultation":
    raw = pd.read_csv(SAMPLE_CSV)
    comment_col = "comment"
elif uploaded is not None:
    raw = pd.read_csv(uploaded)
    text_cols = [c for c in raw.columns if pd.api.types.is_string_dtype(raw[c])]
    if not text_cols:
        st.error("This file has no text column. Add a column that contains the comments.")
        st.stop()
    default = text_cols.index("comment") if "comment" in text_cols else 0
    comment_col = st.sidebar.selectbox("Column with the comments", text_cols, index=default)
else:
    st.info("Upload a CSV file from the sidebar to begin. Each row should hold one public comment.")
    st.stop()

raw = raw.dropna(subset=[comment_col]).reset_index(drop=True)
if len(raw) < 12:
    st.warning("At least 12 comments are needed to group them into topics.")
    st.stop()

result = analyse(tuple(raw[comment_col].astype(str)), model_name, k)
df = pd.concat([raw.drop(columns=[c for c in ["sentiment", "confidence", "topic"] if c in raw.columns]), result.drop(columns="comment")], axis=1)
df["comment"] = raw[comment_col].astype(str)

# ----------------------------------------------------------------- overview
counts = df["sentiment"].value_counts().reindex(["negative", "neutral", "positive"], fill_value=0)
shares = counts / counts.sum() * 100

st.markdown('<div class="section">The temperature of the room</div>', unsafe_allow_html=True)
segments = "".join(
    f'<div style="width:{max(shares[s], 0.0):.1f}%;background:{COLORS[s]}">'
    f'{shares[s]:.0f}% {s}<small>{counts[s]} comments</small></div>'
    for s in ["negative", "neutral", "positive"] if counts[s] > 0
)
st.markdown(f'<div class="tally">{segments}</div>', unsafe_allow_html=True)

by_topic = df.groupby("topic")["sentiment"].value_counts().unstack(fill_value=0).reindex(columns=LABELS, fill_value=0)
by_topic["total"] = by_topic.sum(axis=1)
by_topic["neg_share"] = by_topic["negative"] / by_topic["total"]
hot = by_topic.sort_values("neg_share", ascending=False).index[0]

st.markdown(f"""
<div class="facts">
  <div class="fact"><b>{len(df):,}</b><span>comments analysed</span></div>
  <div class="fact"><b>{df['topic'].nunique()}</b><span>topics found</span></div>
  <div class="fact"><b>{df['confidence'].mean():.0%}</b><span>average model confidence</span></div>
  <div class="fact"><b>{'—' if model_accuracy(model_name) is None else f'{model_accuracy(model_name):.0%}'}</b><span>model accuracy (evaluated)</span></div>
  <div class="fact"><b>{(df['confidence'] < 0.6).sum():,}</b><span>comments to check by hand (low confidence)</span></div>
  <div class="fact"><b style="font-size:1.05rem;line-height:1.9rem">{hot}</b><span>topic with the highest share of negative comments</span></div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------- briefing
st.markdown('<div class="section">A brief from the record</div>', unsafe_allow_html=True)
st.markdown('<div class="note">Extracted from the comments themselves. No new claims are written into the record.</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    quote_box("Main concerns", summarize(df.loc[df.sentiment == "negative", "comment"], n_sent), "negative")
with c2:
    quote_box("Main support", summarize(df.loc[df.sentiment == "positive", "comment"], n_sent), "positive")

# ----------------------------------------------------------------- topics
st.markdown('<div class="section">The subjects people returned to</div>', unsafe_allow_html=True)
order = by_topic.sort_values("total")
fig = go.Figure()
for s in LABELS:
    fig.add_bar(y=order.index, x=order[s], name=s, orientation="h", marker_color=COLORS[s])
fig.update_layout(barmode="stack", height=60 + 52 * len(order), margin=dict(l=0, r=0, t=10, b=0),
                  paper_bgcolor="#F4EFE6", plot_bgcolor="#F4EFE6",
                  legend=dict(orientation="h", y=1.12, x=0, font=dict(color="#5A4031")),
                  xaxis_title="Number of comments",
                  font=dict(family="DM Sans, sans-serif", color="#5A4031"))
st.plotly_chart(fig, width="stretch")

for topic in by_topic.sort_values("total", ascending=False).index:
    sub = df[df.topic == topic]
    row = by_topic.loc[topic]
    with st.expander(f"{topic}  ({int(row['total'])} comments: {int(row['negative'])} negative, "
                     f"{int(row['neutral'])} neutral, {int(row['positive'])} positive)"):
        a, b = st.columns(2)
        with a:
            quote_box("Concerns raised", summarize(sub.loc[sub.sentiment == "negative", "comment"], n_sent), "negative")
        with b:
            quote_box("Support voiced", summarize(sub.loc[sub.sentiment == "positive", "comment"], n_sent), "positive")

# ----------------------------------------------------------------- word clouds
st.markdown('<div class="section">The vocabulary of the record</div>', unsafe_allow_html=True)
tabs = st.tabs(["All comments", "Negative comments", "Positive comments"])
for tab, key in zip(tabs, ["all", "negative", "positive"]):
    with tab:
        subset = df["comment"] if key == "all" else df.loc[df.sentiment == key, "comment"]
        img = make_cloud(subset, CLOUD_PALETTES[key])
        if img is None:
            st.write("Not enough text to draw a word cloud.")
        else:
            st.image(img, width="stretch")

# ----------------------------------------------------------------- explorer
st.markdown('<div class="section">Open the record</div>', unsafe_allow_html=True)
f1, f2, f3 = st.columns([1, 2, 2])
pick_s = f1.multiselect("Sentiment", LABELS, default=LABELS)
pick_t = f2.multiselect("Topic", sorted(df.topic.unique()), default=sorted(df.topic.unique()))
word = f3.text_input("Contains the word")
review_only = st.checkbox("Flag uncertain comments for a human read (confidence below 60%)")
view = df[df.sentiment.isin(pick_s) & df.topic.isin(pick_t)]
if word:
    view = view[view.comment.str.contains(word, case=False, regex=False)]
if review_only:
    view = view[view.confidence < 0.6]
st.dataframe(view, width="stretch", height=330)
st.download_button("Download results as CSV", view.to_csv(index=False).encode("utf-8"), "janmat_results.csv", "text/csv")

# ----------------------------------------------------------------- accuracy check
if "true_label" in df.columns:
    st.markdown('<div class="section">How accurate was the model on this file?</div>', unsafe_allow_html=True)
    known = df.dropna(subset=["true_label"])
    acc = (known["true_label"] == known["sentiment"]).mean()
    st.write(f"The file includes human labels. The {model_name} model agreed with them on **{acc:.1%}** of {len(known)} comments.")
    st.dataframe(pd.crosstab(known["true_label"], known["sentiment"], rownames=["Human label"], colnames=["Model said"]))

# ----------------------------------------------------------------- single comment
st.markdown('<div class="section">Make a margin note</div>', unsafe_allow_html=True)
st.markdown('<div class="note">Test a single sentence without adding it to the record.</div>', unsafe_allow_html=True)
text = st.text_area("Type or paste a comment", height=110, label_visibility="collapsed",
                    placeholder="Example: The compliance deadline is too short for small businesses.")
if st.button("Read this comment") and text.strip():
    one = predict([text], model_name).iloc[0]
    st.markdown(f'<div class="quote {one["sentiment"]}"><span class="who">JanMat reading</span>'
                f'<strong>{one["sentiment"].capitalize()}</strong> &nbsp;·&nbsp; {one["confidence"]:.0%} confidence</div>',
                unsafe_allow_html=True)
