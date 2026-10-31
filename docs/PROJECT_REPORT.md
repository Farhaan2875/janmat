# JanMat: Sentiment Analysis of Public E-Consultation Comments

**Course Outcome:** CO1  **Bloom's level:** L6 (Create)
**Inspired by:** Smart India Hackathon problem statement SIH25035, MoCA eConsultation platform

## 1. Problem statement
Ministries publish draft policies online and invite public comments. Reading thousands of comments by hand
is slow and inconsistent between readers. A policy officer needs a fast, repeatable view of how people feel
and which issues they raise.

## 2. Objectives
1. Clean and classify each comment as positive, neutral or negative.
2. Compare a classical baseline (TF-IDF + Logistic Regression) with a deep model (Bi-LSTM), and optionally BERT.
3. Group comments by topic using k-means.
4. Produce an automatic summary and word clouds in a dashboard for policy officers.

## 3. Datasets
A single dataset of tweets teaches informal language, whereas consultation comments are formal and often long.
Six public datasets were therefore combined, with every label mapped to negative / neutral / positive:

| Dataset | Domain | Label mapping |
|---|---|---|
| TweetEval Sentiment (SemEval-2017 Task 4) | Tweets | native 3 classes |
| Twitter Financial News Sentiment | Finance news tweets | bearish = negative, bullish = positive |
| Financial PhraseBank (all-agree subset) | Formal financial news | native 3 classes |
| Stanford Sentiment Treebank (SST-5) | Movie review sentences | very negative and negative = negative; positive and very positive = positive |
| Yelp Reviews (full) | Longer reviews | 1-2 stars negative, 3 stars neutral, 4-5 stars positive |
| IMDB | Long movie reviews | negative / positive only |

Each source is capped (30,000 training rows) so no dataset dominates, and class weights offset remaining imbalance.
Validation and test splits are kept per dataset, so accuracy is reported for each domain separately.
Optionally, labelled real comments placed in `data/extra_labeled/` are added and weighted 3x.

**Simulated e-consultation sample (360 comments).** Written for a fictional draft data-governance rule with five
themes (privacy and consent, burden on startups and MSMEs, penalties and enforcement, grievance redressal,
implementation). Real portal comments are not available as an open dataset, and a search of public sources found no
open labelled English policy-consultation sentiment dataset. The sample is never used for training; it measures
transfer to policy language and demonstrates the dashboard.

## 4. Methodology
**Pre-processing.** Lower-casing, removal of URLs, @mentions and symbols. Negation words are kept.

**Model 1, baseline.** TF-IDF (unigrams and bigrams, 100k features) followed by Logistic Regression with
balanced class weights.

**Model 2, Bi-LSTM.** Text vectoriser (20k words, length 80), 128-dimension embedding, bidirectional LSTM
(64 units), dropout 0.4, dense layer, 3-way softmax. Class weights offset the large neutral class. Early
stopping on validation loss.

**Model 3, transformer (stretch).** DistilBERT, or a RoBERTa model that is already tuned for sentiment, fine-tuned on the combined data with the HuggingFace Trainer.

**Model 4, pretrained RoBERTa.** `cardiffnlp/twitter-roberta-base-sentiment-latest` used without further training (trained on about 124M tweets and tuned on TweetEval).

**Model 5, ensemble.** The average of the class probabilities of all available models.

**Topic clustering.** TF-IDF on cleaned comments, with common policy words removed, then k-means. The number of
topics is chosen by silhouette score (3 to 6) or set by the user. Each topic is named by its three highest
weighted words.

**Summarisation.** Comments are split into sentences, each sentence is scored by cosine similarity to the
centroid of the group, and the top sentences that are not near-duplicates are shown in original order.
Summaries are produced for negative and positive comments overall and for each topic.

**Dashboard.** Streamlit: sentiment tally, topic breakdown, per-topic summaries, word clouds, comment explorer
with CSV download, accuracy check when human labels are present, and a single-comment tester.

## 5. Results
Run the training scripts and copy the numbers from `models/metrics_*.json` into this table.

Run `python -m src.compare_models` and paste `models/model_comparison.csv` here (overall accuracy and macro-F1,
accuracy per dataset, and accuracy on the e-consultation sample).

| Model | Overall acc. | Overall macro-F1 | Tweets | Fin. news | Fin. PhraseBank | SST-5 | Yelp | IMDB | E-consult. sample |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | | | | | | | | | |
| Bi-LSTM | | | | | | | | | |
| Fine-tuned transformer (optional) | | | | | | | | | |
| Pretrained RoBERTa | | | | | | | | | |
| Ensemble | | | | | | | | | |

Topic clustering on the sample was checked by cross-tabulating clusters against the five known themes. In a
development run with the automatic setting, every cluster mapped to one theme and the largest theme was split across
two or three clusters, so the clusters were consistent with the five themes.

## 6. Discussion
* Compare the three rows above; note where the deep models beat the baseline and by how much.
* Compare the per-dataset columns with the e-consultation column. A drop indicates domain shift from the
  training data to policy language, which is the main limitation of this system. Discuss which formal-text
  datasets (Financial PhraseBank, Twitter Financial News) transfer best.
* Neutral comments (questions, clarification requests) are the hardest class for every model.

## 7. Limitations and future work
* Fine-tune on labelled real consultation comments (a few hundred hand-labelled comments is a realistic start).
* Support Hindi and code-mixed comments using a multilingual model such as XLM-R or MuRIL.
* Replace extractive summaries with an abstractive model.
* Add sarcasm handling and detection of duplicate or campaign-style comments.

## 8. How to run
See `README.md`.

## 9. References
* Rosenthal, Farra, Nakov (2017). SemEval-2017 Task 4: Sentiment Analysis in Twitter.
* Barbieri et al. (2020). TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification.
* Malo et al. (2014). Good debt or bad debt: detecting semantic orientations in economic texts (Financial PhraseBank).
* Socher et al. (2013). Recursive deep models for semantic compositionality over a sentiment treebank.
* Maas et al. (2011). Learning word vectors for sentiment analysis (IMDB).
* Loureiro et al. (2022). TimeLMs: Diachronic language models from Twitter.
* Hochreiter and Schmidhuber (1997). Long Short-Term Memory.
* Sanh et al. (2019). DistilBERT, a distilled version of BERT.
* Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
