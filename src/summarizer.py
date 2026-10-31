"""Extractive auto-summary: pick the sentences that best represent a group of comments."""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def summarize(comments, n_sentences=3) -> str:
    # 1) split every comment into sentences (ignore very short ones)
    sentences = []
    for comment in comments:
        for s in re.split(r"(?<=[.!?])\s+", str(comment)):
            if len(s.split()) >= 5:
                sentences.append(s.strip())
    sentences = list(dict.fromkeys(sentences))   # remove exact duplicates, keep order
    if len(sentences) <= n_sentences:
        return " ".join(sentences)

    # 2) score each sentence by similarity to the "average" sentence of the group
    X = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    centroid = np.asarray(X.mean(axis=0))
    scores = cosine_similarity(X, centroid).ravel()

    # 3) walk down the ranking, skipping sentences that repeat one we already picked
    sims = cosine_similarity(X)
    chosen = []
    for i in np.argsort(scores)[::-1]:
        if all(sims[i][j] < 0.6 for j in chosen):
            chosen.append(i)
        if len(chosen) == n_sentences:
            break

    # 4) put them back in their original order so the summary reads naturally
    return " ".join(sentences[i] for i in sorted(chosen))
