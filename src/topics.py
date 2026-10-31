"""Topic clustering: TF-IDF vectors + k-means."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score

# Words that appear in nearly every consultation comment say nothing about the topic
EXTRA_STOP = {"draft", "rules", "rule", "ministry", "proposed", "clause", "section", "please",
              "clarify", "citizens", "citizen", "government", "india", "indian", "would", "could"}
STOP_WORDS = list(ENGLISH_STOP_WORDS | EXTRA_STOP)


def cluster_topics(texts, k=None, max_k=6):
    """Group comments by topic. If k is None we pick the k with the best silhouette score.

    Returns (labels, topic_names) where topic_names[i] is a short name like
    "penalties, penalty, breaches" made from the cluster's top words.
    """
    vec = TfidfVectorizer(stop_words=STOP_WORDS, min_df=2, max_df=0.7)
    X = vec.fit_transform(texts)

    if k is None:
        best_k, best_score = 3, -1
        for trial in range(3, min(max_k, X.shape[0] - 1) + 1):
            labels = KMeans(n_clusters=trial, n_init=5, random_state=42).fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = trial, score
        k = best_k

    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    terms = np.array(vec.get_feature_names_out())

    names = []
    for center in km.cluster_centers_:
        top = terms[center.argsort()[::-1][:3]]
        names.append(", ".join(top))
    return km.labels_, names
