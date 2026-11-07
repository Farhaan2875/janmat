"""Data-quality checks used before human interpretation of consultation results."""

import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _match_text(text: str) -> str:
    """Normalize text for matching while preserving the original comment."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(text).lower())).strip()


def _groups(values, threshold: float = 0.9):
    """Return connected near-duplicate groups for a bounded set of comments."""
    groups = []
    if len(values) < 2 or not any(values):
        return groups

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    try:
        matrix = vectorizer.fit_transform(values)
    except ValueError:
        return groups
    similarities = cosine_similarity(matrix)
    parent = list(range(len(values)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if similarities[left, right] >= threshold:
                union(left, right)

    grouped = {}
    for index in range(len(values)):
        grouped.setdefault(find(index), []).append(index)
    return [members for members in grouped.values() if len(members) > 1]


def quality_flags(comments, confidences=None, confidence_threshold=0.6):
    """Return row-level quality metadata and aggregate counts.

    Near-duplicate comparisons are intentionally capped to avoid turning an
    upload into an unbounded quadratic operation.
    """
    normalized = [_match_text(comment) for comment in comments]
    exact_counts = Counter(value for value in normalized if value)
    groups = {index: {index} for index in range(len(comments))}

    for value, count in exact_counts.items():
        if count < 2:
            continue
        members = [i for i, item in enumerate(normalized) if item == value]
        for member in members:
            groups[member].update(members)

    if len(comments) <= 2000:
        for members in _groups(normalized):
            for member in members:
                groups[member].update(members)

    rows = []
    for index, comment in enumerate(comments):
        duplicate_count = len(groups[index])
        reasons = []
        if duplicate_count > 1:
            reasons.append("duplicate or near-duplicate")
        if duplicate_count >= 3:
            reasons.append("repeated campaign-style text")
        if confidences is not None and confidences[index] < confidence_threshold:
            reasons.append("low model confidence")
        rows.append({
            "duplicate_group": min(groups[index]) + 1 if duplicate_count > 1 else None,
            "duplicate_count": duplicate_count,
            "review_required": bool(reasons),
            "review_reasons": reasons,
        })

    return rows, {
        "comments": len(comments),
        "unique_comments": len({value for value in normalized if value}),
        "duplicate_comments": sum(row["duplicate_count"] > 1 for row in rows),
        "campaign_groups": len({
            row["duplicate_group"] for row in rows
            if row["duplicate_count"] >= 3
        }),
        "low_confidence_comments": sum(
            "low model confidence" in row["review_reasons"] for row in rows
        ),
    }
