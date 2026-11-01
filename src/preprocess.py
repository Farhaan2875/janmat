"""Text cleaning shared by training and the dashboard."""
import re


SLANG_REPLACEMENTS = {
    "af": " extremely",
    "banger": " excellent",
    "based": " confident",
    "bruh": " friend",
    "btw": " by the way",
    "cap": " lie",
    "cringe": " embarrassing",
    "fire": " excellent",
    "fr": " truly",
    "goated": " excellent",
    "idk": " uncertain",
    "ikr": " agree",
    "imo": " opinion",
    "lit": " excellent",
    "lmao": " laugh",
    "lol": " laugh",
    "mid": " mediocre",
    "ngl": " honestly",
    "omg": " surprise",
    "pls": " please",
    " rn": " now",
    "r": " are",
    "smh": " disappointment",
    "slaps": " excellent",
    "sus": " suspicious",
    "tbh": " honestly",
    "tf": " unacceptable",
    "trash": " terrible",
    "u": " you",
    "ur": " your",
    "w": " win",
    "wtf": " unacceptable",
    "y'all": " you all",
}


def _expand_slang(text: str) -> str:
    """Replace common chat abbreviations with words classifiers can understand."""
    text = re.sub(r"\bno\s+cap\b", " no lie ", text, flags=re.IGNORECASE)
    for slang, replacement in SLANG_REPLACEMENTS.items():
        text = re.sub(rf"(?<![a-z]){re.escape(slang.strip())}(?![a-z])",
                      replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"([!?])\1{2,}", r"\1\1", text)
    return text


def clean_text(text) -> str:
    """Lower-case and strip noise (links, @mentions, odd symbols).

    We deliberately KEEP words like 'not' and 'no' - removing them would flip
    the meaning of a sentence ("not good" -> "good").
    """
    text = _expand_slang(str(text).lower())
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"@\w+", " ", text)               # remove @mentions
    text = text.replace("#", " ")                   # '#budget' -> 'budget'
    text = re.sub(r"[^a-z0-9\s'!?.,]", " ", text)   # drop emojis / symbols
    text = re.sub(r"\s+", " ", text).strip()        # collapse spaces
    return text


def light_clean(text) -> str:
    """Gentler cleaning for transformer models: keep case and punctuation, drop only URLs/@mentions."""
    text = _expand_slang(str(text))
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
