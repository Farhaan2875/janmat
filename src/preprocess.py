"""Text cleaning shared by training and the dashboard."""
import re


def clean_text(text) -> str:
    """Lower-case and strip noise (links, @mentions, odd symbols).

    We deliberately KEEP words like 'not' and 'no' - removing them would flip
    the meaning of a sentence ("not good" -> "good").
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"@\w+", " ", text)               # remove @mentions
    text = text.replace("#", " ")                   # '#budget' -> 'budget'
    text = re.sub(r"[^a-z0-9\s'!?.,]", " ", text)   # drop emojis / symbols
    text = re.sub(r"\s+", " ", text).strip()        # collapse spaces
    return text


def light_clean(text) -> str:
    """Gentler cleaning for transformer models: keep case and punctuation, drop only URLs/@mentions."""
    text = re.sub(r"http\S+|www\.\S+", " ", str(text))
    text = re.sub(r"@\w+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
