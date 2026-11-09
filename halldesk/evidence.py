"""Check that a quoted excerpt really occurs in the resident's report."""

import unicodedata

QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def normalize(text):
    """Ignore case, spacing and typographic quotes. Wording must still match."""
    return " ".join(unicodedata.normalize("NFKC", text).translate(QUOTES).casefold().split())


def quoted_in(excerpt, message):
    excerpt = normalize(excerpt or "")
    return bool(excerpt) and excerpt in normalize(message)
