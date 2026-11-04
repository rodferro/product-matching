"""Text normalization for Portuguese product names.

The same function is applied to listings and to search queries, so that
"Notebook Sony 4 GB", "notebook sony 4gb" and "Notebook Sony 4096 MB" all
end up with the token "4gb".
"""

from __future__ import annotations

import re
import unicodedata

from ftfy import fix_text

# Canonical unit spellings. Keys are regex alternatives, values are the canonical form.
_UNITS = {
    r"tb": "tb",
    r"gb": "gb",
    r"mb": "mb",
    r"kb": "kb",
    r"ghz": "ghz",
    r"mhz": "mhz",
    r"hz": "hz",
    r"mah": "mah",
    r"mp|megapixels?|mpx": "mp",
    r"mbps": "mbps",
    r"w|watts?": "w",
    r"v|volts?": "v",
    r"kg|kgs|quilos?": "kg",
    r"mg": "mg",
    r"g|gr|grs|gramas?": "g",
    r"ml|mls": "ml",
    r"l|lts?|litros?": "l",
    r"mm": "mm",
    r"cm": "cm",
    r"m|metros?": "m",
    r"pol|polegadas?": "pol",
    r"btus?": "btus",
    r"pecas|pcs|pc": "pecas",
    r"fios": "fios",
    r"portas": "portas",
    r"lugares": "lugares",
    r"pares": "pares",
    r"rpm": "rpm",
    r"bar": "bar",
}
_UNIT_LOOKUP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"^(?:{alts})$"), canon) for alts, canon in _UNITS.items()
]
_UNIT_ALT = "|".join(f"(?:{a})" for a in sorted(_UNITS, key=len, reverse=True))
_NUM_UNIT = re.compile(rf"(?<![a-z0-9.])(\d+(?:\.\d+)?(?:x\d+(?:\.\d+)?)*)\s*({_UNIT_ALT})\b")
_THOUSANDS = re.compile(r"(?<=\d)\.(?=\d{3}\b)")
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")
_INCHES = re.compile(r"(\d)\s*(?:\"|''|”|″)")
_MB_TO_GB = re.compile(r"\b(\d+)mb\b")
_DIMS = re.compile(r"(?<=\d)\s*x\s*(?=\d)")
_NON_ALNUM = re.compile(r"[^a-z0-9.\-\s]")
_SPACES = re.compile(r"\s+")

STOPWORDS = frozenset(
    "a o as os e de da do das dos com para p c em no na nos nas por um uma ou s n".split()
)


def fix_encoding(text: str) -> str:
    """Repair mojibake such as 'SÃ©rie' -> 'Série' and HTML entities."""
    return fix_text(text, normalization="NFC")


def strip_accents(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _canonical_unit(unit: str) -> str:
    for pattern, canon in _UNIT_LOOKUP:
        if pattern.match(unit):
            return canon
    return unit


def _merge_units(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        return f"{m.group(1)}{_canonical_unit(m.group(2))}"

    text = _NUM_UNIT.sub(repl, text)

    def mb_to_gb(m: re.Match[str]) -> str:
        n = int(m.group(1))
        if n >= 1024 and n % 1024 == 0:
            return f"{n // 1024}gb"
        return m.group(0)

    return _MB_TO_GB.sub(mb_to_gb, text)


def _join_hyphenated_codes(token: str, keep_parts: bool = False) -> str:
    """'mss-18cr' -> 'mss18cr' (a model code); 'wi-fi' -> 'wi fi' (plain words).

    With keep_parts, a joined code is followed by its parts ('i5-2450m' ->
    'i52450m i5 2450m') so a search for 'core i5' still finds it.
    """
    parts = [p for p in token.split("-") if p]
    if not parts:
        return ""
    joined = "".join(parts)
    if len(parts) > 1 and re.search(r"[a-z]", joined) and re.search(r"\d", joined):
        return " ".join([joined, *parts]) if keep_parts else joined
    return " ".join(parts)


def _clean_dots(token: str) -> str:
    # keep decimal points between digits ("14.2mp"), drop all others
    return re.sub(r"(?<!\d)\.|\.(?!\d)", " ", token)


def normalize_text(text: str | None, keep_code_parts: bool = False) -> str:
    """Lowercase, remove accents, unify units and model codes, drop punctuation.

    >>> normalize_text("Notebook Sony 4 GB")
    'notebook sony 4gb'
    """
    if not text:
        return ""
    t = fix_encoding(text)
    t = strip_accents(t).lower()
    t = _INCHES.sub(r"\1 pol ", t)
    t = _THOUSANDS.sub("", t)
    t = _DECIMAL_COMMA.sub(".", t)
    t = _DIMS.sub("x", t)
    t = _NON_ALNUM.sub(" ", t)
    tokens = []
    for tok in t.split():
        tok = _clean_dots(tok)
        tokens.extend(_join_hyphenated_codes(tok, keep_code_parts).split())
    t = " ".join(tokens)
    t = _merge_units(t)
    return _SPACES.sub(" ", t).strip()


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Split an already normalized string into tokens."""
    toks = text.split()
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def normalize_and_tokenize(text: str | None) -> list[str]:
    return tokenize(normalize_text(text))
