"""Rule-based attributes read from a normalized name: model codes, colors, specs.

These are used as features for matching (a shared model code is strong
evidence; two different capacities are strong counter-evidence), as candidate
generators, and for the color/model extractors.
"""

from __future__ import annotations

import re

from .normalize import STOPWORDS

# A token that is just a quantity with a unit ("4gb", "220v", "50x70cm") is a spec, not a code.
_UNIT_SUFFIX = (
    r"(?:tb|gb|mb|kb|ghz|mhz|hz|mah|mp|mbps|w|v|kg|mg|g|ml|l|mm|cm|m|pol|btus|pecas|fios|"
    r"portas|lugares|pares|rpm|bar|x|p|s|un|unid|ohms|k|lm|db|rms|a|vca|vcc|hp|cv|d)"
)
_SPEC_TOKEN = re.compile(rf"^\d+(?:\.\d+)?(?:x\d+(?:\.\d+)?)*{_UNIT_SUFFIX}?$")
_HAS_LETTER = re.compile(r"[a-z]")
_HAS_DIGIT = re.compile(r"\d")

# Canonical colors. The first 17 are the values used in the dataset's `cor` column.
COLOR_WORDS: dict[str, str] = {
    "preto": "Preto",
    "preta": "Preto",
    "black": "Preto",
    "branco": "Branco",
    "branca": "Branco",
    "white": "Branco",
    "inox": "Inox",
    "azul": "Azul",
    "rosa": "Rosa",
    "pink": "Rosa",
    "prata": "Prata",
    "prateado": "Prata",
    "prateada": "Prata",
    "silver": "Prata",
    "vermelho": "Vermelho",
    "vermelha": "Vermelho",
    "red": "Vermelho",
    "verde": "Verde",
    "bege": "Bege",
    "laranja": "Laranja",
    "amarelo": "Amarelo",
    "amarela": "Amarelo",
    "grafite": "Grafite",
    "marrom": "Marrom",
    "roxo": "Roxo",
    "roxa": "Roxo",
    "marfim": "Marfim",
    "dourado": "Dourada",
    "dourada": "Dourada",
    "cinza": "Cinza",
    "lilas": "Lilás",
    "tabaco": "Tabaco",
    "caramelo": "Caramelo",
}
# two-word colors, matched before single words
COLOR_PHRASES: dict[str, str] = {"aco escovado": "Aço Escovado", "inox escovado": "Inox"}

_VOLTAGE = re.compile(r"^(110|127|220|110220|127220)v$")
_CAPACITY = re.compile(r"^(\d+(?:\.\d+)?)(gb|tb)$")
_SCREEN = re.compile(r"^(\d+(?:\.\d+)?)pol$")
_VOLUME = re.compile(r"^(\d+(?:\.\d+)?)l$")
_WEIGHT = re.compile(r"^(\d+(?:\.\d+)?)kg$")
_BTUS = re.compile(r"^(\d+)btus$")
_MEGAPIXEL = re.compile(r"^(\d+(?:\.\d+)?)mp$")
_DIMENSIONS = re.compile(r"^\d+(?:\.\d+)?(?:x\d+(?:\.\d+)?)+(?:cm|mm|m)?$")


def is_spec(token: str) -> bool:
    return bool(_SPEC_TOKEN.match(token))


def is_code(token: str) -> bool:
    """Model-code-like token: letters and digits mixed (>=4 chars), or a long number."""
    if token.isdigit():
        return len(token) >= 5
    if token in STOPWORDS or is_spec(token):
        return False
    return len(token) >= 4 and bool(_HAS_LETTER.search(token)) and bool(_HAS_DIGIT.search(token))


def extract_codes(norm_name: str) -> list[str]:
    """Model-code-like tokens of a normalized name, in order, without duplicates."""
    seen: dict[str, None] = {}
    for tok in norm_name.split():
        tok = tok.replace(".", "")
        if is_code(tok):
            seen.setdefault(tok, None)
    return list(seen)


def extract_colors(norm_name: str) -> list[str]:
    """Canonical colors mentioned in a normalized name, in order of appearance."""
    found: list[tuple[int, str]] = []
    text = f" {norm_name} "
    for phrase, canon in COLOR_PHRASES.items():
        idx = text.find(f" {phrase} ")
        if idx >= 0:
            found.append((idx, canon))
            text = text.replace(f" {phrase} ", " " + "_" * len(phrase) + " ")
    pos = 0
    for tok in text.split():
        idx = text.find(f" {tok} ", pos)
        pos = max(pos, idx + 1)
        if tok in COLOR_WORDS:
            found.append((idx, COLOR_WORDS[tok]))
    out: list[str] = []
    for _, canon in sorted(found):
        if canon not in out:
            out.append(canon)
    return out


def extract_specs(norm_name: str) -> dict[str, frozenset[str]]:
    """Quantities that tell variants apart: capacity, voltage, screen size, etc."""
    specs: dict[str, set[str]] = {
        k: set()
        for k in ("capacity", "voltage", "screen", "volume", "weight", "btus", "mp", "dims")
    }
    for tok in norm_name.split():
        if m := _CAPACITY.match(tok):
            value = float(m.group(1)) * (1024 if m.group(2) == "tb" else 1)
            specs["capacity"].add(f"{value:g}gb")
        elif _VOLTAGE.match(tok):
            specs["voltage"].add(tok)
        elif _SCREEN.match(tok):
            specs["screen"].add(tok)
        elif _VOLUME.match(tok):
            specs["volume"].add(tok)
        elif _WEIGHT.match(tok):
            specs["weight"].add(tok)
        elif _BTUS.match(tok):
            specs["btus"].add(tok)
        elif _MEGAPIXEL.match(tok):
            specs["mp"].add(tok)
        elif _DIMENSIONS.match(tok):
            specs["dims"].add(tok)
    return {k: frozenset(v) for k, v in specs.items()}


def conflicts(a: frozenset[str], b: frozenset[str]) -> bool:
    """Both sides state a value and they share none."""
    return bool(a) and bool(b) and not (a & b)
