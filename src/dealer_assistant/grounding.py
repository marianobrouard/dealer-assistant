"""Check that every number an assistant reply states comes from tool results or the customer.

A number is "grounded" if the same value appears in the evidence texts. Argentine formats are
understood: `27.900.000` (thousands dots), `2,8` (decimal comma), `38 mil`, `27,9 millones`.
"""

import re
from collections.abc import Iterable
from decimal import Decimal

SMALL_NUMBER_LIMIT = 10  # "3 autos", "4x4" or "1.4" are not claims worth checking

_NUMBER = re.compile(
    r"(\d[\d.,]*\d|\d)(\s*(?:millones|millón|millon|mil)\b)?", re.IGNORECASE
)
_THOUSANDS = re.compile(r"^\d{1,3}([.,]\d{3})+$")


def _to_decimal(token: str) -> Decimal:
    if _THOUSANDS.match(token):
        return Decimal(re.sub(r"[.,]", "", token))
    try:
        return Decimal(token.replace(",", "."))
    except ArithmeticError:  # e.g. "10.000,50": fall back to the digits only
        return Decimal(re.sub(r"\D", "", token))


def _scale(suffix: str | None) -> int:
    word = (suffix or "").strip().casefold()
    if word.startswith("mill"):
        return 1_000_000
    return 1_000 if word == "mil" else 1


def extract_numbers(text: str) -> set[Decimal]:
    """Numeric values mentioned in `text`, ignoring very small ones."""
    found: set[Decimal] = set()
    for match in _NUMBER.finditer(text):
        value = _to_decimal(match.group(1)) * _scale(match.group(2))
        if value > SMALL_NUMBER_LIMIT:
            found.add(value)
    return found


def ungrounded_numbers(reply: str, evidence: Iterable[str]) -> list[Decimal]:
    """Numbers in `reply` that appear in none of the `evidence` texts."""
    allowed: set[Decimal] = set()
    for text in evidence:
        allowed |= extract_numbers(text)
    return sorted(number for number in extract_numbers(reply) if number not in allowed)
