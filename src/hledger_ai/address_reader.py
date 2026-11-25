"""The shop's name and address as printed on the receipt (ST-12).

When the user enters a shop by hand ("manual address" in the labeller),
these reads are the suggestions for its name, street, house number,
zipcode and city: about a third of the receipts come from shops that were
never labelled before, so the known shops cannot help there. The rules
below are plain text rules on the OCR rows; how often each read is right
is measured on the labelled receipts (address_parts.AddressKnowledge).

Values are written the way the labels write them: zipcodes ``1234AB``,
streets and cities with letters and spaces only (the labeller's input
rules), all-capital text in title case.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from typeguard import typechecked

from hledger_ai.ocr.types import OcrLine
from hledger_ai.suggest.evidence import ReceiptEvidence

PARTS: Tuple[str, ...] = (
    "shop_name",
    "street",
    "house_nr",
    "zipcode",
    "city",
    "country",
)
# Read parts (the country is never printed; see address_parts).
READ_PARTS: Tuple[str, ...] = PARTS[:5]
MAX_READS = 3  # candidates per part
# The shop name is looked for in the first rows of the first photo.
NAME_ROWS = 5

_LETTER = r"A-Za-zÀ-ÖØ-öø-ÿ"
# 1234 AB; Dutch zipcodes never use SA, SD or SS. Units after a number
# (1500 ML) are not zipcodes.
_NL_ZIP = re.compile(
    r"(?<![0-9A-Za-z])([1-9][0-9]{3}) ?([A-Za-z]{2})(?![A-Za-z])"
)
_NOT_ZIP_LETTERS = {"SA", "SD", "SS", "ML", "CL", "DL", "KG", "GR", "LT", "ST"}
_NOT_ZIP_LETTERS |= {"PK", "CM", "MM", "MG", "KM", "PC"}
# B-2000 Antwerpen, 12345 Beispielstadt: digits, then a name to the end.
_OTHER_ZIP = re.compile(
    r"(?<![0-9.,:/-])(?:[A-Z]{1,2}-)?([0-9]{4,5})\s+"
    rf"([{_LETTER}][{_LETTER}' -]*)$"
)
_STREET = re.compile(
    rf"^\s*([{_LETTER}][{_LETTER}.' -]*[{_LETTER}.])\s*,?\s*"
    r"([0-9]{1,4})\s?([A-Za-z])?(?:\s*[-/]\s*[0-9]{1,4}[A-Za-z]?)?\s*,?\s*$"
)
_CITY = re.compile(rf"^\s*,?\s*([{_LETTER}][{_LETTER}' -]*[{_LETTER}])\s*$")
# Words that make a row anything but a shop's name or address.
_RECEIPT_WORDS = {
    "aantal",
    "artikel",
    "bedankt",
    "bedrag",
    "betaald",
    "betaling",
    "bon",
    "btw",
    "contant",
    "datum",
    "dank",
    "eur",
    "euro",
    "fax",
    "filiaal",
    "iban",
    "kassa",
    "kassabon",
    "kassier",
    "klant",
    "kvk",
    "omschrijving",
    "pin",
    "prijs",
    "retour",
    "subtotaal",
    "tel",
    "telefoon",
    "terminal",
    "tijd",
    "totaal",
    "transactie",
    "vat",
    "wisselgeld",
    "www",
    # Compound words: "Bonnummer 1234" is no street.
    "autorisatiecode",
    "bonnr",
    "bonnummer",
    "kaartnummer",
    "kassanr",
    "kassanummer",
    "klantnummer",
    "medewerker",
    "pasnummer",
    "postbus",
    "terminalnummer",
    "transactienummer",
    "volgnummer",
    # "SINDS 1923 AMBACHTELIJK" is no zipcode and city.
    "established",
    "opgericht",
    "seit",
    "since",
    "sinds",
}
# A web or mail address: "voorbeeld.nl", not "J.de Vries" or "St.Bernard".
_WEB = re.compile(
    r"www|https?:|@|[a-z0-9-]{2,}\.(?:nl|com|de|be|eu|org|net)\b", re.I
)
# Where the address ends on a row that goes on with a phone number.
_REST_OF_ROW = re.compile(r"\b(?:tel|telefoon|fax|btw|kvk)\b", re.I)
_WELCOME = re.compile(
    r"^\s*(welkom|welcome|willkommen)(\s+(bij|at|bei))?\b", re.I
)


@typechecked
@dataclass(frozen=True)
class Read:
    """One value read for one part, and where it was printed."""

    value: str
    photo: int  # index of the photo in the evidence
    lines: Tuple[OcrLine, ...]


# Not @typechecked: called for every value compared while training.
def comparable_part(*, part: str, value: Optional[str]) -> str:
    """*value* as two answers for the same part compare: lower case, no
    accents, letters and digits only ("1234 ab" is "1234AB")."""
    decomposed = unicodedata.normalize("NFKD", value or "")
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^0-9a-z]", "", plain.casefold())


def _words(*, text: str) -> List[str]:
    return re.findall(rf"[{_LETTER}]+", text.casefold())


def _is_receipt_text(*, text: str) -> bool:
    if _WEB.search(text):
        return True
    return any(word in _RECEIPT_WORDS for word in _words(text=text))


def _capitalised(*, text: str) -> str:
    """Title case for text printed in capitals ("MCDONALD'S" ->
    "Mcdonald's", not "Mcdonald'S")."""
    if not (text.isupper() and len(text) > 3):
        return text
    return " ".join(w[:1].upper() + w[1:].lower() for w in text.split(" "))


@typechecked
def _letters_only(*, text: str) -> str:
    """Letters and single spaces (the labeller's street and city rules),
    title case when it was printed in capitals."""
    cleaned = re.sub(rf"[^{_LETTER}]+", " ", text).strip()
    return _capitalised(text=cleaned)


NAME_CHARACTERS = " &'-./"  # besides letters and digits: the NAME rule


@typechecked
def _name(*, text: str) -> str:
    """The labeller's shop name rule: letters, digits and " &'-./" ("DE
    BAKKER: VERS" -> "De Bakker Vers"); a ":" would also break the
    address list, which writes "name: address"."""
    kept = "".join(
        c if c.isalnum() or c in NAME_CHARACTERS else " " for c in text
    )
    cleaned = re.sub(r"\s+", " ", kept).strip(" &'-./")
    return _capitalised(text=cleaned)


@typechecked
@dataclass(frozen=True)
class _Row:
    photo: int
    index: int  # row number in its photo
    lines: Tuple[OcrLine, ...]

    def text(self) -> str:
        return " ".join(line.text for line in self.lines)


@typechecked
def _rows(*, evidence: ReceiptEvidence) -> List[_Row]:
    return [
        _Row(photo=photo, index=index, lines=tuple(row))
        for photo, result in enumerate(evidence.ocr)
        for index, row in enumerate(result.rows())
    ]


@typechecked
def _zipcodes(*, row: _Row) -> List[Tuple[str, str, str]]:
    """(zipcode, text before it, text after it) on *row*, Dutch ones
    first."""
    text = row.text()
    rest = _REST_OF_ROW.search(text)
    if rest is not None:  # "1234 AB Voorbeeldstad Tel 012-3456789"
        text = text[: rest.start()]
    if not text.strip() or _is_receipt_text(text=text):
        return []
    found = []
    for match in _NL_ZIP.finditer(text):
        letters = match.group(2).upper()
        if letters in _NOT_ZIP_LETTERS:
            continue
        found.append(
            (
                f"{match.group(1)}{letters}",
                text[: match.start()],
                text[match.end() :],  # noqa: E203
            )
        )
    if not found:
        other = _OTHER_ZIP.search(text)
        if other is not None:
            found.append(
                (other.group(1), text[: other.start()], " " + other.group(2))
            )
    return found


@typechecked
def _street(*, text: str) -> Optional[Tuple[str, str]]:
    """(street, house number) when *text* is a street and a number."""
    if _is_receipt_text(text=text):
        return None
    match = _STREET.match(text)
    if match is None:
        return None
    street = _letters_only(text=match.group(1))
    if len(re.sub(r"\s", "", street)) < 4:
        return None
    # "Dorpsstr." is written out in the labels: Dorpsstraat.
    street = re.sub(r"(?i)(str)$", lambda m: m.group(1) + "aat", street)
    return street, match.group(2) + (match.group(3) or "")


@typechecked
def _city(*, text: str) -> Optional[str]:
    if _is_receipt_text(text=text):
        return None
    match = _CITY.match(text)
    if match is None or len(_words(text=match.group(1))) > 4:
        return None
    return _letters_only(text=match.group(1))


class _Found:
    def __init__(self) -> None:
        self.reads: Dict[str, List[Read]] = {}

    def add(self, *, part: str, value: str, row: _Row) -> None:
        if not value:
            return
        kept = self.reads.setdefault(part, [])
        key = comparable_part(part=part, value=value)
        if key and all(
            comparable_part(part=part, value=r.value) != key for r in kept
        ):
            kept.append(Read(value=value, photo=row.photo, lines=row.lines))


@typechecked
def _read(*, evidence: ReceiptEvidence) -> Dict[str, List[Read]]:
    rows = _rows(evidence=evidence)
    found = _Found()
    address_rows = set()
    for position, row in enumerate(rows):
        for zipcode, before, after in _zipcodes(row=row):
            address_rows.add(position)
            found.add(part="zipcode", value=zipcode, row=row)
            city = _city(text=after)
            if city is None and position + 1 < len(rows):
                below = rows[position + 1]
                if below.photo == row.photo:
                    city = _city(text=below.text())
                    if city is not None:
                        address_rows.add(position + 1)
            if city is not None:
                found.add(part="city", value=city, row=row)
            street = _street(text=before) if before.strip() else None
            if street is None and position > 0:
                above = rows[position - 1]
                if above.photo == row.photo:
                    street = _street(text=above.text())
                    if street is not None:
                        address_rows.add(position - 1)
            if street is not None:
                found.add(part="street", value=street[0], row=row)
                found.add(part="house_nr", value=street[1], row=row)
    # A street without a zipcode near it: first rows of every photo.
    for position, row in enumerate(rows):
        if position in address_rows or row.index >= NAME_ROWS + 3:
            continue
        street = _street(text=row.text())
        if street is not None:
            address_rows.add(position)
            found.add(part="street", value=street[0], row=row)
            found.add(part="house_nr", value=street[1], row=row)
    # The name: the first rows of the first photo that are no address.
    for position, row in enumerate(rows):
        if row.photo != 0 or row.index >= NAME_ROWS:
            break
        if position in address_rows:
            continue
        text = _WELCOME.sub("", row.text())
        letters = len(re.findall(rf"[{_LETTER}]", text))
        if (
            letters < 3
            or letters < 0.6 * len(text.replace(" ", ""))
            or len(text) > 40
            or _is_receipt_text(text=text)
        ):
            continue
        found.add(part="shop_name", value=_name(text=text), row=row)
    return {
        part: reads[:MAX_READS] for part, reads in found.reads.items() if reads
    }


@typechecked
def read_address(*, evidence: ReceiptEvidence) -> Dict[str, List[Read]]:
    """Per part, the values printed on the receipt, most likely first
    (parts with nothing read are left out). Worked out once per receipt."""
    return evidence.memo(
        key="address_reads", compute=lambda: _read(evidence=evidence)
    )


@typechecked
def respell(
    *, part: str, value: str, spellings: Dict[str, Dict[str, str]]
) -> str:
    """The way the labels write *value* when it was labelled before
    ("VOORBEELDSTAD" -> "Voorbeeldstad")."""
    known = spellings.get(part, {})
    return known.get(comparable_part(part=part, value=value), value)


__all__: Sequence[str] = (
    "PARTS",
    "READ_PARTS",
    "Read",
    "comparable_part",
    "read_address",
    "respell",
)
