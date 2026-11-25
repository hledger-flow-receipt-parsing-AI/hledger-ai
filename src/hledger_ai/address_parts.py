"""Suggestions for the labeller's manual address questions (CI-6).

When the user picks "manual address" (a shop that is not in the address
list: mostly a shop never labelled before, about a third of the
receipts), the labeller asks for the shop's name, street, house number,
zipcode, city and country. Each gets ranked suggestions from:

* what is printed on the receipt (``address_reader`` and the names,
  streets and cities the labels know, found in the first rows; ST-12),
  with as confidence how often that read was the labelled value;
* the usual city and country of a shop (ST-19), and the country that the
  kind of zipcode points at.

The known shops' own addresses are not offered: picking "manual address"
says the shop is none of them.

Everything is measured as for a shop seen for the first time: every shop
counts once (not every receipt), and a receipt may only use the spellings
of *other* shops. A value offered twice keeps its highest confidence and
lists every source, best first. A part the user typed is never suggested
again (CI-4).
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from typeguard import typechecked

from hledger_ai.address_reader import (
    MAX_READS,
    PARTS,
    READ_PARTS,
    comparable_part,
    read_address,
)
from hledger_ai.models.experts import TrainingExample
from hledger_ai.ocr.types import OcrLine
from hledger_ai.suggest.engine import Suggestion
from hledger_ai.suggest.evidence import ReceiptEvidence

# The labeller's question id of every part.
QUESTION_OF_PART: Dict[str, str] = {
    "shop_name": "shop_name",
    "street": "shop_street",
    "house_nr": "shop_house_nr",
    "zipcode": "shop_zipcode",
    "city": "shop_city",
    "country": "shop_country",
}
_USUAL_PARTS = ("city", "country")
_READ = ("ST-12",)
_USUAL = ("ST-19",)
# Names, streets and cities the labels know are looked for in the first
# rows of the receipt; a read one letter off a known one (an OCR slip) is
# offered in that spelling too.
_KNOWN_SCANNED = ("shop_name", "street", "city")
HEADER_ROWS = 8
_NEAR_MIN_LENGTH = 5
# Letters a read may differ from a known value: 1, or 2 from this length.
_TWO_EDITS_FROM = 12


@typechecked
def zipcode_kind(*, zipcode: str) -> str:
    """ "nl" for 1234AB, "digits" for 2000 or 12345, else "other"."""
    text = zipcode.strip()
    if re.fullmatch(r"[0-9]{4} ?[A-Za-z]{2}", text):
        return "nl"
    if re.fullmatch(r"(?:[A-Za-z]{1,2}-)?[0-9]{4,5}", text):
        return "digits"
    return "other"


@typechecked
def _parts_of(*, facts: Any) -> Dict[str, str]:
    values = {
        "shop_name": facts.shop_name,
        "street": facts.street,
        "house_nr": facts.house_nr,
        "zipcode": facts.zipcode,
        "city": facts.city,
        "country": facts.country,
    }
    return {part: str(value) for part, value in values.items() if value}


@typechecked
def _shop_id(*, facts: Any) -> str:
    """One shop, however its label was written ("9995VV" or "9995 vv")."""
    return "|".join(
        comparable_part(part=part, value=getattr(facts, attribute))
        for part, attribute in (
            ("shop_name", "shop_name"),
            ("zipcode", "zipcode"),
        )
    )


# Not @typechecked: called for every row and known value while training.
def _spaced(*, text: str) -> str:
    """Lower case, no accents, words separated by single spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(re.findall(r"[0-9a-z]+", plain.casefold()))


def _edits_within(left: str, right: str, limit: int) -> bool:
    """Whether at most *limit* letters must change to turn one into the
    other (Levenshtein distance)."""
    if abs(len(left) - len(right)) > limit:
        return False
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (a != b),
                )
            )
        if min(current) > limit:
            return False
        previous = current
    return previous[-1] <= limit


@dataclass(frozen=True)
class _Known:
    """A value the labels know, in the forms it is compared in."""

    spelling: str
    key: str  # comparable_part
    words: str  # _spaced


def _known(*, part: str, spellings: Sequence[str]) -> List[_Known]:
    return [
        _Known(
            spelling=spelling,
            key=comparable_part(part=part, value=spelling),
            words=_spaced(text=spelling),
        )
        for spelling in spellings
    ]


def _respelled(
    *, part: str, value: str, known: Sequence[_Known]
) -> Tuple[str, Optional[str]]:
    """(*value* in its known spelling, a known value one OCR slip away)."""
    key = comparable_part(part=part, value=value)
    for entry in known:
        if entry.key == key:
            return entry.spelling, None
    if part not in _KNOWN_SCANNED or len(key) < _NEAR_MIN_LENGTH:
        return value, None
    limit = 2 if len(key) >= _TWO_EDITS_FROM else 1
    for entry in known:
        if len(entry.key) >= _NEAR_MIN_LENGTH and _edits_within(
            key, entry.key, limit
        ):
            return value, entry.spelling
    return value, None


@dataclass(frozen=True)
class _Located:
    value: str
    photo: int
    lines: Tuple[OcrLine, ...]


def _located_values(
    *, evidence: ReceiptEvidence, part: str, known: Sequence[_Known]
) -> List[_Located]:
    """What the receipt says *part* is, most likely first, with where it
    is printed: known values found in its first rows, then the reader's
    own reads (each followed by a known value one OCR slip away)."""
    found: List[_Located] = []

    def add(value: str, photo: int, lines: Tuple[OcrLine, ...]) -> None:
        key = comparable_part(part=part, value=value)
        if key and all(
            comparable_part(part=part, value=f.value) != key for f in found
        ):
            found.append(_Located(value=value, photo=photo, lines=lines))

    if part in _KNOWN_SCANNED and known and evidence.ocr:
        first = evidence.ocr[0]
        for row, text in zip(
            first.rows()[:HEADER_ROWS], first.row_texts()[:HEADER_ROWS]
        ):
            padded = f" {_spaced(text=text)} "
            hits = [e for e in known if e.words and f" {e.words} " in padded]
            # The longest first: "Voorbeeld Markt Noord" before "Markt".
            for entry in sorted(hits, key=lambda e: -len(e.words)):
                add(entry.spelling, 0, tuple(row))
    for read in read_address(evidence=evidence).get(part, []):
        value, near = _respelled(part=part, value=read.value, known=known)
        add(value, read.photo, read.lines)
        if near is not None:
            add(near, read.photo, read.lines)
    return found[:MAX_READS]


@typechecked
def receipt_values(
    *, evidence: ReceiptEvidence, part: str, known: Sequence[str]
) -> List[str]:
    """What the receipt says *part* is, most likely first (see
    _located_values); *known* are the spellings the labels use."""
    return [
        found.value
        for found in _located_values(
            evidence=evidence,
            part=part,
            known=_known(part=part, spellings=known),
        )
    ]


@typechecked
@dataclass
class AddressKnowledge:
    """What the labelled receipts say about shop addresses, every shop
    counted once."""

    # City and country -> value -> number of shops.
    usual: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Kind of zipcode ("nl", "digits") -> country -> number of shops.
    country_by_zip_kind: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    # Part -> comparable value -> the spelling the labels use most.
    spellings: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # "part:rank" -> [right, read], each shop weighing 1 in total.
    reads: Dict[str, List[float]] = field(default_factory=dict)

    @staticmethod
    def fit(*, examples: Sequence[TrainingExample]) -> "AddressKnowledge":
        knowledge = AddressKnowledge()
        labelled = [e for e in examples if e.facts.shop_name]
        receipts_of: Dict[str, int] = defaultdict(int)
        for example in labelled:
            receipts_of[_shop_id(facts=example.facts)] += 1
        usual: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        by_kind: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        spelled: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        for example in labelled:
            weight = 1.0 / receipts_of[_shop_id(facts=example.facts)]
            parts = _parts_of(facts=example.facts)
            for part in _USUAL_PARTS:
                if part in parts:
                    usual[part][parts[part]] += weight
            if "zipcode" in parts and "country" in parts:
                kind = zipcode_kind(zipcode=parts["zipcode"])
                by_kind[kind][parts["country"]] += weight
            for part, value in parts.items():
                spelled[part][comparable_part(part=part, value=value)][
                    value
                ] += 1
        knowledge.usual = {k: dict(v) for k, v in usual.items()}
        knowledge.country_by_zip_kind = {k: dict(v) for k, v in by_kind.items()}
        knowledge.spellings = {
            part: {
                key: max(counts, key=lambda s: (counts[s], s))
                for key, counts in keys.items()
            }
            for part, keys in spelled.items()
        }
        knowledge._measure_reads(examples=labelled, receipts_of=receipts_of)
        return knowledge

    def _measure_reads(
        self,
        *,
        examples: Sequence[TrainingExample],
        receipts_of: Dict[str, int],
    ) -> None:
        """How often each read is the labelled value, as for a shop seen
        for the first time: a receipt may only use the spellings of other
        shops, and every shop weighs 1 however often it was labelled."""
        shops_of: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for example in examples:
            shop = _shop_id(facts=example.facts)
            for part, value in _parts_of(facts=example.facts).items():
                shops_of[part][comparable_part(part=part, value=value)].add(
                    shop
                )
        known_of = {
            part: _known(
                part=part,
                spellings=list(self.spellings.get(part, {}).values()),
            )
            for part in READ_PARTS
        }
        for example in examples:
            if not example.evidence.ocr:
                continue
            shop = _shop_id(facts=example.facts)
            weight = 1.0 / receipts_of[shop]
            labelled = _parts_of(facts=example.facts)
            for part in READ_PARTS:
                known = [
                    entry
                    for entry in known_of[part]
                    if shops_of[part][entry.key] - {shop}
                ]
                truth = comparable_part(part=part, value=labelled.get(part))
                values = _located_values(
                    evidence=example.evidence, part=part, known=known
                )
                for rank, found in enumerate(values):
                    counts = self.reads.setdefault(f"{part}:{rank}", [0.0, 0.0])
                    right = bool(truth) and (
                        comparable_part(part=part, value=found.value) == truth
                    )
                    counts[0] += weight * right
                    counts[1] += weight

    def read_counts(self, *, part: str, rank: int) -> Tuple[float, float]:
        """(right, read), in shops."""
        right, read = self.reads.get(f"{part}:{rank}", [0.0, 0.0])
        return right, read

    def read_precision(self, *, part: str, rank: int) -> Optional[float]:
        """How often a read at *rank* is right (Laplace-smoothed, so a few
        lucky shops do not make it certain); None if never measured."""
        right, read = self.read_counts(part=part, rank=rank)
        if read <= 0:
            return None
        return (right + 1) / (read + 2)

    def to_json(self) -> Dict[str, Any]:
        return {
            "usual": self.usual,
            "country_by_zip_kind": self.country_by_zip_kind,
            "spellings": self.spellings,
            "reads": self.reads,
        }

    @staticmethod
    def from_json(*, data: Dict[str, Any]) -> "AddressKnowledge":
        return AddressKnowledge(
            usual={
                k: {v: float(n) for v, n in c.items()}
                for k, c in data.get("usual", {}).items()
            },
            country_by_zip_kind={
                k: {v: float(n) for v, n in c.items()}
                for k, c in data.get("country_by_zip_kind", {}).items()
            },
            spellings={
                k: dict(v) for k, v in data.get("spellings", {}).items()
            },
            reads={
                k: [float(n) for n in v]
                for k, v in data.get("reads", {}).items()
            },
        )


class _Pool:
    """Candidates per part; a value keeps its highest confidence and lists
    every source that offered it, best first."""

    def __init__(self, *, answered: Dict[str, str]) -> None:
        self._answered = answered
        self.best: Dict[str, Dict[str, Tuple[str, float, Tuple[str, ...]]]] = (
            defaultdict(dict)
        )

    def offer(
        self, *, part: str, value: str, confidence: float, sources: Tuple
    ) -> None:
        if part in self._answered or not value:
            return
        key = comparable_part(part=part, value=value)
        if not key:
            return
        kept = self.best[part].get(key)
        if kept is None:
            self.best[part][key] = (value, confidence, tuple(sources))
        elif confidence > kept[1]:
            merged = tuple(dict.fromkeys(tuple(sources) + kept[2]))
            self.best[part][key] = (value, confidence, merged)
        else:
            merged = tuple(dict.fromkeys(kept[2] + tuple(sources)))
            self.best[part][key] = (kept[0], kept[1], merged)

    def ranked(self, *, limit: int) -> Dict[str, List[Suggestion]]:
        found: Dict[str, List[Suggestion]] = {}
        for part in PARTS:
            candidates = sorted(
                self.best.get(part, {}).values(),
                key=lambda c: (-c[1], c[0]),
            )
            if candidates:
                found[QUESTION_OF_PART[part]] = [
                    Suggestion(
                        value=value,
                        confidence=min(max(confidence, 0.0), 1.0),
                        sources=sources,
                    )
                    for value, confidence, sources in candidates[:limit]
                ]
        return found


def _known_for(*, knowledge: AddressKnowledge, part: str) -> List[_Known]:
    return _known(
        part=part, spellings=list(knowledge.spellings.get(part, {}).values())
    )


@typechecked
def _offer_reads(
    *,
    pool: _Pool,
    knowledge: AddressKnowledge,
    evidence: ReceiptEvidence,
    answered: Dict[str, str],
) -> None:
    zipcodes: List[str] = []
    for part in READ_PARTS:
        values = _located_values(
            evidence=evidence,
            part=part,
            known=_known_for(knowledge=knowledge, part=part),
        )
        if part == "zipcode":
            zipcodes = [found.value for found in values]
        for rank, found in enumerate(values):
            precision = knowledge.read_precision(part=part, rank=rank)
            if precision is not None:
                pool.offer(
                    part=part,
                    value=found.value,
                    confidence=precision,
                    sources=_READ,
                )
    # The country is not printed, but the kind of zipcode points at it.
    if "zipcode" in answered:
        kind = zipcode_kind(zipcode=answered["zipcode"])
        weight = 1.0
    elif zipcodes:
        kind = zipcode_kind(zipcode=zipcodes[0])
        weight = knowledge.read_precision(part="zipcode", rank=0) or 0.0
    else:
        return
    countries = knowledge.country_by_zip_kind.get(kind, {})
    total = sum(countries.values())
    for country, count in countries.items():
        pool.offer(
            part="country",
            value=country,
            confidence=weight * count / total,
            sources=_READ,
        )


@typechecked
def suggest_address_parts(
    *,
    knowledge: Optional[AddressKnowledge],
    evidence: Optional[ReceiptEvidence],
    answers: Dict[str, Any],
    limit: int = 5,
) -> Dict[str, List[Suggestion]]:
    """Ranked suggestions per manual address question (keyed by the
    labeller's question id); *answers* are the labeller's answers by
    question id."""
    if knowledge is None:
        return {}
    answered = {
        part: str(answers[question]).strip()
        for part, question in QUESTION_OF_PART.items()
        if str(answers.get(question) or "").strip()
    }
    pool = _Pool(answered=answered)
    if evidence is not None and evidence.ocr:
        _offer_reads(
            pool=pool, knowledge=knowledge, evidence=evidence, answered=answered
        )
    for part in _USUAL_PARTS:
        counts = knowledge.usual.get(part, {})
        total = sum(counts.values())
        for value, count in counts.items():
            pool.offer(
                part=part, value=value, confidence=count / total, sources=_USUAL
            )
    return pool.ranked(limit=limit)


@typechecked
def address_highlights(
    *,
    knowledge: Optional[AddressKnowledge],
    evidence: ReceiptEvidence,
    image_paths: Sequence[str],
    suggestions: Dict[str, List[Suggestion]],
) -> Dict[str, List[Any]]:
    """Per manual address question, the boxes where its first suggestion
    was read on the receipt (UI-7); questions whose first suggestion was
    not read there are left out."""
    from hledger_ai.highlights import Highlight

    if knowledge is None:
        return {}
    found: Dict[str, List[Any]] = {}
    for part, question in QUESTION_OF_PART.items():
        ranked = suggestions.get(question) or []
        if part not in READ_PARTS or not ranked:
            continue
        if not set(ranked[0].sources) & set(_READ):
            continue
        key = comparable_part(part=part, value=str(ranked[0].value))
        for located in _located_values(
            evidence=evidence,
            part=part,
            known=_known_for(knowledge=knowledge, part=part),
        ):
            if comparable_part(part=part, value=located.value) != key:
                continue
            if located.photo < len(image_paths):
                found[question] = [
                    Highlight(
                        image_path=image_paths[located.photo],
                        box=line.box,
                        text=line.text,
                    )
                    for line in located.lines
                ]
            break
    return found
