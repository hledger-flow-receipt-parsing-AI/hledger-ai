"""Adapter between the suggestion service and the labelling TUI.

tui-image-labeller opens one ``LabellerSuggester`` per receipt, asks it for
suggestions before the first question and again whenever the user confirms
a field (CI-1), and reports field events for the timing log (SP-4).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence

from hledger_core.AISuggestion import AISuggestion
from typeguard import typechecked

from hledger_ai.candidates.dates import capture_time
from hledger_ai.dataset.examples import cropped_path
from hledger_ai.evaluation.scoring import answer_of
from hledger_ai.highlights import Highlight, field_highlights
from hledger_ai.models.catalogue import build_ocr_engine
from hledger_ai.ocr.cache import OcrCache, ocr_image
from hledger_ai.ocr.types import OcrEngine
from hledger_ai.paths import AiPaths, FinancePaths
from hledger_ai.privacy import make_private
from hledger_ai.receipt_fields import OPTIONAL_AMOUNTS
from hledger_ai.suggest.engine import Suggestion
from hledger_ai.suggest.evidence import ReceiptEvidence
from hledger_ai.suggest.service import (
    DEFAULT_BUDGET_SECONDS,
    SuggestionService,
)
from hledger_ai.timing import FieldTimer, strategy_budgets

log = logging.getLogger(__name__)

MANUAL_ADDRESS = "manual address"
_MONEY = ("amount", "change", "subtotal", "total_tax")
# The suggestion "leave this optional amount empty" (the subtotal).
NOTHING_TEXT = ""
_TEXT = ("category", "bank_account", "currency")
_DATE_SUGGESTIONS = 3
# The labeller question that shows a field's evidence on the photo (UI-7).
_QUESTION_OF_FIELD = {
    "date": "date",
    "time": "date",
    "amount": "amount",
    "change": "change",
    "subtotal": "subtotal",
    "total_tax": "total_tax",
    "category": "category",
    "bank_account": "bank_account",
    "currency": "currency",
    "is_withdrawal": "is_withdrawal",
    "shop": "address_selector",
}

# OCR engines stay loaded between receipts of one labelling session.
_ENGINES: Dict[str, OcrEngine] = {}


@typechecked
def known_from_answers(*, answers: Dict[str, Any]) -> Dict[str, Any]:
    """The labeller's answers (by question id) as confirmed fields."""
    known: Dict[str, Any] = {}
    moment = answers.get("date")
    if isinstance(moment, str) and moment:
        try:
            moment = datetime.fromisoformat(moment)
        except ValueError:
            moment = None
    if isinstance(moment, datetime):
        known["date"] = moment.date()
        known["time"] = moment.strftime("%H:%M")
    if answers.get("is_withdrawal") in ("y", "n"):
        known["is_withdrawal"] = answers["is_withdrawal"] == "y"
    for name in _TEXT:
        text = str(answers.get(name) or "").strip()
        if text:
            known[name] = text
    for name in _MONEY:
        value = answers.get(name)
        if isinstance(value, str):
            value = value.strip().replace(",", ".")  # "6,35" is 6.35
        if value in (None, ""):
            continue
        try:
            known[name] = float(value)
        except (TypeError, ValueError):
            continue
    shop = answers.get("address_selector")
    if shop and shop != MANUAL_ADDRESS:
        known["shop"] = str(shop)
    if answers.get("shop_name"):
        known["shop_name"] = str(answers["shop_name"])
    if answers.get("shop_zipcode"):
        known["address"] = str(answers["shop_zipcode"])
    return known


@typechecked
def _as_ai(*, text: str, suggestion: Suggestion) -> AISuggestion:
    return AISuggestion(
        question=text,
        probability=suggestion.confidence,
        model_name="+".join(suggestion.sources),
    )


@typechecked
def _money_text(*, value: Any) -> str:
    """An amount as the labeller shows it; None (a subtotal the receipt
    does not have) is the empty answer."""
    return NOTHING_TEXT if value is None else f"{float(value):.2f}"


@typechecked
def to_labeller_suggestions(
    *, suggestions: Dict[str, List[Suggestion]]
) -> Dict[str, List[AISuggestion]]:
    """Suggestions keyed and written the way the labeller's questions take
    them: ``receipt_date`` as ``YYYY-MM-DD HH:MM``, money with two decimals,
    the shop under ``address_selector``."""
    converted: Dict[str, List[AISuggestion]] = {}
    times = suggestions.get("time") or []
    clock = times[0].value if times else "00:00"
    dates = [
        s for s in suggestions.get("date", []) if isinstance(s.value, date)
    ]
    if dates:
        converted["receipt_date"] = [
            _as_ai(text=f"{s.value.isoformat()} {clock}", suggestion=s)
            for s in dates[:_DATE_SUGGESTIONS]
        ]
    for name in _MONEY:
        if name in suggestions:
            converted[name] = [
                _as_ai(text=_money_text(value=s.value), suggestion=s)
                for s in suggestions[name]
            ]
    for name in _TEXT + ("is_withdrawal",):
        if name in suggestions:
            converted[name] = [
                _as_ai(text=str(s.value), suggestion=s)
                for s in suggestions[name]
            ]
    if "shop" in suggestions:
        converted["address_selector"] = [
            _as_ai(text=str(s.value), suggestion=s) for s in suggestions["shop"]
        ]
    return converted


@typechecked
def highlights_by_question(
    *, per_field: Dict[str, List[Highlight]]
) -> Dict[str, List[Highlight]]:
    """Highlights keyed by labeller question; the date question shows the
    date and the time, each box once."""
    by_question: Dict[str, List[Highlight]] = {}
    for field, found in per_field.items():
        question_id = _QUESTION_OF_FIELD.get(field)
        if question_id is None:
            continue
        kept = by_question.setdefault(question_id, [])
        kept += [h for h in found if h not in kept]
    return by_question


@typechecked
def _logged(*, value: Any) -> str:
    """A value in the corrections log; an empty optional amount is ""."""
    return NOTHING_TEXT if value is None else str(value)


@typechecked
class LabellerSuggester:
    """Suggestions and timing for one receipt that is being labelled."""

    def __init__(
        self,
        *,
        service: SuggestionService,
        evidence: Dict[str, ReceiptEvidence],
        timer: FieldTimer,
        corrections_path: str,
        image_path: str,
        image_paths: Sequence[str] = (),
    ) -> None:
        self._service = service
        self._evidence = evidence
        self._timer = timer
        self._corrections_path = corrections_path
        self._image_path = image_path
        # The photos read, in the order of the evidence (for highlights).
        self._image_paths = tuple(image_paths)
        # What was suggested before the user answered anything (TR-4).
        self._first_guess: Optional[Dict[str, List[Suggestion]]] = None
        self._latest: Optional[Dict[str, List[Suggestion]]] = None
        # The manual address questions' suggestions, by question id.
        self._latest_parts: Dict[str, List[Suggestion]] = {}

    @staticmethod
    def open(
        *,
        config: Any,
        raw_image_paths: Sequence[str],
        ocr_engines: Optional[Dict[str, OcrEngine]] = None,
    ) -> Optional["LabellerSuggester"]:
        """Load the trained models and read the receipt's photos (from the
        OCR cache when the background job already read them).

        Returns None when nothing is trained yet. *ocr_engines* replaces
        the real OCR engines (tests).
        """
        ai = AiPaths.from_config(config=config)
        finance = FinancePaths.from_config(config=config)
        service = SuggestionService.load(
            ai=ai,
            budget_seconds=strategy_budgets(
                timings_path=ai.timings_path, default=DEFAULT_BUDGET_SECONDS
            ),
        )
        if service.is_empty():
            return None
        make_private(folder=ai.root)
        crops = [
            cropped_path(
                raw_path=raw,
                processed_dir=finance.processed_dir,
                crop_suffix=finance.crop_suffix,
                crop_ext=finance.crop_ext,
            )
            for raw in raw_image_paths
        ]
        crops = [path for path in crops if os.path.isfile(path)]
        cache = OcrCache(cache_dir=ai.ocr_cache_dir)
        captured_at = capture_time(image_paths=list(raw_image_paths))
        evidence: Dict[str, ReceiptEvidence] = {}
        for engine_id in service.ocr_engine_ids():
            engine = (ocr_engines or {}).get(engine_id) or _ENGINES.get(
                engine_id
            )
            if engine is None:
                engine = _ENGINES[engine_id] = build_ocr_engine(
                    engine_id=engine_id
                )
            try:
                results = tuple(
                    ocr_image(image_path=path, engine=engine, cache=cache)
                    for path in crops
                )
            except Exception:  # a broken engine must not stop labelling
                log.exception("OCR with %s failed", engine_id)
                continue
            evidence[engine_id] = ReceiptEvidence(
                ocr=results, captured_at=captured_at
            )
        receipt_id = (
            os.path.splitext(os.path.basename(raw_image_paths[0]))[0]
            if raw_image_paths
            else "unknown"
        )
        return LabellerSuggester(
            service=service,
            evidence=evidence,
            timer=FieldTimer(path=ai.timings_path, receipt_id=receipt_id),
            corrections_path=ai.corrections_path,
            image_path=raw_image_paths[0] if raw_image_paths else "",
            image_paths=crops,
        )

    def suggest(
        self, *, answers: Dict[str, Any]
    ) -> Dict[str, List[AISuggestion]]:
        """Suggestions for every question the user has not answered."""
        suggestions = self._service.suggest(
            evidence=self._evidence,
            known=known_from_answers(answers=answers),
        )
        if self._first_guess is None:
            self._first_guess = suggestions
        self._latest = suggestions
        converted = to_labeller_suggestions(suggestions=suggestions)
        # The manual address questions (CI-6): asked when the shop is not
        # in the address list, so mostly for shops never labelled before.
        try:
            self._latest_parts = self._service.address_parts(
                evidence=self._evidence, answers=answers
            )
        except Exception:  # never let them stop the other suggestions
            log.exception("Manual address suggestions failed")
            self._latest_parts = {}
        for question_id, ranked in self._latest_parts.items():
            converted[question_id] = [
                _as_ai(text=str(s.value), suggestion=s) for s in ranked
            ]
        return converted

    def highlights(self) -> Dict[str, List[Highlight]]:
        """Where on the photos the latest suggestions come from, per
        labeller question, for the TUI to box on the receipt (UI-7)."""
        if self._latest is None:
            return {}
        found = self._service.image_engine(evidence=self._evidence)
        if found is None:
            return {}
        engine_id, engine = found
        boxes = highlights_by_question(
            per_field=field_highlights(
                engine=engine,
                evidence=self._evidence[engine_id],
                image_paths=self._image_paths,
                suggestions=self._latest,
            )
        )
        try:
            boxes.update(
                self._service.address_highlights(
                    evidence=self._evidence,
                    image_paths=self._image_paths,
                    suggestions=self._latest_parts,
                )
            )
        except Exception:  # the other boxes still count
            log.exception("Manual address boxes failed")
        return boxes

    def finish(self, *, answers: Dict[str, Any]) -> int:
        """TR-4: log every first suggestion the user did not keep, with the
        value they accepted instead. Returns how many were logged."""
        accepted: Dict[str, Any] = known_from_answers(answers=answers)
        for field in OPTIONAL_AMOUNTS:
            if field not in accepted and answers.get(field) == NOTHING_TEXT:
                accepted[field] = None  # left empty on purpose
        rows = []
        for field, suggested in (self._first_guess or {}).items():
            if field not in accepted or not suggested:
                continue
            best = suggested[0]
            if answer_of(field=field, value=best.value) == answer_of(
                field=field, value=accepted[field]
            ):
                continue
            rows.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "image_path": self._image_path,
                    "field": field,
                    "suggested": _logged(value=best.value),
                    "accepted": _logged(value=accepted[field]),
                    "confidence": best.confidence,
                    "sources": list(best.sources),
                }
            )
        if rows:
            os.makedirs(
                os.path.dirname(self._corrections_path),
                mode=0o700,
                exist_ok=True,
            )
            with open(self._corrections_path, "a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(rows)

    def answered(self, *, question_id: str, answer: Any) -> None:
        self._timer.answered(question_id=question_id, answer=answer)

    def reached(self, *, question_id: str) -> None:
        self._timer.reached(question_id=question_id)
