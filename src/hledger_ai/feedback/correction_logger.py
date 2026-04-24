"""Log user corrections to AI suggestions for feedback learning.

Each correction is appended as a JSONL line to a corrections file.
These corrections are incorporated into the training dataset during
``--train-models``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from typeguard import typechecked

log = logging.getLogger(__name__)


@typechecked
class CorrectionLogger:
    """Append-only logger for AI suggestion corrections."""

    def __init__(self, *, feedback_dir: str) -> None:
        self._feedback_dir = os.path.expanduser(feedback_dir)
        self._corrections_path = os.path.join(
            self._feedback_dir, "corrections.jsonl"
        )

    def log_correction(
        self,
        *,
        question_id: str,
        ai_suggestion: Optional[str],
        ai_model: Optional[str],
        ai_probability: Optional[float],
        user_answer: str,
        image_path: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single correction where the user overrode the AI.

        Only logs when the user's answer differs from the AI suggestion.
        """
        if ai_suggestion is not None and ai_suggestion == user_answer:
            return  # No correction needed.

        os.makedirs(self._feedback_dir, exist_ok=True)

        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question_id": question_id,
            "ai_suggestion": ai_suggestion,
            "ai_model": ai_model,
            "ai_probability": ai_probability,
            "corrected_category": user_answer,
            "text": user_answer,
        }
        if image_path:
            entry["image_path"] = image_path
        if extra:
            entry.update(extra)

        try:
            with open(self._corrections_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.debug(
                "Logged correction for %s: %s -> %s",
                question_id,
                ai_suggestion,
                user_answer,
            )
        except OSError:
            log.warning(
                "Failed to write correction to %s",
                self._corrections_path,
            )

    def log_receipt_corrections(
        self,
        *,
        ai_suggestions: Dict[str, Any],
        user_answers: Dict[str, str],
        image_path: Optional[str] = None,
    ) -> None:
        """Log all corrections for a receipt labelling session.

        Compares each AI suggestion against the user's actual answer
        and logs corrections where they differ.
        """
        for question_id, answer in user_answers.items():
            ai_sugs = ai_suggestions.get(question_id, [])
            if not ai_sugs:
                continue

            top = ai_sugs[0]
            ai_value = getattr(top, "question", None)
            ai_model = getattr(top, "model_name", None)
            ai_prob = getattr(top, "probability", None)

            self.log_correction(
                question_id=question_id,
                ai_suggestion=ai_value,
                ai_model=ai_model,
                ai_probability=ai_prob,
                user_answer=str(answer),
                image_path=image_path,
            )
