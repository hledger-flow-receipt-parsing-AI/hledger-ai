"""Cross-check: flag suspicious receipt dates."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from typeguard import typechecked

log = logging.getLogger(__name__)


@typechecked
class DateSanityCheck:
    """Flags receipt dates that are in the future or unreasonably old."""

    name = "date_sanity"

    def __init__(
        self, *, max_age_days: int = 365
    ) -> None:
        self._max_age_days = max_age_days

    def apply(self, merged: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(merged)

        the_date = merged.get("the_date")
        if the_date is None:
            return result

        # Parse string dates
        if isinstance(the_date, str):
            try:
                the_date = datetime.fromisoformat(
                    the_date.replace("Z", "+00:00")
                )
            except ValueError:
                result["_checks"] = result.get("_checks", {})
                result["_checks"]["date_sanity"] = "unparseable"
                return result

        if not isinstance(the_date, datetime):
            return result

        # Strip timezone info for comparison (receipts are local)
        if the_date.tzinfo is not None:
            the_date = the_date.replace(tzinfo=None)

        now = datetime.now()
        result["_checks"] = result.get("_checks", {})

        if the_date > now + timedelta(days=1):
            log.warning(
                "Receipt date is in the future: %s",
                the_date.isoformat(),
            )
            result["_checks"]["date_sanity"] = "future"
        elif the_date < now - timedelta(
            days=self._max_age_days
        ):
            log.warning(
                "Receipt date is older than %d days: %s",
                self._max_age_days,
                the_date.isoformat(),
            )
            result["_checks"]["date_sanity"] = "too_old"
        else:
            result["_checks"]["date_sanity"] = "ok"

        return result
