"""Tier-1 priors (TR-1): frequency tables built from the labels.

They rebuild in milliseconds, answer instantly (SP-3) and are the baseline
every heavier model has to beat (EV-3).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from typeguard import typechecked

_SMOOTHING = 1.0  # pseudo-count given to the less specific distribution


@typechecked
class FrequencyPrior:
    """P(target | known value), smoothed towards the overall frequencies.

    With ``hierarchical=True`` a known value ``a:b:c`` that was never seen
    backs off to ``a:b`` and then ``a`` (bookkeeping categories).
    """

    def __init__(
        self,
        *,
        counts: Dict[str, Dict[str, int]],
        totals: Dict[str, int],
        hierarchical: bool,
    ) -> None:
        self._counts = counts  # known value -> target -> count
        self._totals = totals  # target -> count
        self._hierarchical = hierarchical

    @staticmethod
    def fit(
        *, pairs: Iterable[Tuple[str, str]], hierarchical: bool = False
    ) -> "FrequencyPrior":
        counts: Dict[str, Dict[str, int]] = defaultdict(Counter)
        totals: Dict[str, int] = Counter()
        for known, target in pairs:
            totals[target] += 1
            for level in _levels(value=known, hierarchical=hierarchical):
                counts[level][target] += 1
        return FrequencyPrior(
            counts={k: dict(v) for k, v in counts.items()},
            totals=dict(totals),
            hierarchical=hierarchical,
        )

    def marginal(self) -> Dict[str, float]:
        """P(target) over everything seen."""
        total = sum(self._totals.values())
        return {t: n / total for t, n in self._totals.items()} if total else {}

    def distribution(self, *, known: str) -> Dict[str, float]:
        """P(target | known) for every target ever seen."""
        probabilities = self.marginal()
        for level in _levels(value=known, hierarchical=self._hierarchical):
            seen = self._counts.get(level)
            if not seen:
                continue
            size = sum(seen.values())
            probabilities = {
                target: (seen.get(target, 0) + _SMOOTHING * base)
                / (size + _SMOOTHING)
                for target, base in probabilities.items()
            }
        return probabilities

    def ranked(self, *, known: str) -> List[Tuple[str, float]]:
        """Targets with probabilities, most likely first."""
        probabilities = self.distribution(known=known)
        return sorted(probabilities.items(), key=lambda kv: (-kv[1], kv[0]))

    def log_ratios(self, *, known: str) -> Dict[str, float]:
        """log P(target | known) - log P(target): what knowing adds."""
        marginal = self.marginal()
        return {
            target: math.log(probability) - math.log(marginal[target])
            for target, probability in self.distribution(known=known).items()
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            "counts": self._counts,
            "totals": self._totals,
            "hierarchical": self._hierarchical,
        }

    @staticmethod
    def from_json(*, data: Dict[str, Any]) -> "FrequencyPrior":
        return FrequencyPrior(
            counts=data["counts"],
            totals=data["totals"],
            hierarchical=data["hierarchical"],
        )


@typechecked
def _levels(*, value: str, hierarchical: bool) -> List[str]:
    """``a:b:c`` -> ``[a, a:b, a:b:c]`` (least to most specific)."""
    if not hierarchical:
        return [value]
    parts = value.split(":")
    return [":".join(parts[: n + 1]) for n in range(len(parts))]
