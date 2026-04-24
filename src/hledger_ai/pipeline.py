"""Extraction pipeline: runs modules in order, merges outputs, runs
cross-checks."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from typeguard import typechecked

from hledger_ai.modules.base import CrossCheck, ExtractionModule

log = logging.getLogger(__name__)


@typechecked
class ExtractionPipeline:
    """Orchestrates extraction modules and cross-checks.

    Modules run sequentially; each receives the merged output of all
    modules that ran before it.  Cross-checks run after all modules.
    """

    def __init__(
        self,
        *,
        modules: List[ExtractionModule],
        cross_checks: List[CrossCheck],
    ) -> None:
        self._modules = modules
        self._cross_checks = cross_checks

    def run(self, *, image_path: str) -> Dict[str, Any]:
        """Run all modules and cross-checks, return merged result."""
        merged: Dict[str, Any] = {}

        for module in self._modules:
            log.info("Running module: %s", module.name)
            try:
                result = module.extract(
                    image_path=image_path,
                    existing=dict(merged),
                )
                merged.update(result)
                log.debug(
                    "Module %s produced keys: %s",
                    module.name,
                    list(result.keys()),
                )
            except Exception:
                log.exception(
                    "Module %s failed, skipping", module.name
                )

        for check in self._cross_checks:
            log.info("Running cross-check: %s", check.name)
            try:
                merged = check.apply(merged)
            except Exception:
                log.exception(
                    "Cross-check %s failed, skipping",
                    check.name,
                )

        return merged
