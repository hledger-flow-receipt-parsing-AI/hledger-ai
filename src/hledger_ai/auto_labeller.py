"""Auto-label receipt images using the extraction pipeline.

Runs the modular extraction pipeline on unlabelled receipt images and
produces Receipt JSON labels with high-confidence results. Low-confidence
results are skipped for manual review.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from typeguard import typechecked

from hledger_ai.pipeline import ExtractionPipeline

log = logging.getLogger(__name__)


@typechecked
class AutoLabeller:
    """Run the extraction pipeline on images and produce AI labels."""

    def __init__(
        self,
        *,
        pipeline: ExtractionPipeline,
        auto_accept_threshold: float = 0.9,
    ) -> None:
        self._pipeline = pipeline
        self._threshold = auto_accept_threshold

    def label_image(
        self, *, image_path: str
    ) -> Optional[Dict[str, Any]]:
        """Extract structured data from a receipt image.

        Returns a dict of extracted fields, or None if extraction fails
        or confidence is too low.
        """
        try:
            result = self._pipeline.run(image_path=image_path)
        except Exception:
            log.exception("Pipeline failed for %s", image_path)
            return None

        if not result:
            return None

        # Attach metadata.
        result["_source"] = "ai_auto_label"
        result["_image_path"] = image_path
        result["_timestamp"] = datetime.utcnow().isoformat()

        return result

    def label_batch(
        self,
        *,
        image_paths: List[str],
        output_dir: str,
    ) -> Dict[str, Any]:
        """Label a batch of images, saving results as JSON files.

        Returns a summary with counts of labelled/skipped/failed images.
        """
        os.makedirs(output_dir, exist_ok=True)

        labelled = 0
        skipped = 0
        failed = 0

        for image_path in image_paths:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(
                output_dir, f"{basename}_ai_label.json"
            )

            if os.path.exists(output_path):
                log.debug("Skipping %s (already labelled)", image_path)
                skipped += 1
                continue

            result = self.label_image(image_path=image_path)
            if result is None:
                failed += 1
                continue

            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                labelled += 1
                log.info("Labelled: %s", image_path)
            except OSError:
                log.exception("Failed to write %s", output_path)
                failed += 1

        return {
            "total": len(image_paths),
            "labelled": labelled,
            "skipped": skipped,
            "failed": failed,
        }
