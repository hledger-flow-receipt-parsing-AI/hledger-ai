"""Dataset configurations (requirements DS-1, DS-3, DS-4, DS-5).

A dataset configuration is one exact set of labels, identified by a hash.
The registry remembers every configuration in the order it was first seen.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from typeguard import typechecked

from hledger_ai.receipt_fields import LABEL_SCHEMA_VERSION

# Label keys that are not label data.
_NOT_LABEL_DATA = ("config",)


@typechecked
@dataclass(frozen=True)
class DatasetConfiguration:
    dataset_hash: str
    private: bool  # DS-5: contains private data
    receipt_count: int  # receipts, or categorised rows for transactions
    registered_at: str  # DS-4: when it was first registered
    kind: str = "receipts"  # receipts | transactions


@typechecked
def dataset_hash(
    *,
    labels: Iterable[Dict[str, Any]],
    private: bool,
    schema_version: str = LABEL_SCHEMA_VERSION,
) -> str:
    """DS-1: hash over the label data and the label-schema version."""
    rows: List[str] = []
    for label in labels:
        data = {k: v for k, v in label.items() if k not in _NOT_LABEL_DATA}
        rows.append(json.dumps(data, sort_keys=True, default=str))
    digest = hashlib.sha256()
    digest.update(f"schema={schema_version};private={private}\n".encode())
    for row in sorted(rows):
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@typechecked
class DatasetRegistry:
    """Chronological list of dataset configurations in one JSON file."""

    def __init__(self, *, path: str) -> None:
        self._path = path

    def configurations(self) -> List[DatasetConfiguration]:
        if not os.path.isfile(self._path):
            return []
        with open(self._path, encoding="utf-8") as handle:
            rows = json.load(handle)
        return [DatasetConfiguration(**row) for row in rows]

    def register(
        self,
        *,
        dataset_hash: str,
        private: bool,
        receipt_count: int,
        now: Optional[str] = None,
        kind: str = "receipts",
    ) -> DatasetConfiguration:
        """Add the configuration unless it is already known (DS-3, DS-4)."""
        known = self.configurations()
        for configuration in known:
            if configuration.dataset_hash == dataset_hash:
                return configuration
        new = DatasetConfiguration(
            dataset_hash=dataset_hash,
            private=private,
            receipt_count=receipt_count,
            registered_at=now or datetime.now().isoformat(timespec="seconds"),
            kind=kind,
        )
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        temporary = self._path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump([asdict(c) for c in known + [new]], handle, indent=2)
        os.replace(temporary, self._path)
        return new
