"""Persistent training state per model and dataset configuration.

One JSON file per (model, dataset hash) records what the model was trained
on (DS-2), how far training and inference are, and how accurate each
strategy is on unseen receipts. The progress view (UI-*) and the engine's
choice of strategy (EV-4) both read it; training resumes from it (BG-1).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from typeguard import typechecked

STATE_FILENAME = "state.json"
ENGINE_FILENAME = "engine.joblib"


@typechecked
@dataclass
class StrategyState:
    infer_done: int = 0  # held-out receipts processed
    infer_total: int = 0
    infer_seconds: float = 0.0
    # fold index -> field -> {"correct", "total", "baseline_correct"}
    folds: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)

    def inference_fraction(self) -> float:
        return self.infer_done / self.infer_total if self.infer_total else 0.0

    def inference_eta_seconds(self) -> Optional[float]:
        if not self.infer_done:
            return None
        left = self.infer_total - self.infer_done
        return left * self.infer_seconds / self.infer_done

    def _sum(self, *, name: str, key: str) -> int:
        return sum(
            int(fields.get(name, {}).get(key, 0))
            for fields in self.folds.values()
        )

    def fields(self) -> List[str]:
        names: List[str] = []
        for fields in self.folds.values():
            names += [name for name in fields if name not in names]
        return names

    def field_accuracy(self, *, field: str) -> Optional[float]:
        total = self._sum(name=field, key="total")
        return self._sum(name=field, key="correct") / total if total else None

    def _average(self, *, key: str) -> Optional[float]:
        rates = []
        for name in self.fields():
            total = self._sum(name=name, key="total")
            if total:
                rates.append(self._sum(name=name, key=key) / total)
        return sum(rates) / len(rates) if rates else None

    def accuracy(self) -> Optional[float]:
        """EV-1: average accuracy over the strategy's improved fields."""
        return self._average(key="correct")

    def baseline_accuracy(self) -> Optional[float]:
        """EV-3: the same for the baseline."""
        return self._average(key="baseline_correct")

    def seconds_per_receipt(self) -> Optional[float]:
        """SP-5: measured inference time per receipt."""
        return self.infer_seconds / self.infer_done if self.infer_done else None


@typechecked
@dataclass
class ModelState:
    model: str
    dataset_hash: str
    private: bool  # PR-3: trained on private data
    model_version: str
    hyperparameters: Dict[str, Any]
    ocr_done: int = 0
    ocr_total: int = 0
    ocr_read: int = 0  # photos actually read (the rest came from the cache)
    ocr_seconds: float = 0.0
    fits_done: int = 0
    fits_total: int = 0
    fit_seconds: float = 0.0
    finished: bool = False
    updated_at: str = ""
    strategies: Dict[str, StrategyState] = field(default_factory=dict)

    def _seconds_left(self) -> Optional[float]:
        left = 0.0
        for done, total, timed, seconds in (
            (self.ocr_done, self.ocr_total, self.ocr_read, self.ocr_seconds),
            (self.fits_done, self.fits_total, self.fits_done, self.fit_seconds),
        ):
            if done < total:
                if not timed:
                    return None  # no speed measured yet
                left += (total - done) * seconds / timed
        return left

    def training_fraction(self) -> float:
        if self.finished:
            return 1.0
        spent = self.ocr_seconds + self.fit_seconds
        left = self._seconds_left()
        if left is None or spent + left <= 0:
            return 0.0
        return spent / (spent + left)

    def training_eta_seconds(self) -> Optional[float]:
        return 0.0 if self.finished else self._seconds_left()

    def is_current(
        self,
        *,
        dataset_hash: str,
        model_version: str,
        hyperparameters: Dict[str, Any],
    ) -> bool:
        """DS-2: trained to the end on exactly this data, version and
        settings."""
        return (
            self.finished
            and self.dataset_hash == dataset_hash
            and self.model_version == model_version
            and self.hyperparameters == hyperparameters
        )

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(*, data: Dict[str, Any]) -> "ModelState":
        values = dict(data)
        values["strategies"] = {
            name: StrategyState(**strategy)
            for name, strategy in data.get("strategies", {}).items()
        }
        return ModelState(**values)


@typechecked
class StateStore:
    """``<models_dir>/<model>/<dataset hash>/`` holds state and engine."""

    def __init__(self, *, models_dir: str) -> None:
        self._dir = os.path.expanduser(models_dir)

    def folder(self, *, model: str, dataset_hash: str) -> str:
        return os.path.join(self._dir, model, dataset_hash)

    def engine_path(self, *, model: str, dataset_hash: str) -> str:
        return os.path.join(
            self.folder(model=model, dataset_hash=dataset_hash),
            ENGINE_FILENAME,
        )

    def load(self, *, model: str, dataset_hash: str) -> Optional[ModelState]:
        path = os.path.join(
            self.folder(model=model, dataset_hash=dataset_hash), STATE_FILENAME
        )
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as handle:
            return ModelState.from_json(data=json.load(handle))

    def save(self, *, state: ModelState) -> None:
        folder = self.folder(model=state.model, dataset_hash=state.dataset_hash)
        os.makedirs(folder, mode=0o700, exist_ok=True)
        state.updated_at = datetime.now().isoformat(timespec="seconds")
        temporary = os.path.join(folder, STATE_FILENAME + ".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(state.to_json(), handle, indent=1)
        os.replace(temporary, os.path.join(folder, STATE_FILENAME))
