"""The user's own choice of models (HW-5): overrides the automatic choice
of HW-4 for training and for suggestions."""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

from typeguard import typechecked

from hledger_ai.paths import AiPaths

_FILENAME = "selected_models.json"


@typechecked
def _path(*, ai: AiPaths) -> str:
    return os.path.join(ai.root, _FILENAME)


@typechecked
def read_selection(*, ai: AiPaths) -> Optional[List[str]]:
    """The chosen model names, or None when the device decides."""
    path = _path(ai=ai)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return list(json.load(handle)["models"])


@typechecked
def write_selection(*, ai: AiPaths, models: Sequence[str]) -> None:
    """Store the choice; an empty choice goes back to automatic."""
    path = _path(ai=ai)
    if not models:
        if os.path.isfile(path):
            os.remove(path)
        return
    os.makedirs(ai.root, mode=0o700, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"models": list(models)}, handle)
