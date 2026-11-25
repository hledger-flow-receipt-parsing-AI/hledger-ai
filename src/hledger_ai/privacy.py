"""Privacy rules of the AI (requirements PR-1 and PR-3).

* Model runtimes must run on this machine: no receipt or transaction data
  is sent anywhere else.
* The AI folder (OCR text, models, logs) is readable by its owner only.
* A model trained on private receipts is never exported.
"""

from __future__ import annotations

import os
import shutil
from urllib.parse import urlparse

from typeguard import typechecked

from hledger_ai.training.state import (
    ENGINE_FILENAME,
    STATE_FILENAME,
    StateStore,
)

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


class PrivacyError(Exception):
    """An action would move private data off this device."""


@typechecked
def assert_local_url(*, url: str) -> None:
    """PR-1: refuse a model runtime that is not on this machine."""
    host = urlparse(url).hostname or ""
    if host not in _LOCAL_HOSTS:
        raise PrivacyError(
            f"Model runtime {url} is not on this machine. Receipts are only"
            " processed locally; use a runtime on localhost."
        )


@typechecked
def make_private(*, folder: str) -> None:
    """Create *folder* (if needed) readable by its owner only."""
    os.makedirs(folder, mode=0o700, exist_ok=True)
    os.chmod(folder, 0o700)


@typechecked
def export_model(
    *, store: StateStore, model: str, dataset_hash: str, destination: str
) -> None:
    """Copy a trained model out of the AI folder.

    PR-3: refused for a model trained on private receipts.
    """
    state = store.load(model=model, dataset_hash=dataset_hash)
    if state is None or not state.finished:
        raise FileNotFoundError(f"{model} is not trained on {dataset_hash}")
    if state.private:
        raise PrivacyError(
            f"{model} was trained on private receipts and is never"
            " exported. Train with --public-only to get a model that can"
            " be shared."
        )
    source = store.folder(model=model, dataset_hash=dataset_hash)
    os.makedirs(destination, exist_ok=True)
    for name in (ENGINE_FILENAME, STATE_FILENAME):
        shutil.copy2(os.path.join(source, name), destination)
