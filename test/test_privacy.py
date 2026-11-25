"""Privacy rules: local runtimes only, private folder, no private export
(PR-1, PR-3)."""

import os
import stat
from pathlib import Path

import pytest
from test_app_training import FakeOcr, _ai, _finance_root

from hledger_ai.app import train
from hledger_ai.ollama_client import OllamaClient
from hledger_ai.privacy import (
    PrivacyError,
    assert_local_url,
    export_model,
    make_private,
)
from hledger_ai.training.state import StateStore


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://[::1]:11434/",
    ],
)
def test_runtime_on_this_machine_is_allowed(url: str) -> None:
    assert_local_url(url=url)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:11434",
        "https://api.example.com/v1",
        "http://192.168.1.20:11434",
        "http://localhost.example.com:11434",
    ],
)
def test_runtime_on_another_machine_is_refused(url: str) -> None:
    with pytest.raises(PrivacyError):
        assert_local_url(url=url)


def test_ollama_client_refuses_a_remote_server() -> None:
    with pytest.raises(PrivacyError):
        OllamaClient(base_url="http://ollama.example.com:11434")


def test_ai_folder_is_readable_by_its_owner_only(tmp_path: Path) -> None:
    folder = tmp_path / "ai" / "nested"

    make_private(folder=str(folder))

    assert stat.S_IMODE(os.stat(folder).st_mode) == 0o700


def test_training_leaves_the_ai_folder_owner_only(tmp_path: Path) -> None:
    finance, ai = _finance_root(tmp_path, FakeOcr()), _ai(tmp_path)

    train(finance=finance, ai=ai, names=["priors"], folds=1)

    assert stat.S_IMODE(os.stat(ai.root).st_mode) == 0o700


def test_model_trained_on_private_receipts_cannot_be_exported(
    tmp_path: Path,
) -> None:
    finance, ai = _finance_root(tmp_path, FakeOcr()), _ai(tmp_path)
    state = train(finance=finance, ai=ai, names=["priors"], folds=1)[0]
    destination = tmp_path / "share"

    with pytest.raises(PrivacyError):
        export_model(
            store=StateStore(models_dir=ai.models_dir),
            model="priors",
            dataset_hash=state.dataset_hash,
            destination=str(destination),
        )

    assert not destination.exists()


def test_model_trained_on_the_public_dataset_can_be_exported(
    tmp_path: Path,
) -> None:
    finance, ai = _finance_root(tmp_path, FakeOcr()), _ai(tmp_path)
    state = train(
        finance=finance, ai=ai, names=["priors"], folds=1, public_only=True
    )[0]
    destination = tmp_path / "share"

    export_model(
        store=StateStore(models_dir=ai.models_dir),
        model="priors",
        dataset_hash=state.dataset_hash,
        destination=str(destination),
    )

    assert sorted(p.name for p in destination.iterdir()) == [
        "engine.joblib",
        "state.json",
    ]
