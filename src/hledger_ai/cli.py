"""Command line of hledger-ai.

``--train`` trains everything that is out of date; names limit it to those
models and strategies (TR-2, TR-3).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, Optional, Sequence, TextIO

from typeguard import typechecked

from hledger_ai.app import (
    UnknownName,
    benchmarks,
    default_models,
    resolve_names,
    train,
)
from hledger_ai.background import (
    SERVICE_NAME,
    lower_priority,
    power_pause_reason,
    read_unlabelled_receipts,
    systemd_units,
)
from hledger_ai.dataset.configuration import DatasetRegistry
from hledger_ai.hardware import Device, probe_device, render_model_table
from hledger_ai.models.catalogue import MODELS, build_ocr_engine, get_model
from hledger_ai.ocr.types import OcrEngine
from hledger_ai.paths import AiPaths, FinancePaths
from hledger_ai.progress.format import format_eta, format_percent
from hledger_ai.progress.view import run_view
from hledger_ai.selection import write_selection
from hledger_ai.strategies import STRATEGIES
from hledger_ai.suggest.service import SuggestionService
from hledger_ai.training.state import ModelState, StateStore
from hledger_ai.transactions.dataset import TransactionData
from hledger_ai.transactions.model import NotEnoughData


@typechecked
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hledger-ai",
        description="Local receipt AI: train, inspect and run the models.",
    )
    parser.add_argument("--config", required=True, help="path to config.yaml")
    parser.add_argument(
        "--train",
        nargs="*",
        metavar="NAME",
        default=None,
        help=(
            "train what is out of date; with names, only those models"
            " (e.g. rapidocr-linear) and strategies (e.g. ST-1)"
        ),
    )
    parser.add_argument(
        "--public-only",
        action="store_true",
        help="train on the public dataset only (no private receipts)",
    )
    parser.add_argument(
        "--force", action="store_true", help="train even when up to date"
    )
    parser.add_argument(
        "--folds", type=int, default=4, help="time-ordered evaluation folds"
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="list the models with their requirements on this device",
    )
    parser.add_argument(
        "--select",
        nargs="*",
        metavar="MODEL",
        default=None,
        help=(
            "use only these models from the --models list (no names: let"
            " the device decide again)"
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="show training progress and accuracy per model and strategy",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="with --status: print the table once instead of the live view",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="read the receipts that still need a label, in the background",
    )
    parser.add_argument(
        "--on-battery",
        action="store_true",
        help="with --infer: keep reading while on battery power",
    )
    parser.add_argument(
        "--install-service",
        action="store_true",
        help=(
            "write systemd user units that run --train --infer after every"
            " boot and every six hours"
        ),
    )
    parser.add_argument(
        "--install-qvac",
        action="store_true",
        help="install the qvac runtime (about 3.5 GB) after a disk check",
    )
    parser.add_argument(
        "--service-dir",
        default="~/.config/systemd/user",
        help="where --install-service writes the units",
    )
    return parser


@typechecked
def format_progress(*, state: ModelState) -> str:
    """One line per training step for a terminal or a log."""
    parts = (
        [f"photos {state.ocr_done}/{state.ocr_total}"]
        if state.ocr_total
        else []
    )
    parts.append(f"fits {state.fits_done}/{state.fits_total}")
    parts.append(format_percent(fraction=state.training_fraction()))
    eta = state.training_eta_seconds()
    if not state.finished and eta is not None:
        parts.append(f"{format_eta(seconds=eta)} left")
    return f"{state.model}: " + ", ".join(parts)


@typechecked
def _summary(*, state: ModelState) -> str:
    scored = [
        f"{name} {format_percent(fraction=strategy.accuracy())}"
        for name, strategy in state.strategies.items()
        if strategy.accuracy() is not None
    ]
    kind = "private" if state.private else "public"
    unit = (
        "transactions"
        if get_model(name=state.model).data == "transactions"
        else "receipts"
    )
    return (
        f"{state.model}: trained on {kind} dataset"
        f" {state.dataset_hash[:12]}; accuracy on unseen {unit}: "
        + (", ".join(scored) or f"not enough {unit} to measure")
    )


@typechecked
def _show_models(
    *,
    ai: AiPaths,
    device: Device,
    out: Callable[[str], None],
    colour: bool,
) -> None:
    """HW-5: every model with its requirements against this device."""
    out(render_model_table(specs=MODELS, device=device, colour=colour))
    chosen = default_models(ai=ai, device=device, announce=out)
    out("Used by default on this device: " + (", ".join(chosen) or "none"))
    measured = benchmarks(ai=ai, specs=[])
    if measured:
        out(
            "Measured seconds per photo: "
            + ", ".join(f"{n} {s:.1f}" for n, s in sorted(measured.items()))
        )


@typechecked
def _show_status(*, ai: AiPaths, once: bool, stream: TextIO) -> None:
    store = StateStore(models_dir=ai.models_dir)
    run_view(
        load_configurations=DatasetRegistry(
            path=ai.registry_path
        ).configurations,
        load_states=lambda configuration: {
            spec.name: store.load(
                model=spec.name, dataset_hash=configuration.dataset_hash
            )
            for spec in MODELS
            if spec.data == configuration.kind
        },
        strategy_ids=[s.id for s in STRATEGIES if s.applicable],
        once=once,
        out=stream,
    )


@typechecked
def _infer(
    *,
    finance: FinancePaths,
    ai: AiPaths,
    out: Callable[[str], None],
    ocr_engines: Optional[Dict[str, OcrEngine]],
    on_battery: bool,
) -> None:
    """SP-1: read the receipts without a label ahead of labelling."""
    if ocr_engines is None:
        engine_ids = SuggestionService.load(ai=ai).ocr_engine_ids() or [
            get_model(name=name).ocr_engine
            for name in default_models(ai=ai)
            if get_model(name=name).ocr_engine
        ]
        ocr_engines = {
            engine_id: build_ocr_engine(engine_id=engine_id)
            for engine_id in engine_ids
        }
    lower_priority()
    if power_pause_reason() is not None and not on_battery:
        out(
            "On battery power: reading waits for the charger"
            " (--on-battery reads anyway)."
        )
    report = read_unlabelled_receipts(
        finance=finance,
        ai=ai,
        ocr_engines=ocr_engines,
        pause_reason=(lambda: None) if on_battery else None,
        on_progress=lambda r: (
            out(f"read {r.photos_done}/{r.photos_total} photos")
            if r.photos_done % 10 == 0
            else None
        ),
    )
    out(
        f"{report.receipts_ready} of {report.receipts} receipts without a"
        " label are ready for suggestions"
        + (
            f" ({report.photos_missing} photo(s) not cropped yet)"
            if report.photos_missing
            else ""
        )
    )


@typechecked
def _install_service(
    *, config_path: str, service_dir: str, out: Callable[[str], None]
) -> None:
    """BG-1: write the systemd user units; enabling them is left to the
    user."""
    folder = os.path.expanduser(service_dir)
    os.makedirs(folder, exist_ok=True)
    for name, content in systemd_units(
        python=sys.executable, config_path=config_path
    ).items():
        with open(os.path.join(folder, name), "w", encoding="utf-8") as handle:
            handle.write(content)
        out(f"wrote {os.path.join(folder, name)}")
    out(
        "Enable with: systemctl --user daemon-reload &&"
        f" systemctl --user enable --now {SERVICE_NAME}.timer"
    )


@typechecked
def run(
    *,
    args: argparse.Namespace,
    finance: FinancePaths,
    ai: AiPaths,
    out: Callable[[str], None],
    stream: TextIO = sys.stdout,
    ollama_url: str = "http://localhost:11434",
    ocr_engines: Optional[Dict[str, OcrEngine]] = None,
    device: Optional[Device] = None,
    transactions: Optional[Callable[[], TransactionData]] = None,
) -> int:
    """Run the parsed command; returns the exit code. *ocr_engines* and
    *device* replace the real OCR engines and the probed machine (tests);
    *transactions* loads the dataset of the transaction model."""
    if args.install_qvac:
        from hledger_ai.runtimes.qvac_client import install_qvac

        target = device or probe_device(
            storage_dir=ai.qvac_dir, ollama_url=ollama_url, qvac_dir=ai.qvac_dir
        )
        if not install_qvac(
            qvac_dir=ai.qvac_dir,
            disk_free_gb=target.disk_free_gb,
            needed_gb=get_model(name="qvac-ocr-linear").disk_gb,
            out=out,
        ):
            return 1
    if args.select is not None:
        try:
            chosen, strategies = resolve_names(names=args.select)
        except UnknownName as error:
            out(str(error))
            return 2
        if strategies:
            out(f"--select takes model names, not {sorted(strategies)}.")
            return 2
        write_selection(ai=ai, models=chosen)
        out(
            "Models in use: "
            + (", ".join(chosen) if chosen else "chosen by the device")
        )
    if args.models:
        _show_models(
            ai=ai,
            device=device
            or probe_device(
                storage_dir=ai.root, ollama_url=ollama_url, qvac_dir=ai.qvac_dir
            ),
            out=out,
            colour=stream.isatty(),
        )
    if args.train is not None:
        try:
            states = train(
                finance=finance,
                ai=ai,
                names=args.train,
                public_only=args.public_only,
                force=args.force,
                folds=args.folds,
                on_progress=lambda state: out(format_progress(state=state)),
                transactions=transactions,
            )
        except (UnknownName, NotEnoughData) as error:
            out(str(error))
            return 2
        for state in states:
            out(_summary(state=state))
    if args.infer:
        _infer(
            finance=finance,
            ai=ai,
            out=out,
            ocr_engines=ocr_engines,
            on_battery=args.on_battery,
        )
    if args.install_service:
        _install_service(
            config_path=args.config, service_dir=args.service_dir, out=out
        )
    if args.status:
        _show_status(ai=ai, once=args.once, stream=stream)
    return 0


@typechecked
def main(*, argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from hledger_config.config.load_config import load_config

    config = load_config(
        config_path=args.config, pre_processed_output_dir=None
    )
    from hledger_ai.transactions.dataset import load_transaction_data

    return run(
        args=args,
        finance=FinancePaths.from_config(config=config),
        ai=AiPaths.from_config(config=config),
        out=print,
        ollama_url=(
            config.ai.ollama_url if config.ai else "http://localhost:11434"
        ),
        transactions=lambda: load_transaction_data(config=config),
    )


if __name__ == "__main__":
    sys.exit(main())
