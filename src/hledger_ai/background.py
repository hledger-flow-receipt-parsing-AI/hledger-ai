"""Background reading of the receipts that still need a label (SP-1).

Reading a photo (OCR) is the slow step; once it is in the cache the models
answer in milliseconds while labelling. Finished photos are never read
again, so the job can be stopped or the machine rebooted at any time (BG-1).
It waits while the laptop runs on battery and runs at the lowest priority
(BG-2).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

from typeguard import typechecked

from hledger_ai.dataset.examples import cropped_path, unlabelled_image_groups
from hledger_ai.ocr.cache import OcrCache, file_sha256
from hledger_ai.ocr.types import OcrEngine
from hledger_ai.paths import AiPaths, FinancePaths
from hledger_ai.privacy import make_private

log = logging.getLogger(__name__)

PAUSE_SECONDS = 60.0
SERVICE_NAME = "hledger-ai-background"


@typechecked
@dataclass(frozen=True)
class BackgroundReport:
    receipts: int  # receipts that still need a label
    receipts_ready: int  # ... of which every photo has been read
    photos_total: int  # photo readings needed (photos x OCR engines)
    photos_done: int
    photos_missing: int  # photos without a cropped copy yet


@typechecked
def power_pause_reason() -> Optional[str]:
    """BG-2: a reason to wait, or None. Waits while on battery power."""
    try:
        import psutil

        battery = psutil.sensors_battery()
    except Exception:  # no psutil, no battery information
        return None
    if battery is not None and battery.power_plugged is False:
        return "on battery"
    return None


@typechecked
def lower_priority() -> None:
    """BG-2: let everything else on the machine go first."""
    try:
        os.nice(19)
    except OSError:
        pass


@typechecked
def read_unlabelled_receipts(
    *,
    finance: FinancePaths,
    ai: AiPaths,
    ocr_engines: Dict[str, OcrEngine],
    pause_reason: Optional[Callable[[], Optional[str]]] = None,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Optional[Callable[[BackgroundReport], None]] = None,
) -> BackgroundReport:
    """Read every photo of every unlabelled receipt with every engine in
    *ocr_engines* (engine id -> engine), skipping what is cached.

    *pause_reason* says why to wait (default: while on battery power).
    """
    if pause_reason is None:
        pause_reason = power_pause_reason
    make_private(folder=ai.root)
    cache = OcrCache(cache_dir=ai.ocr_cache_dir)
    groups = unlabelled_image_groups(
        input_dir=finance.input_dir,
        labels_dir=finance.labels_dir,
        grouping_path=finance.grouping_path,
        label_filename=finance.label_filename,
    )
    missing = 0
    crops_per_group: List[List[str]] = []
    for group in groups:
        crops = []
        for raw in group:
            crop = cropped_path(
                raw_path=raw,
                processed_dir=finance.processed_dir,
                crop_suffix=finance.crop_suffix,
                crop_ext=finance.crop_ext,
            )
            if os.path.isfile(crop):
                crops.append(crop)
            else:
                missing += 1
        crops_per_group.append(crops)

    work = []  # (crop, sha, engine id)
    for crops in crops_per_group:
        for crop in crops:
            sha = file_sha256(path=crop)
            work += [(crop, sha, engine_id) for engine_id in ocr_engines]
    done = {
        (sha, engine_id)
        for _, sha, engine_id in work
        if cache.get(engine=engine_id, image_sha256=sha) is not None
    }

    def report() -> BackgroundReport:
        ready = sum(
            1
            for group, crops in zip(groups, crops_per_group)
            if len(crops) == len(group)
            and all(
                (file_sha256(path=crop), engine_id) in done
                for crop in crops
                for engine_id in ocr_engines
            )
        )
        current = BackgroundReport(
            receipts=len(groups),
            receipts_ready=ready,
            photos_total=len(work),
            photos_done=len(done),
            photos_missing=missing,
        )
        _save(path=ai.background_state_path, report=current)
        if on_progress is not None:
            on_progress(current)
        return current

    for crop, sha, engine_id in work:
        if (sha, engine_id) in done:
            continue
        while (reason := pause_reason()) is not None:
            log.info("Background reading paused: %s", reason)
            sleep(PAUSE_SECONDS)
        cache.put(
            result=ocr_engines[engine_id].read(
                image_path=crop, image_sha256=sha
            )
        )
        done.add((sha, engine_id))
        report()
    return report()


@typechecked
def _save(*, path: str, report: BackgroundReport) -> None:
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    data = asdict(report)
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1)
    os.replace(temporary, path)


@typechecked
def systemd_units(*, python: str, config_path: str) -> Dict[str, str]:
    """BG-1: user units that continue training and reading after a reboot
    and every few hours. File name -> content."""
    config_path = os.path.abspath(config_path)
    service = f"""[Unit]
Description=hledger-ai: train what is out of date, read unlabelled receipts

[Service]
Type=oneshot
WorkingDirectory={os.path.dirname(config_path)}
ExecStart={python} -m hledger_ai --config {config_path} --train --infer
Nice=19
IOSchedulingClass=idle
"""
    timer = f"""[Unit]
Description=hledger-ai background work after boot and every six hours

[Timer]
OnBootSec=5min
OnUnitActiveSec=6h
Unit={SERVICE_NAME}.service

[Install]
WantedBy=timers.target
"""
    return {f"{SERVICE_NAME}.service": service, f"{SERVICE_NAME}.timer": timer}
