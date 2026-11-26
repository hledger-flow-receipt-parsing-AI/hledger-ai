"""Where things live: the finance root (read-only for this package) and the
private AI folder (models, caches, logs; never leaves the device, PR-1)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

from typeguard import typechecked

DEFAULT_AI_ROOT = "~/.hledger-ai"
# The folders hledger-config's AIConfig holds when the user set nothing.
_UNSET_MODELS_DIR = "~/.hledger-ai/models"
_UNSET_FEEDBACK_DIR = "~/.hledger-ai/feedback"
GROUPING_FILENAME = "image_grouping.json"
TRAINING_DATASET_DIRNAME = "receipt_training_dataset"


@typechecked
@dataclass(frozen=True)
class FinancePaths:
    input_dir: str
    processed_dir: str
    labels_dir: str
    training_dataset_dir: str  # the public dataset ("not mine" receipts)
    grouping_path: str
    label_filename: str
    crop_suffix: str
    crop_ext: str

    @staticmethod
    def from_config(*, config: Any) -> "FinancePaths":
        """From a loaded ``hledger_config`` Config."""
        dirs, names = config.dir_paths, config.file_names
        processed = dirs.get_path("receipt_images_processed_dir", absolute=True)
        return FinancePaths(
            input_dir=dirs.get_path("receipt_images_input_dir", absolute=True),
            processed_dir=processed,
            labels_dir=dirs.get_path("receipt_labels_dir", absolute=True),
            training_dataset_dir=os.path.join(
                os.path.abspath(dirs.root_finance_path),
                TRAINING_DATASET_DIRNAME,
            ),
            grouping_path=os.path.join(processed, GROUPING_FILENAME),
            label_filename=f"{names.tui_label_filename}.json",
            crop_suffix=names.receipt_img.crop,
            crop_ext=names.receipt_img.crop_ext,
        )


@typechecked
@dataclass(frozen=True)
class AiPaths:
    models_dir: str
    feedback_dir: str

    @staticmethod
    def from_config(*, config: Any) -> "AiPaths":
        """The AI folder of this finance root.

        Models are trained on the receipts of one finance root, so by
        default each root gets its own folder under ``~/.hledger-ai``: a
        demo or test root never loads models trained on real receipts.
        Folders set explicitly in the config's ``ai:`` section win.
        """
        own = AiPaths.own_folder(
            finance_root=config.dir_paths.root_finance_path
        )
        ai = getattr(config, "ai", None)
        models = ai.models_dir if ai else _UNSET_MODELS_DIR
        feedback = ai.feedback_dir if ai else _UNSET_FEEDBACK_DIR
        return AiPaths(
            models_dir=(
                os.path.join(own, "models")
                if models == _UNSET_MODELS_DIR
                else os.path.expanduser(models)
            ),
            feedback_dir=(
                os.path.join(own, "feedback")
                if feedback == _UNSET_FEEDBACK_DIR
                else os.path.expanduser(feedback)
            ),
        )

    @staticmethod
    def own_folder(*, finance_root: str) -> str:
        """``~/.hledger-ai/<basename>-<sha8>`` of a finance root."""
        finance_root = os.path.abspath(os.path.expanduser(finance_root))
        digest = hashlib.sha256(finance_root.encode("utf-8")).hexdigest()[:8]
        return os.path.join(
            os.path.expanduser(DEFAULT_AI_ROOT),
            f"{os.path.basename(finance_root)}-{digest}",
        )

    @staticmethod
    def for_root(*, finance_root: str) -> "AiPaths":
        """The default AI folder of another finance root on this machine
        (``ai.transaction_model_root``)."""
        own = AiPaths.own_folder(finance_root=finance_root)
        return AiPaths(
            models_dir=os.path.join(own, "models"),
            feedback_dir=os.path.join(own, "feedback"),
        )

    @property
    def root(self) -> str:
        return os.path.dirname(self.models_dir)

    @property
    def ocr_cache_dir(self) -> str:
        return os.path.join(self.root, "cache", "ocr")

    @property
    def registry_path(self) -> str:
        return os.path.join(self.root, "datasets", "registry.json")

    @property
    def corrections_path(self) -> str:
        return os.path.join(self.feedback_dir, "corrections.jsonl")

    @property
    def timings_path(self) -> str:
        return os.path.join(self.feedback_dir, "field_timings.jsonl")

    @property
    def qvac_dir(self) -> str:
        """The qvac SDK holds no private data: all finance roots share it."""
        return os.path.join(
            os.path.expanduser(DEFAULT_AI_ROOT), "runtimes", "qvac"
        )

    @property
    def background_state_path(self) -> str:
        return os.path.join(self.root, "background", "state.json")
