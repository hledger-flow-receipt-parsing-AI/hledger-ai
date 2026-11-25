"""Where things live: the finance root and the private AI folder."""

from pathlib import Path

from hledger_config.config.AIConfig import AIConfig
from hledger_config.config.CategorisationConfig import CategorisationConfig
from hledger_config.config.Config import Config
from hledger_config.config.DirPathsConfig import DirPathsConfig
from hledger_config.config.Filenames import FileNames
from hledger_config.config.MatchingAlgoConfig import MatchingAlgoConfig
from hledger_config.config.ReceiptImgConfig import ReceiptImgConfig

from hledger_ai.paths import AiPaths, FinancePaths


def _config(tmp_path: Path, ai: AIConfig | None = None) -> Config:
    """A real Config, as the application builds it from config.yaml."""
    root = tmp_path / "finance"
    root.mkdir(parents=True, exist_ok=True)
    (root / "categories.yaml").write_text("food: {}\n", encoding="utf-8")
    return Config(
        accounts=[],
        dir_paths=DirPathsConfig(
            root_finance_path=str(root),
            working_subdir="work",
            receipt_images_input_dir="photos_in",
            hledger_plot_dir="plots",
            receipt_images_processed_dir="photos_done",
            receipt_labels_dir="labels",
            asset_transaction_csvs_dir="asset_transaction_csvs",
        ),
        file_names=FileNames(
            start_journal_filepath="start.journal",
            root_journal_filename="all.journal",
            tui_label_filename="my_label",
            receipt_img=ReceiptImgConfig(
                processing_metadata_ext=".json",
                rotate="_rotated",
                rotate_ext=".jpg",
                crop="_cut",
                crop_ext=".png",
            ),
            categories_filename="categories.yaml",
        ),
        categorisation=CategorisationConfig(quick=False),
        matching_algo=MatchingAlgoConfig(
            days=2,
            amount_range=0.0,
            days_month_swap=False,
            multiple_receipts_per_transaction=False,
        ),
        csv_encoding="utf-8",
        ai=ai,
    )


def test_finance_paths_follow_the_configured_folders_and_file_names(
    tmp_path: Path,
) -> None:
    root = tmp_path / "finance"

    paths = FinancePaths.from_config(config=_config(tmp_path))

    assert paths.input_dir == str(root / "photos_in")
    assert paths.processed_dir == str(root / "photos_done")
    assert paths.labels_dir == str(root / "labels")
    assert paths.training_dataset_dir == str(root / "receipt_training_dataset")
    assert paths.grouping_path == str(
        root / "photos_done" / "image_grouping.json"
    )
    assert paths.label_filename == "my_label.json"
    assert (paths.crop_suffix, paths.crop_ext) == ("_cut", ".png")


def test_ai_folder_is_private_to_its_finance_root(tmp_path: Path) -> None:
    # Models are trained on one finance root's receipts: another root (a
    # demo, a test fixture) must never load them.
    here = AiPaths.from_config(config=_config(tmp_path / "one"))
    there = AiPaths.from_config(config=_config(tmp_path / "two"))
    again = AiPaths.from_config(config=_config(tmp_path / "one"))

    assert here.models_dir != there.models_dir
    assert here.models_dir == again.models_dir
    assert here.models_dir.startswith(str(Path.home() / ".hledger-ai"))
    assert Path(here.models_dir).parent.name.startswith("finance-")
    assert here.feedback_dir == str(Path(here.models_dir).parent / "feedback")
    assert here.ocr_cache_dir == str(
        Path(here.models_dir).parent / "cache" / "ocr"
    )


def test_untouched_ai_defaults_in_the_config_still_mean_per_root(
    tmp_path: Path,
) -> None:
    # An ``ai:`` section that leaves the folders at their defaults.
    with_section = AiPaths.from_config(
        config=_config(tmp_path / "one", ai=AIConfig())
    )
    without = AiPaths.from_config(config=_config(tmp_path / "one"))

    assert with_section.models_dir == without.models_dir


def test_ai_folder_follows_the_ai_section_of_the_config(tmp_path: Path) -> None:
    ai = AIConfig(
        models_dir=str(tmp_path / "ai" / "models"),
        feedback_dir=str(tmp_path / "elsewhere" / "feedback"),
    )

    paths = AiPaths.from_config(config=_config(tmp_path, ai=ai))

    assert paths.models_dir == str(tmp_path / "ai" / "models")
    assert paths.feedback_dir == str(tmp_path / "elsewhere" / "feedback")
    # Caches and the dataset registry sit next to the models folder.
    assert paths.ocr_cache_dir == str(tmp_path / "ai" / "cache" / "ocr")
    assert paths.registry_path == str(
        tmp_path / "ai" / "datasets" / "registry.json"
    )
