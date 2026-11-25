"""Dataset configurations: hash identity and chronological registry (DS-*)."""

from pathlib import Path

from helpers import make_label

from hledger_ai.dataset.configuration import DatasetRegistry, dataset_hash


def test_hash_ignores_the_order_labels_are_read_in() -> None:
    a, b = make_label(amount=1.0), make_label(amount=2.0)

    assert dataset_hash(labels=[a, b], private=True) == dataset_hash(
        labels=[b, a], private=True
    )


def test_editing_one_label_value_gives_a_new_hash() -> None:
    before = dataset_hash(labels=[make_label(amount=11.54)], private=True)
    after = dataset_hash(labels=[make_label(amount=11.55)], private=True)

    assert before != after


def test_adding_a_label_gives_a_new_hash() -> None:
    one = dataset_hash(labels=[make_label()], private=True)
    two = dataset_hash(
        labels=[make_label(), make_label(amount=3.0)], private=True
    )

    assert one != two


def test_label_schema_version_is_part_of_the_hash() -> None:
    labels = [make_label()]

    assert dataset_hash(
        labels=labels, private=True, schema_version="1"
    ) != dataset_hash(labels=labels, private=True, schema_version="2")


def test_embedded_config_snapshot_is_not_label_data() -> None:
    plain = make_label()
    with_config = dict(make_label(), config={"root": "/somewhere/else"})

    assert dataset_hash(labels=[plain], private=True) == dataset_hash(
        labels=[with_config], private=True
    )


def test_registry_keeps_first_registration_order(tmp_path: Path) -> None:
    registry = DatasetRegistry(path=str(tmp_path / "registry.json"))
    registry.register(
        dataset_hash="aaa", private=True, receipt_count=3, now="2026-01-01"
    )
    registry.register(
        dataset_hash="bbb", private=False, receipt_count=1, now="2026-01-02"
    )
    # Seen again later: must keep its original place and timestamp.
    registry.register(
        dataset_hash="aaa", private=True, receipt_count=3, now="2026-01-03"
    )

    found = DatasetRegistry(
        path=str(tmp_path / "registry.json")
    ).configurations()

    assert [(c.dataset_hash, c.registered_at) for c in found] == [
        ("aaa", "2026-01-01"),
        ("bbb", "2026-01-02"),
    ]
    assert [c.private for c in found] == [True, False]
    assert [c.receipt_count for c in found] == [3, 1]


def test_empty_registry_has_no_configurations(tmp_path: Path) -> None:
    registry = DatasetRegistry(path=str(tmp_path / "missing.json"))

    assert registry.configurations() == []
