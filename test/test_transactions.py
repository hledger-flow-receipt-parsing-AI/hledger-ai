"""The bank-transaction model: dataset from the rules, training with
time-ordered folds, suggestions, rule drafts, and the CLI.  Every payee,
description and category here is invented."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from hledger_ai.app import train
from hledger_ai.cli import build_parser, run
from hledger_ai.dataset.configuration import DatasetRegistry
from hledger_ai.models.catalogue import get_model
from hledger_ai.paths import AiPaths, FinancePaths
from hledger_ai.training.state import StateStore
from hledger_ai.transactions.dataset import (
    TransactionExample,
    transaction_data,
    transaction_dataset_hash,
)
from hledger_ai.transactions.model import (
    MostFrequent,
    NotEnoughData,
    TransactionModel,
    score,
)
from hledger_ai.transactions.suggest import (
    draft_rules,
    latest_engine,
    load_group_suggester,
    load_model,
)
from hledger_ai.transactions.train import (
    FIELD,
    STRATEGY_ID,
    train_transaction_model,
)

from hledger_core.categorisation.rules import (
    Condition,
    RowView,
    Rule,
    RuleStore,
)

SHOPS = {
    "Voorbeeld Supermarkt": "groceries:voorbeeld",
    "Bakker Jansen": "food:bakery",
    "Tankstation Oost": "travel:fuel",
}


def row(
    index: int,
    *,
    payee: str,
    description: str = "",
    amount: float = 10.0,
    direction: str = "debit",
) -> RowView:
    return RowView(
        row_id=f"alex:examplebank:checking#{index}",
        direction=direction,
        payee=payee,
        description=description or f"pas {index}",
        counter_account="",
        own_account="alex:examplebank:checking",
        amount=amount,
        date=f"2026-{index % 12 + 1:02d}-{index % 28 + 1:02d}-00-00-00",
    )


def rows_and_rules():
    rows, rules = [], []
    for shop_index, (payee, target) in enumerate(SHOPS.items()):
        rules.append(
            Rule(
                id=shop_index + 1,
                direction="debit",
                match=(Condition(field="payee", op="contains", value=payee),),
                target=target,
            )
        )
        for k in range(8):
            rows.append(
                row(
                    shop_index * 8 + k,
                    payee=f"{payee} {k}",
                    amount=5.0 + k + 20 * shop_index,
                )
            )
    rows.append(row(90, payee="Nieuwe Winkel", description="pas 90"))
    rows.append(row(91, payee="Nieuwe Winkel", description="pas 91"))
    rows.append(
        row(
            92,
            payee="Werkgever",
            description="salaris",
            amount=-1000.0,
            direction="credit",
        )
    )
    return rows, RuleStore(rules=rules)


class TestDataset:
    def test_labels_come_from_the_rules_and_only_one_hot_rows(self):
        rows, store = rows_and_rules()
        data = transaction_data(rows=rows, store=store)
        assert data.row_count == 27 and len(data.examples) == 24
        assert data.private
        assert {e.label for e in data.examples} == set(SHOPS.values())

    def test_hash_changes_with_labels(self):
        rows, store = rows_and_rules()
        examples = transaction_data(rows=rows, store=store).examples
        changed = list(examples)
        changed[0] = TransactionExample(
            row=examples[0].row, label="other:thing"
        )
        assert transaction_dataset_hash(
            examples=examples
        ) != transaction_dataset_hash(examples=changed)
        assert transaction_dataset_hash(
            examples=examples
        ) == transaction_dataset_hash(examples=list(reversed(examples)))


class TestModel:
    def test_learns_the_shops_and_ranks_with_confidence(self):
        rows, store = rows_and_rules()
        examples = transaction_data(rows=rows, store=store).examples
        model = TransactionModel().fit(examples=examples)
        ranked = model.suggest(row=row(99, payee="Bakker Jansen 9"))
        assert ranked[0][0] == "food:bakery"
        assert 0.0 < ranked[0][1] <= 1.0
        assert sum(p for _, p in ranked) == pytest.approx(1.0, abs=1e-6)
        group = model.suggest_group(
            rows=[
                row(99, payee="Tankstation Oost 9"),
                row(98, payee="Tankstation Oost 8"),
            ],
            limit=2,
        )
        assert group[0][0] == "travel:fuel" and len(group) == 2

    def test_one_category_only_is_always_that_category(self):
        examples = [
            TransactionExample(row=row(i, payee="X"), label="a:b")
            for i in range(3)
        ]
        model = TransactionModel().fit(examples=examples)
        assert model.suggest(row=row(9, payee="Y")) == [("a:b", 1.0)]

    def test_nothing_to_learn_from(self):
        with pytest.raises(NotEnoughData):
            TransactionModel().fit(examples=[])

    def test_round_trip_through_joblib(self, tmp_path: Path):
        rows, store = rows_and_rules()
        examples = transaction_data(rows=rows, store=store).examples
        TransactionModel().fit(examples=examples).save(
            path=str(tmp_path / "m.joblib")
        )
        loaded = TransactionModel.load(path=str(tmp_path / "m.joblib"))
        assert (
            loaded.suggest(row=row(99, payee="Voorbeeld Supermarkt 1"))[0][0]
            == "groceries:voorbeeld"
        )

    def test_score_counts_top_one_hits_and_the_baseline(self):
        rows, store = rows_and_rules()
        examples = transaction_data(rows=rows, store=store).examples
        model = TransactionModel().fit(examples=examples)
        baseline = MostFrequent().fit(examples=examples)
        result = score(model=model, baseline=baseline, test=examples[:5])
        assert result["total"] == 5 and result["correct"] == 5
        assert 0 <= result["baseline_correct"] <= 5


class TestTraining:
    def test_state_holds_fold_accuracy_and_the_engine_is_saved(
        self, tmp_path: Path
    ):
        rows, store = rows_and_rules()
        data = transaction_data(rows=rows, store=store)
        states = StateStore(models_dir=str(tmp_path / "models"))
        progress = []
        state = train_transaction_model(
            spec=get_model(name="csv-linear"),
            data=data,
            store=states,
            folds=3,
            on_progress=progress.append,
        )
        assert state.finished and state.fits_done == state.fits_total == 4
        strategy = state.strategies[STRATEGY_ID]
        assert strategy.fields() == [FIELD]
        assert strategy.accuracy() is not None and strategy.accuracy() > 0.5
        assert strategy.baseline_accuracy() is not None
        assert Path(
            states.engine_path(
                model="csv-linear", dataset_hash=data.dataset_hash
            )
        ).exists()
        assert len(progress) >= 4
        again = train_transaction_model(
            spec=get_model(name="csv-linear"), data=data, store=states, folds=3
        )
        assert (
            again.updated_at == state.updated_at
        )  # up to date: not retrained

    def test_too_little_data_is_an_error(self, tmp_path: Path):
        data = transaction_data(rows=[row(1, payee="X")], store=RuleStore())
        with pytest.raises(NotEnoughData):
            train_transaction_model(
                spec=get_model(name="csv-linear"),
                data=data,
                store=StateStore(models_dir=str(tmp_path)),
            )


def _ai(tmp_path: Path) -> AiPaths:
    root = tmp_path / "ai"
    return AiPaths(
        models_dir=str(root / "models"), feedback_dir=str(root / "feedback")
    )


def _finance(tmp_path: Path) -> FinancePaths:
    return FinancePaths(
        input_dir=str(tmp_path / "in"),
        processed_dir=str(tmp_path / "proc"),
        labels_dir=str(tmp_path / "labels"),
        training_dataset_dir=str(tmp_path / "public"),
        grouping_path=str(tmp_path / "proc" / "image_grouping.json"),
        label_filename="receipt_image_to_obj_label.json",
        crop_suffix="_cropped",
        crop_ext=".jpg",
    )


class TestAppAndCli:
    def test_train_by_name_registers_a_transactions_dataset(
        self, tmp_path: Path
    ):
        rows, store = rows_and_rules()
        ai = _ai(tmp_path)
        states = train(
            finance=_finance(tmp_path),
            ai=ai,
            names=["csv-linear"],
            folds=2,
            transactions=lambda: transaction_data(rows=rows, store=store),
        )
        assert [s.model for s in states] == ["csv-linear"]
        registry = DatasetRegistry(path=ai.registry_path).configurations()
        assert [c.kind for c in registry] == ["transactions"]
        assert registry[0].private and registry[0].receipt_count == 24
        found = latest_engine(ai=ai)
        assert found is not None and found[1].finished
        assert (
            load_model(ai=ai).suggest(row=row(99, payee="Bakker Jansen 9"))[0][
                0
            ]
            == "food:bakery"
        )

    def test_train_everything_without_a_loader_skips_the_transaction_model(
        self, tmp_path: Path
    ):
        # An older --train only passes the receipt folders: it must keep
        # working now that csv-linear is one of the default models.  (The
        # selection is written so that no device is probed or benchmarked.)
        from hledger_ai.selection import write_selection

        ai = _ai(tmp_path)
        write_selection(ai=ai, models=["priors", "csv-linear"])
        states = train(finance=_finance(tmp_path), ai=ai, names=[])
        assert "csv-linear" not in [s.model for s in states]
        assert "priors" in [s.model for s in states]

    def test_train_by_name_without_a_loader_explains(self, tmp_path: Path):
        from hledger_ai.app import UnknownName

        with pytest.raises(UnknownName, match="finance config"):
            train(
                finance=_finance(tmp_path),
                ai=_ai(tmp_path),
                names=["csv-linear"],
            )

    def test_cli_train_prints_accuracy_on_unseen_transactions(
        self, tmp_path: Path
    ):
        rows, store = rows_and_rules()
        lines = []
        code = run(
            args=build_parser().parse_args(
                ["--config", "c.yaml", "--train", "csv-linear", "--folds", "2"]
            ),
            finance=_finance(tmp_path),
            ai=_ai(tmp_path),
            out=lines.append,
            transactions=lambda: transaction_data(rows=rows, store=store),
        )
        assert code == 0
        assert any(
            "accuracy on unseen transactions" in line and "ST-21" in line
            for line in lines
        )

    def test_cli_train_with_too_few_rows_says_so(self, tmp_path: Path):
        lines = []
        code = run(
            args=build_parser().parse_args(
                ["--config", "c.yaml", "--train", "csv-linear"]
            ),
            finance=_finance(tmp_path),
            ai=_ai(tmp_path),
            out=lines.append,
            transactions=lambda: transaction_data(
                rows=[row(1, payee="X")], store=RuleStore()
            ),
        )
        assert code == 2
        assert any("not enough" in line for line in lines)

    def test_suggester_for_a_config_and_for_another_root(self, tmp_path: Path):
        rows, store = rows_and_rules()
        other_root = tmp_path / "other_root"
        own = AiPaths.for_root(finance_root=str(other_root))
        train(
            finance=_finance(tmp_path),
            ai=own,
            names=["csv-linear"],
            folds=2,
            transactions=lambda: transaction_data(rows=rows, store=store),
        )
        config = SimpleNamespace(
            dir_paths=SimpleNamespace(
                root_finance_path=str(tmp_path / "this_root")
            ),
            ai=SimpleNamespace(
                models_dir="~/.hledger-ai/models",
                feedback_dir="~/.hledger-ai/feedback",
                transaction_model_root=str(other_root),
            ),
        )
        suggest = load_group_suggester(config=config)
        assert suggest is not None
        from hledger_core.categorisation.rules import group_rows

        group = group_rows([row(99, payee="Voorbeeld Supermarkt 9")])[0]
        assert suggest(group)[0][0] == "groceries:voorbeeld"
        config.ai.transaction_model_root = None
        assert (
            load_group_suggester(config=config) is None
        )  # this root has no model


class TestDrafts:
    def test_one_draft_per_group_with_category_and_probe(self):
        rows, store = rows_and_rules()
        examples = transaction_data(rows=rows, store=store).examples
        model = TransactionModel().fit(examples=examples)
        uncategorised = [
            r for r in rows if r.payee in ("Nieuwe Winkel", "Werkgever")
        ]
        drafts = draft_rules(
            model=model, store=store, rows=rows, uncategorised=uncategorised
        )
        assert [d.group.n for d in drafts] == [2, 1]
        first = drafts[0]
        assert first.draft.match[0] == Condition(
            field="payee", op="contains", value="Nieuwe Winkel"
        )
        assert (
            first.draft.target in SHOPS.values() and 0 < first.confidence <= 1
        )
        assert first.is_one_hot and len(first.probe.gained) == 2

    def test_without_a_model_the_drafts_have_no_category(self):
        rows, store = rows_and_rules()
        drafts = draft_rules(
            model=None, store=store, rows=rows, uncategorised=rows[-3:]
        )
        assert all(
            d.draft.target == "" and d.suggestions == [] for d in drafts
        )


class TestResume:
    def test_an_interrupted_run_keeps_its_scored_folds(self, tmp_path: Path):
        rows, store = rows_and_rules()
        data = transaction_data(rows=rows, store=store)
        states = StateStore(models_dir=str(tmp_path / "models"))
        spec = get_model(name="csv-linear")
        finished = train_transaction_model(spec=spec, data=data, store=states, folds=3)
        # Pretend the final fit never happened.
        finished.finished = False
        finished.fits_done = 3
        states.save(state=finished)
        folds_before = dict(finished.strategies[STRATEGY_ID].folds)
        resumed = train_transaction_model(spec=spec, data=data, store=states, folds=3)
        assert resumed.finished
        assert resumed.strategies[STRATEGY_ID].folds == folds_before
        assert resumed.fits_done == resumed.fits_total == 4


def test_status_table_of_a_transactions_dataset_shows_st21_and_rows():
    from hledger_ai.dataset.configuration import DatasetConfiguration
    from hledger_ai.progress.view import build_table, render_table
    from hledger_ai.strategies import STRATEGIES
    from hledger_ai.training.state import ModelState, StrategyState

    configuration = DatasetConfiguration(
        dataset_hash="a" * 64, private=True, receipt_count=24, registered_at="2026-11-26T10:00:00", kind="transactions"
    )
    state = ModelState(model="csv-linear", dataset_hash="a" * 64, private=True, model_version="1", hyperparameters={}, finished=True)
    state.strategies[STRATEGY_ID] = StrategyState(folds={"0": {FIELD: {"correct": 7, "total": 10, "baseline_correct": 2}}})
    table = build_table(configuration=configuration, index=0, count=1, states={"csv-linear": state}, strategy_ids=[s.id for s in STRATEGIES])
    assert table.strategy_ids == (STRATEGY_ID,)
    text = render_table(table=table, colour=False, width=120)
    assert "24 rows" in text and "unseen rows" in text and "ST-21" in text and "70%" in text
