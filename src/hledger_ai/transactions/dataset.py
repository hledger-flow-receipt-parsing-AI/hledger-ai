"""The training data of the transaction model: every bank row of the
finance root that the rule store categorises (exactly one rule), with that
category as its label.  Always private (PR-*, DS-5)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

from typeguard import typechecked

from hledger_core.categorisation.rules import RowView, RuleStore, verdict
from hledger_core.csv_parsing.bank_csv import load_bank_transactions

SCHEMA_VERSION = "1"


@typechecked
@dataclass(frozen=True)
class TransactionExample:
    row: RowView
    label: str  # the category path (or own account) the rules give it


@typechecked
@dataclass(frozen=True)
class TransactionData:
    examples: Tuple[TransactionExample, ...]
    private: bool
    dataset_hash: str
    row_count: int  # every row, categorised or not


@typechecked
def load_rows(*, config: Any) -> List[RowView]:
    """Every bank row of every CSV account of the root (hledger-config)."""
    rows: List[RowView] = []
    for account_config in config.accounts:
        if not account_config.has_input_csv():
            continue
        path = account_config.get_abs_csv_filepath(
            dir_paths_config=config.dir_paths
        )
        if not os.path.isfile(path):
            continue
        for transaction in load_bank_transactions(
            account_config=account_config,
            abs_csv_filepath=path,
            csv_encoding=config.csv_encoding,
        ):
            rows.append(RowView.from_transaction(transaction))
    return rows


@typechecked
def labelled_examples(
    *, rows: Sequence[RowView], store: RuleStore
) -> List[TransactionExample]:
    """The rows exactly one rule categorises, with its result as label."""
    examples: List[TransactionExample] = []
    for row in rows:
        answer = verdict(store=store, row=row)
        if answer.is_categorised:
            examples.append(TransactionExample(row=row, label=answer.result))
    return examples


@typechecked
def transaction_dataset_hash(*, examples: Sequence[TransactionExample]) -> str:
    """DS-1 for transactions: over the rows and their labels."""
    digest = hashlib.sha256()
    digest.update(
        f"schema={SCHEMA_VERSION};private=True;kind=transactions\n".encode()
    )
    lines = sorted(
        "\t".join(
            (
                e.row.direction,
                e.row.payee,
                e.row.description,
                e.row.counter_account,
                e.row.own_account,
                e.row.amount_text,
                e.row.date,
                e.label,
            )
        )
        for e in examples
    )
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@typechecked
def transaction_data(
    *, rows: Sequence[RowView], store: RuleStore
) -> TransactionData:
    examples = labelled_examples(rows=rows, store=store)
    return TransactionData(
        examples=tuple(examples),
        private=True,
        dataset_hash=transaction_dataset_hash(examples=examples),
        row_count=len(rows),
    )


@typechecked
def load_transaction_data(*, config: Any) -> TransactionData:
    """The dataset of the finance root in *config*."""
    store = RuleStore.load(path=config.categorisation_rules_path())
    return transaction_data(rows=load_rows(config=config), store=store)
