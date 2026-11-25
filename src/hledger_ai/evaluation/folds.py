"""Time-ordered evaluation folds (EV-2).

Receipts are sorted oldest first and cut into ``folds + 1`` chunks. Each
fold trains on all earlier chunks and is tested on the next one, so a model
never learns from receipts newer than the ones it is scored on. ``folds=1``
is a single split with the newest half held out.
"""

from __future__ import annotations

from typing import List, Tuple

from typeguard import typechecked


@typechecked
def forward_chaining_folds(
    *, count: int, folds: int
) -> List[Tuple[range, range]]:
    """(train indices, test indices) per fold; empty folds are dropped."""
    chunks = folds + 1
    bounds = [round(i * count / chunks) for i in range(chunks + 1)]
    result: List[Tuple[range, range]] = []
    for index in range(1, chunks):
        train = range(0, bounds[index])
        test = range(bounds[index], bounds[index + 1])
        if len(train) and len(test):
            result.append((train, test))
    return result
