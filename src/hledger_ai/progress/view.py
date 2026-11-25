"""The training progress table (requirements UI-1 to UI-6).

Models are rows, strategies are columns, one dataset configuration at a
time. Each cell reads ``23%,352h 45%,21m 87%`` (training, inference,
accuracy) on a background that goes from red to green with the accuracy.
Ctrl+Left and Ctrl+Right step through the dataset configurations in the
order they were registered; private ones are marked orange.
"""

from __future__ import annotations

import os
import select
import shutil
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, TextIO, Tuple

from typeguard import typechecked

from hledger_ai.dataset.configuration import DatasetConfiguration
from hledger_ai.progress.format import format_cell
from hledger_ai.training.state import ModelState

_NAME_WIDTH = 24
_CELL_WIDTH = 26
_ORANGE = "\x1b[38;5;208m"
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"

_KEYS = {
    b"\x1b[1;5D": "ctrl left",
    b"\x1b[5D": "ctrl left",
    b"\x1bOd": "ctrl left",
    b"\x1b[1;5C": "ctrl right",
    b"\x1b[5C": "ctrl right",
    b"\x1bOc": "ctrl right",
    b"\x1b[D": "left",
    b"\x1bOD": "left",
    b"\x1b[C": "right",
    b"\x1bOC": "right",
    b"q": "q",
    b"Q": "q",
    b"\x03": "q",
}


@typechecked
@dataclass(frozen=True)
class Cell:
    text: str
    accuracy: Optional[float]  # None: nothing scored, no background


@typechecked
@dataclass(frozen=True)
class Table:
    configuration: DatasetConfiguration
    index: int  # position in the timeline, from 0
    count: int
    strategy_ids: Tuple[str, ...]
    rows: Tuple[Tuple[str, Tuple[Cell, ...]], ...]


@typechecked
def build_table(
    *,
    configuration: DatasetConfiguration,
    index: int,
    count: int,
    states: Dict[str, Optional[ModelState]],
    strategy_ids: Sequence[str],
) -> Table:
    """UI-1, UI-2: one row per model in *states* (None: never trained on
    this configuration), one cell per strategy."""
    rows = []
    for model, state in states.items():
        cells = []
        for strategy_id in strategy_ids:
            progress = state.strategies.get(strategy_id) if state else None
            if state is None or progress is None:
                cells.append(Cell(text="-", accuracy=None))
                continue
            cells.append(
                Cell(
                    text=format_cell(
                        training_fraction=state.training_fraction(),
                        training_eta=state.training_eta_seconds(),
                        inference_fraction=progress.inference_fraction(),
                        inference_eta=progress.inference_eta_seconds(),
                        accuracy=progress.accuracy(),
                    ),
                    accuracy=progress.accuracy(),
                )
            )
        rows.append((model, tuple(cells)))
    return Table(
        configuration=configuration,
        index=index,
        count=count,
        strategy_ids=tuple(strategy_ids),
        rows=tuple(rows),
    )


@typechecked
def accuracy_colour(*, accuracy: float) -> Tuple[int, int, int]:
    """UI-3: red at 0%, yellow at 50%, green at 100% (RGB)."""
    low, middle, high = (230, 60, 50), (230, 200, 60), (60, 200, 80)
    start, end, share = (
        (low, middle, accuracy * 2)
        if accuracy < 0.5
        else (middle, high, (accuracy - 0.5) * 2)
    )
    share = min(max(share, 0.0), 1.0)
    return tuple(  # type: ignore[return-value]
        round(a + (b - a) * share) for a, b in zip(start, end)
    )


@typechecked
def _paint(*, cell: Cell, colour: bool) -> str:
    text = cell.text.ljust(_CELL_WIDTH - 2)
    if not colour or cell.accuracy is None:
        return text + "  "
    red, green, blue = accuracy_colour(accuracy=cell.accuracy)
    return f"\x1b[48;2;{red};{green};{blue}m\x1b[30m{text}{_RESET}  "


@typechecked
def render_table(
    *, table: Table, colour: bool, width: int, first_column: int = 0
) -> str:
    """The table as text; *first_column* scrolls the strategy columns when
    the terminal is too narrow for all of them."""
    configuration = table.configuration
    kind = "private" if configuration.private else "public"
    title = (
        f"dataset {table.index + 1}/{table.count}"
        f"  {configuration.dataset_hash[:12]}  {kind}"
        f"  {configuration.receipt_count} receipts"
        f"  registered {configuration.registered_at}"
    )
    if colour:
        # UI-5: datasets with private data stand out in orange.
        title = (
            f"{_ORANGE}{_BOLD}{title}{_RESET}"
            if configuration.private
            else f"{_BOLD}{title}{_RESET}"
        )
    total = len(table.strategy_ids)
    fits = max(1, (width - _NAME_WIDTH) // _CELL_WIDTH)
    first = min(max(first_column, 0), max(total - 1, 0))
    last = min(first + fits, total)
    lines = [
        title,
        "cell: training %,ETA  inference %,ETA  accuracy on unseen receipts",
        "",
        "model".ljust(_NAME_WIDTH)
        + "".join(s.ljust(_CELL_WIDTH) for s in table.strategy_ids[first:last]),
    ]
    for model, cells in table.rows:
        lines.append(
            model.ljust(_NAME_WIDTH)
            + "".join(_paint(cell=c, colour=colour) for c in cells[first:last])
        )
    lines.append("")
    if last - first < total:
        lines.append(
            f"columns {first + 1}-{last} of {total}"
            "  (Left/Right: other strategies)"
        )
    lines.append("Ctrl+Left/Right: other dataset configuration   q: quit")
    return "\n".join(lines)


@typechecked
def step_configuration(*, index: int, count: int, key: str) -> int:
    """UI-4: Ctrl+arrows move through the timeline and wrap at its ends."""
    if count <= 0:
        return 0
    if key == "ctrl right":
        return (index + 1) % count
    if key == "ctrl left":
        return (index - 1) % count
    return index


@typechecked
def decode_key(*, raw: bytes) -> Optional[str]:
    """Name of the key a terminal sent, or None for anything else."""
    return _KEYS.get(raw)


@typechecked
def run_view(
    *,
    load_configurations: Callable[[], List[DatasetConfiguration]],
    load_states: Callable[
        [DatasetConfiguration], Dict[str, Optional[ModelState]]
    ],
    strategy_ids: Sequence[str],
    once: bool,
    out: TextIO = sys.stdout,
) -> None:
    """Show the table; live with key handling on a terminal, else once."""
    interactive = (not once) and sys.stdin.isatty() and out.isatty()
    colour = out.isatty() and "NO_COLOR" not in os.environ

    def draw(index: int, first_column: int) -> int:
        configurations = load_configurations()
        if not configurations:
            out.write("Nothing has been trained yet. Run --train first.\n")
            return 0
        index = min(index, len(configurations) - 1)
        table = build_table(
            configuration=configurations[index],
            index=index,
            count=len(configurations),
            states=load_states(configurations[index]),
            strategy_ids=strategy_ids,
        )
        text = render_table(
            table=table,
            colour=colour,
            width=shutil.get_terminal_size((200, 40)).columns,
            first_column=first_column,
        )
        out.write(("\x1b[H\x1b[2J" if interactive else "") + text + "\n")
        out.flush()
        return len(configurations)

    newest = max(len(load_configurations()) - 1, 0)
    if not interactive:
        draw(newest, 0)
        return

    import termios
    import tty

    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)
    index, first_column = newest, 0
    try:
        tty.setcbreak(descriptor)
        while True:
            count = draw(index, first_column)
            ready, _, _ = select.select([descriptor], [], [], 1.0)
            if not ready:
                continue  # refresh once a second
            key = decode_key(raw=os.read(descriptor, 16))
            if key == "q":
                return
            if key in ("left", "right"):
                step = 1 if key == "right" else -1
                first_column = min(
                    max(first_column + step, 0), max(len(strategy_ids) - 1, 0)
                )
            elif key is not None:
                index = step_configuration(index=index, count=count, key=key)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)
