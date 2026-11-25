"""Dates and times found in the OCR text of a receipt.

Every date on the receipt (and the day the photo was taken) is a candidate
for the receipt date; a trained ranker picks among them.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Tuple

from typeguard import typechecked

from hledger_ai.ocr.types import OcrResult

_YEAR_MIN, _YEAR_MAX = 2000, 2039

_MONTH_WORDS: Dict[int, str] = {
    1: "jan januari january januar janvier janv enero ene gennaio gen janeiro",
    2: "feb februari february februar fevrier fevr fev febrero febbraio"
    " fevereiro",
    3: "mrt maart mar march marz maerz mars marzo marco",
    4: "apr april avril avr abril abr aprile",
    5: "mei may mai mayo maggio mag maio",
    6: "jun juni june juin junio giugno giu junho",
    7: "jul juli july juillet juil julio luglio lug julho",
    8: "aug augustus august aout agosto ago",
    9: "sep sept september septembre septiembre settembre set setembro",
    10: "okt oktober oct october octobre octubre ottobre ott outubro out",
    11: "nov november novembre noviembre novembro",
    12: "dec december dez dezember decembre diciembre dic dicembre dezembro",
}
_MONTHS: Dict[str, int] = {
    word: number
    for number, words in _MONTH_WORDS.items()
    for word in words.split()
}

# OCR often drops the space between a date and the time after it.
_GLUED_TIME = re.compile(r"((?:19|20)\d{2})(\d{1,2}\s?:\s?\d{2})")
_SEP = r"\s?[-/.]\s?"
_NUMERIC = re.compile(
    rf"(?<![\d.,])(\d{{1,2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{4}}|\d{{2}})(?!\d)"
)
_YEAR_FIRST = re.compile(
    rf"(?<!\d)((?:19|20)\d{{2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{1,2}})(?!\d)"
)
_WORD = r"([A-Za-zÀ-ÿ]{3,9})"
_DAY_MONTH_WORD = re.compile(
    rf"(?<!\d)(\d{{1,2}})\s?[-/. ]?\s?{_WORD}\.?\s?[-/., ']?\s?"
    rf"(\d{{4}}|\d{{2}})(?!\d)"
)
_MONTH_WORD_DAY = re.compile(
    rf"(?<![A-Za-z]){_WORD}\.?\s?(\d{{1,2}})(?:st|nd|rd|th)?\s?,?\s?"
    rf"(\d{{4}})(?!\d)"
)
# Not the tail of a longer number or time ("32:07" in "14:32:07"), but a
# label's colon right before it is fine ("Time:10:30").
_COLON_TIME = re.compile(
    r"(?<![\d,])(?<!\d[:.])(\d{1,2})\s?:\s?(\d{2})(?:\s?:\s?(\d{2}))?(?![\d:])"
    r"(\s?[AaPp]\.?[Mm]\b)?"
)
_DOT_TIME = re.compile(
    r"(?<![\d:.,])(\d{1,2})\.(\d{2})(?:\.(\d{2}))?(?![\d.,])"
    r"(\s?(?:uur|u|h)\b)?"
)
_FILENAME_TIME = re.compile(r"(?<!\d)((?:19|20)\d{6})[_-]?(\d{6})(?!\d)")


@typechecked
@dataclass(frozen=True)
class FoundDate:
    value: date
    style: str  # "dmy", "mdy", "ymd" or "text" (month written as a word)
    start: int
    end: int


@typechecked
@dataclass(frozen=True)
class FoundTime:
    value: str  # "HH:MM"
    has_seconds: bool
    start: int
    end: int


@typechecked
def _full_year(*, year: int) -> int:
    return year + 2000 if year < 100 else year


@typechecked
def _make_date(*, year: int, month: int, day: int) -> Optional[date]:
    year = _full_year(year=year)
    if not _YEAR_MIN <= year <= _YEAR_MAX:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


@typechecked
def _month_number(*, word: str) -> Optional[int]:
    plain = unicodedata.normalize("NFKD", word.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return _MONTHS.get(plain)


@typechecked
def _unglue(*, text: str) -> str:
    return _GLUED_TIME.sub(r"\1 \2", text)


@typechecked
def find_dates(*, text: str) -> List[FoundDate]:
    """Every valid date reading in *text*, in reading order."""
    text = _unglue(text=text)
    found: List[Tuple[int, int, FoundDate]] = []

    def add(
        value: Optional[date], style: str, match: re.Match, rank: int
    ) -> None:
        if value is not None:
            found.append(
                (
                    match.start(),
                    rank,
                    FoundDate(
                        value=value,
                        style=style,
                        start=match.start(),
                        end=match.end(),
                    ),
                )
            )

    for match in _NUMERIC.finditer(text):
        first, second, year = (int(g) for g in match.groups())
        day_first = _make_date(year=year, month=second, day=first)
        month_first = _make_date(year=year, month=first, day=second)
        add(day_first, "dmy", match, 0)
        if month_first is not None and month_first != day_first:
            add(month_first, "mdy", match, 1)
    for match in _YEAR_FIRST.finditer(text):
        year, month, day = (int(g) for g in match.groups())
        add(_make_date(year=year, month=month, day=day), "ymd", match, 0)
    for match in _DAY_MONTH_WORD.finditer(text):
        month = _month_number(word=match.group(2))
        if month is not None:
            add(
                _make_date(
                    year=int(match.group(3)),
                    month=month,
                    day=int(match.group(1)),
                ),
                "text",
                match,
                0,
            )
    for match in _MONTH_WORD_DAY.finditer(text):
        month = _month_number(word=match.group(1))
        if month is not None:
            add(
                _make_date(
                    year=int(match.group(3)),
                    month=month,
                    day=int(match.group(2)),
                ),
                "text",
                match,
                0,
            )
    return [item for _, _, item in sorted(found, key=lambda f: (f[0], f[1]))]


@typechecked
def _clock(*, hour: int, minute: int, second: Optional[int]) -> Optional[str]:
    if hour > 23 or minute > 59 or (second is not None and second > 59):
        return None
    return f"{hour:02d}:{minute:02d}"


@typechecked
def find_times(*, text: str) -> List[FoundTime]:
    """Clock times in *text*. ``14.32`` only counts with a cue (seconds,
    "uur", or a date in the same text), since it is usually an amount."""
    text = _unglue(text=text)
    dates = find_dates(text=text)
    masked = list(text)
    for found_date in dates:
        for index in range(found_date.start, found_date.end):
            masked[index] = " "
    searchable = "".join(masked)

    found: List[FoundTime] = []
    for match in _COLON_TIME.finditer(searchable):
        hour, minute = int(match.group(1)), int(match.group(2))
        second = int(match.group(3)) if match.group(3) else None
        suffix = (match.group(4) or "").strip().lower().replace(".", "")
        if suffix and not 1 <= hour <= 12:
            continue
        if suffix == "pm" and hour < 12:
            hour += 12
        if suffix == "am" and hour == 12:
            hour = 0
        value = _clock(hour=hour, minute=minute, second=second)
        if value is not None:
            found.append(
                FoundTime(
                    value=value,
                    has_seconds=second is not None,
                    start=match.start(),
                    end=match.end(),
                )
            )
    for match in _DOT_TIME.finditer(searchable):
        second = int(match.group(3)) if match.group(3) else None
        if second is None and not match.group(4) and not dates:
            continue
        value = _clock(
            hour=int(match.group(1)), minute=int(match.group(2)), second=second
        )
        if value is not None:
            found.append(
                FoundTime(
                    value=value,
                    has_seconds=second is not None,
                    start=match.start(),
                    end=match.end(),
                )
            )
    return sorted(found, key=lambda f: f.start)


@typechecked
def capture_time_from_filename(*, path: str) -> Optional[datetime]:
    """Phone cameras name photos ``20250317_143207.jpg``."""
    match = _FILENAME_TIME.search(os.path.basename(path))
    if match is None:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None


@typechecked
def capture_time(*, image_paths: Sequence[str]) -> Optional[datetime]:
    """When the receipt was photographed: file name first, then EXIF."""
    for path in image_paths:
        from_name = capture_time_from_filename(path=path)
        if from_name is not None:
            return from_name
    for path in image_paths:
        try:
            from PIL import Image

            with Image.open(path) as image:
                exif = image.getexif()
                raw = exif.get_ifd(0x8769).get(36867) or exif.get(306)
            if raw:
                return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
        except (OSError, ValueError, KeyError):
            continue
    return None


@typechecked
@dataclass(frozen=True)
class DateCandidate:
    value: date
    style: str
    source: str  # "ocr" (printed on the receipt) or "capture" (photo day)
    row: int
    row_count: int
    row_text: str
    previous_row_text: str
    has_time_in_row: bool
    days_before_capture: Optional[int]


@typechecked
@dataclass(frozen=True)
class TimeCandidate:
    value: str
    row: int
    row_count: int
    row_text: str
    has_date_in_row: bool
    has_seconds: bool


@typechecked
def _row_texts(*, results: Sequence[OcrResult]) -> List[str]:
    return [text for result in results for text in result.row_texts()]


@typechecked
def date_candidates(
    *, results: Sequence[OcrResult], captured_at: Optional[datetime]
) -> List[DateCandidate]:
    rows = _row_texts(results=results)
    candidates: List[DateCandidate] = []
    for index, row_text in enumerate(rows):
        has_time = bool(find_times(text=row_text))
        for found in find_dates(text=row_text):
            candidates.append(
                DateCandidate(
                    value=found.value,
                    style=found.style,
                    source="ocr",
                    row=index,
                    row_count=len(rows),
                    row_text=row_text,
                    previous_row_text=rows[index - 1] if index else "",
                    has_time_in_row=has_time,
                    days_before_capture=(
                        (captured_at.date() - found.value).days
                        if captured_at
                        else None
                    ),
                )
            )
    if captured_at is not None:
        candidates.append(
            DateCandidate(
                value=captured_at.date(),
                style="capture",
                source="capture",
                row=len(rows),
                row_count=len(rows),
                row_text="",
                previous_row_text="",
                has_time_in_row=False,
                days_before_capture=0,
            )
        )
    return candidates


@typechecked
def time_candidates(*, results: Sequence[OcrResult]) -> List[TimeCandidate]:
    rows = _row_texts(results=results)
    candidates: List[TimeCandidate] = []
    for index, row_text in enumerate(rows):
        has_date = bool(find_dates(text=row_text))
        for found in find_times(text=row_text):
            candidates.append(
                TimeCandidate(
                    value=found.value,
                    row=index,
                    row_count=len(rows),
                    row_text=row_text,
                    has_date_in_row=has_date,
                    has_seconds=found.has_seconds,
                )
            )
    return candidates
