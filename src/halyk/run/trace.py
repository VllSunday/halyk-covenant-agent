from __future__ import annotations

from pathlib import Path
from types import TracebackType

from halyk.models.lineage import LineageRecord


class LineageWriter:
    """Дописывает записи в lineage.jsonl.

    Файл открывается на дозапись, а не собирается в памяти: если прогон упадёт на
    середине, разобранная часть останется на диске.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle = path.open("a", encoding="utf-8", newline="\n")

    def write(self, record: LineageRecord) -> None:
        self._handle.write(record.model_dump_json() + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> LineageWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_lineage(path: Path) -> list[LineageRecord]:
    with path.open(encoding="utf-8") as handle:
        return [LineageRecord.model_validate_json(line) for line in handle if line.strip()]
