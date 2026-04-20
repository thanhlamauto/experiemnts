from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


@dataclass
class _NullProgress:
    total: int | None = None
    desc: str | None = None

    def update(self, _: int = 1) -> None:
        return None

    def close(self) -> None:
        return None

    def set_description(self, desc: str, refresh: bool = False) -> None:
        self.desc = desc
        return None

    def set_postfix_str(self, _: str, refresh: bool = False) -> None:
        return None

    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    leave: bool = False,
) -> Iterable[T]:
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, desc=desc, total=total, leave=leave, dynamic_ncols=True)


def progress_bar(*, total: int | None = None, desc: str, leave: bool = False):
    if _tqdm is None:
        return _NullProgress(total=total, desc=desc)
    return _tqdm(total=total, desc=desc, leave=leave, dynamic_ncols=True)
