from typing import Callable


class Lazy:
    def __init__(self, func: Callable) -> None:
        self._func = func

    def __str__(self) -> str:
        return str(self._func())


class LazyCounter:
    _start: int

    def __init__(self, start: int = 0) -> None:
        self._start = start - 1

    def __str__(self) -> str:
        self._start += 1
        return str(self._start)
