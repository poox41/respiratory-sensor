from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Tuple


@dataclass
class Sample:
    t_ms: int
    x: float


@dataclass
class Rates:
    bpm: float | None = None
    rpm: float | None = None


class RingBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._t: Deque[int] = deque(maxlen=capacity)
        self._v: Deque[float] = deque(maxlen=capacity)

    def add(self, t_ms: int, v: float) -> None:
        self._t.append(t_ms)
        self._v.append(v)

    def clear(self) -> None:
        self._t.clear()
        self._v.clear()

    def is_empty(self) -> bool:
        return not self._v

    def snapshot(self) -> Tuple[List[int], List[float]]:
        return list(self._t), list(self._v)
