from __future__ import annotations

import math
from typing import Tuple

from models import Rates, RingBuffer, Sample


class MovingAverage:
    def __init__(self, window_size: int) -> None:
        self.window_size = max(1, window_size)
        self.buf = [0.0] * self.window_size
        self.sum = 0.0
        self.idx = 0
        self.filled = 0

    def next(self, x: float) -> float:
        self.sum -= self.buf[self.idx]
        self.buf[self.idx] = x
        self.sum += x
        self.idx = (self.idx + 1) % self.window_size
        if self.filled < self.window_size:
            self.filled += 1
        return self.sum / self.filled


class ShortSmoother:
    def __init__(self, window_size: int) -> None:
        self.ma = MovingAverage(max(1, window_size))

    def next(self, x: float) -> float:
        return self.ma.next(x)


class PeakRateEstimator:
    def __init__(
        self,
        fs_hz: int,
        window_sec: int,
        min_bpm: float = 40.0,
        max_bpm: float = 180.0,
    ) -> None:
        self.fs_hz = fs_hz
        self.window_sec = window_sec
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.values = RingBuffer(fs_hz * window_sec)

    def add(self, t_ms: int, v: float) -> None:
        self.values.add(t_ms, v)

    def estimate(self) -> float | None:
        _, vs = self.values.snapshot()
        n = len(vs)
        if n < self.fs_hz * 2:
            return None

        mean = sum(vs) / n

        min_hz = max(0.1, self.min_bpm / 60.0)
        max_hz = max(min_hz, self.max_bpm / 60.0)

        k_min = max(1, int(min_hz * n / self.fs_hz))
        k_max = min(n // 2, int(max_hz * n / self.fs_hz))
        if k_max <= k_min:
            return None

        best_k = -1
        best_mag = 0.0
        two_pi = 2.0 * math.pi

        for k in range(k_min, k_max + 1):
            w = two_pi * k / n
            re = 0.0
            im = 0.0
            for i, x in enumerate(vs):
                x0 = x - mean
                ang = w * i
                re += x0 * math.cos(ang)
                im -= x0 * math.sin(ang)
            mag = re * re + im * im
            if mag > best_mag:
                best_mag = mag
                best_k = k

        if best_k < 0:
            return None
        freq = best_k * self.fs_hz / n
        return freq * 60.0

    def clear(self) -> None:
        self.values.clear()


class Processor:
    def __init__(self, fs_hz: int) -> None:
        self.fs_hz = fs_hz
        self.cap = fs_hz * 30
        self.raw_buf = RingBuffer(self.cap)
        self.resp_buf = RingBuffer(self.cap)
        self.hr_buf = RingBuffer(self.cap)

        self.resp_lp = MovingAverage(window_size=fs_hz * 2)
        self.hr_smooth = ShortSmoother(window_size=max(1, fs_hz // 10))

        self.hr_estimator = PeakRateEstimator(
            fs_hz=fs_hz,
            window_sec=10,
            min_bpm=40.0,
            max_bpm=180.0,
        )
        self.resp_estimator = PeakRateEstimator(
            fs_hz=fs_hz,
            window_sec=30,
            min_bpm=6.0,
            max_bpm=30.0,
        )

        self.rates = Rates()
        self.last_estimate_ms = 0

    def on_sample(self, sample: Sample) -> None:
        t = sample.t_ms
        x = sample.x

        resp = self.resp_lp.next(x)
        hr_raw = x - resp
        hr = self.hr_smooth.next(hr_raw)

        self.raw_buf.add(t, x)
        self.resp_buf.add(t, resp)
        self.hr_buf.add(t, hr)

        self.hr_estimator.add(t, hr)
        self.resp_estimator.add(t, resp)

        if t - self.last_estimate_ms >= 1000:
            self.last_estimate_ms = t
            self.rates = Rates(
                bpm=self.hr_estimator.estimate(),
                rpm=self.resp_estimator.estimate(),
            )

    def reset(self) -> None:
        self.raw_buf.clear()
        self.resp_buf.clear()
        self.hr_buf.clear()
        self.hr_estimator.clear()
        self.resp_estimator.clear()
        self.last_estimate_ms = 0
        self.rates = Rates()
