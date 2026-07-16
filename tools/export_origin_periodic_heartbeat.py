"""Select the most periodic late 10 s window and export an Origin-ready trace.

Selection is based only on continuous, non-enhanced channels.  The enhanced
waveform is exported after selection, so its deliberately regular template
cannot bias the choice of position or interval.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


DATASETS = (
    (
        "chest_152843",
        "胸口",
        Path(r"F:\Users\lik\Documents\xwechat_files\wxid_dkahkwnvrf9422_e9bf\msg\file\2026-07\sensor_20260716_152843.csv"),
    ),
    (
        "abdomen_153115",
        "腹部",
        Path(r"F:\Users\lik\Documents\xwechat_files\wxid_dkahkwnvrf9422_e9bf\msg\file\2026-07\sensor_20260716_153115.csv"),
    ),
    (
        "chairback_153454",
        "椅背",
        Path(r"F:\Users\lik\Documents\xwechat_files\wxid_dkahkwnvrf9422_e9bf\msg\file\2026-07\sensor_20260716_153454.csv"),
    ),
)

NUMERIC_COLUMNS = (
    "time_ms",
    "rawX",
    "xDc",
    "cleanHeart",
    "heartTemplateInput",
    "heartTemplateEnhanced",
    "heartTemplateQuality",
    "cleanPeak",
    "bpm",
)


@dataclass
class Dataset:
    dataset_id: str
    position: str
    path: Path
    values: Dict[str, np.ndarray]
    fs_hz: float


@dataclass
class WindowResult:
    dataset: Dataset
    start_index: int
    end_index: int
    start_s: float
    end_s: float
    clean_autocorrelation: float
    detail_autocorrelation: float
    rr_cv: float
    rr_stability: float
    template_quality: float
    median_bpm: float
    beat_count: int
    score: float


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_dataset(dataset_id: str, position: str, path: Path) -> Dataset:
    columns: Dict[str, List[float]] = {name: [] for name in NUMERIC_COLUMNS}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in NUMERIC_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        for row in reader:
            for name in NUMERIC_COLUMNS:
                columns[name].append(parse_float(row.get(name, "")))

    arrays = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    time_ms = arrays["time_ms"]
    finite_diff = np.diff(time_ms)
    finite_diff = finite_diff[np.isfinite(finite_diff) & (finite_diff > 0)]
    if len(time_ms) < 100 or finite_diff.size == 0:
        raise ValueError(f"{path.name}: insufficient samples")
    fs_hz = 1000.0 / float(np.median(finite_diff))
    return Dataset(dataset_id, position, path, arrays, fs_hz)


def robust_normalize(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    centered = values - float(np.median(finite))
    scale = float(np.percentile(np.abs(centered[np.isfinite(centered)]), 99))
    if not math.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return np.clip(centered / scale, -1.25, 1.25)


def normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    finite = np.isfinite(first) & np.isfinite(second)
    if int(np.sum(finite)) < 20:
        return 0.0
    left = first[finite] - float(np.mean(first[finite]))
    right = second[finite] - float(np.mean(second[finite]))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def autocorrelation_near_period(values: np.ndarray, fs_hz: float, period_s: float) -> float:
    if not math.isfinite(period_s):
        period_s = 0.8
    low_lag = max(1, int(round(fs_hz * max(0.45, period_s * 0.82))))
    high_lag = min(len(values) // 3, int(round(fs_hz * min(1.50, period_s * 1.18))))
    if high_lag <= low_lag:
        return 0.0
    best = -1.0
    for lag in range(low_lag, high_lag + 1):
        value = normalized_correlation(values[:-lag], values[lag:])
        if value > best:
            best = value
    return max(0.0, best)


def finite_median(values: np.ndarray, default: float = math.nan) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else default


def evaluate_window(dataset: Dataset, start: int, end: int) -> WindowResult | None:
    values = dataset.values
    time_ms = values["time_ms"][start:end]
    if len(time_ms) < int(dataset.fs_hz * 9.2):
        return None
    gaps = np.diff(time_ms)
    if gaps.size == 0 or float(np.max(gaps)) > 80.0:
        return None

    peak_mask = values["cleanPeak"][start:end] > 0.5
    peak_times = time_ms[peak_mask]
    rr_s = np.diff(peak_times) / 1000.0
    rr_s = rr_s[(rr_s >= 0.45) & (rr_s <= 1.50)]
    if rr_s.size < 7:
        return None
    rr_median = float(np.median(rr_s))
    rr_cv = float(np.std(rr_s) / max(float(np.mean(rr_s)), 1e-9))
    rr_stability = float(math.exp(-5.0 * rr_cv))

    clean = robust_normalize(values["cleanHeart"][start:end])
    detail = robust_normalize(values["heartTemplateInput"][start:end])
    clean_corr = autocorrelation_near_period(clean, dataset.fs_hz, rr_median)
    detail_corr = autocorrelation_near_period(detail, dataset.fs_hz, rr_median)
    template_quality = finite_median(values["heartTemplateQuality"][start:end], default=0.0)
    template_quality = min(1.0, max(0.0, template_quality))
    logged_bpm = finite_median(values["bpm"][start:end])
    median_bpm = logged_bpm if math.isfinite(logged_bpm) else 60.0 / rr_median

    score = (
        0.40 * clean_corr
        + 0.30 * detail_corr
        + 0.20 * rr_stability
        + 0.10 * template_quality
    )
    base_t = float(values["time_ms"][0])
    return WindowResult(
        dataset=dataset,
        start_index=start,
        end_index=end,
        start_s=(float(time_ms[0]) - base_t) / 1000.0,
        end_s=(float(time_ms[-1]) - base_t) / 1000.0,
        clean_autocorrelation=clean_corr,
        detail_autocorrelation=detail_corr,
        rr_cv=rr_cv,
        rr_stability=rr_stability,
        template_quality=template_quality,
        median_bpm=median_bpm,
        beat_count=int(len(peak_times)),
        score=score,
    )


def best_late_window(dataset: Dataset, window_s: float = 10.0) -> WindowResult:
    time_ms = dataset.values["time_ms"]
    relative_s = (time_ms - time_ms[0]) / 1000.0
    duration_s = float(relative_s[-1])
    search_from_s = max(0.0, duration_s * 0.40)
    start = bisect.bisect_left(relative_s.tolist(), search_from_s)
    step = max(1, int(round(dataset.fs_hz * 0.5)))
    candidates: List[WindowResult] = []
    for begin in range(start, len(time_ms), step):
        target_end_ms = float(time_ms[begin]) + window_s * 1000.0
        end = int(np.searchsorted(time_ms, target_end_ms, side="right"))
        if end > len(time_ms) or end - begin < dataset.fs_hz * (window_s - 0.8):
            continue
        result = evaluate_window(dataset, begin, end)
        if result is not None:
            candidates.append(result)
    if not candidates:
        raise ValueError(f"{dataset.path.name}: no valid late {window_s:.1f} s window")
    # Prefer the later interval when scores are effectively tied.
    return max(candidates, key=lambda item: (round(item.score, 4), item.start_s))


def write_metrics(path: Path, windows: Sequence[WindowResult], selected: WindowResult) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "dataset_id",
                "position",
                "source_file",
                "sample_rate_hz",
                "window_start_s",
                "window_end_s",
                "beat_count",
                "median_bpm",
                "cleanheart_autocorrelation",
                "detail_autocorrelation",
                "rr_cv",
                "rr_stability",
                "template_quality",
                "composite_score",
                "selected",
            )
        )
        for result in windows:
            writer.writerow(
                (
                    result.dataset.dataset_id,
                    result.dataset.position,
                    result.dataset.path.name,
                    f"{result.dataset.fs_hz:.3f}",
                    f"{result.start_s:.3f}",
                    f"{result.end_s:.3f}",
                    result.beat_count,
                    f"{result.median_bpm:.3f}",
                    f"{result.clean_autocorrelation:.6f}",
                    f"{result.detail_autocorrelation:.6f}",
                    f"{result.rr_cv:.6f}",
                    f"{result.rr_stability:.6f}",
                    f"{result.template_quality:.6f}",
                    f"{result.score:.6f}",
                    "1" if result is selected else "0",
                )
            )


def selected_arrays(result: WindowResult) -> Dict[str, np.ndarray]:
    start, end = result.start_index, result.end_index
    raw = {name: values[start:end].copy() for name, values in result.dataset.values.items()}
    raw["time_s"] = (raw["time_ms"] - raw["time_ms"][0]) / 1000.0
    raw["mixed_normalized"] = robust_normalize(raw["xDc"])
    raw["clean_normalized"] = robust_normalize(raw["cleanHeart"])
    raw["detail_normalized"] = robust_normalize(raw["heartTemplateInput"])
    raw["enhanced_normalized"] = robust_normalize(raw["heartTemplateEnhanced"])
    cycle_index = np.full(len(raw["time_s"]), -1, dtype=int)
    current = -1
    for index, is_peak in enumerate(raw["cleanPeak"] > 0.5):
        if is_peak:
            current += 1
        cycle_index[index] = current
    raw["cycle_index"] = cycle_index
    return raw


def circular_shift(values: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(values, shift).copy()


def align_cycle(cycle: np.ndarray, reference: np.ndarray, maximum_shift: int = 6) -> Tuple[np.ndarray, float]:
    best = cycle.copy()
    best_correlation = -1.0
    for shift in range(-maximum_shift, maximum_shift + 1):
        shifted = circular_shift(cycle, shift)
        value = normalized_correlation(shifted, reference)
        if value < 0.0:
            shifted = -shifted
            value = -value
        if value > best_correlation:
            best = shifted
            best_correlation = value
    return best, best_correlation


def representative_template(result: WindowResult) -> Tuple[np.ndarray, float, int, float]:
    start, end = result.start_index, result.end_index
    time_ms = result.dataset.values["time_ms"][start:end]
    detail = result.dataset.values["heartTemplateInput"][start:end]
    peak_times = time_ms[result.dataset.values["cleanPeak"][start:end] > 0.5]
    intervals_ms = np.diff(peak_times)
    valid_intervals = intervals_ms[(intervals_ms >= 500.0) & (intervals_ms <= 1500.0)]
    if valid_intervals.size < 3:
        raise ValueError("selected window has fewer than three valid heartbeat intervals")
    median_interval_ms = float(np.median(valid_intervals))

    cycles: List[np.ndarray] = []
    accepted_intervals: List[float] = []
    target_phase = np.arange(64, dtype=float) / 64.0
    for first, second in zip(peak_times[:-1], peak_times[1:]):
        interval = float(second - first)
        if interval < 500.0 or interval > 1500.0:
            continue
        if interval < median_interval_ms * 0.80 or interval > median_interval_ms * 1.20:
            continue
        targets = first + target_phase * interval
        cycle = np.interp(targets, time_ms, detail)
        cycle = cycle - float(np.mean(cycle))
        rms = float(np.sqrt(np.mean(cycle * cycle)))
        if not math.isfinite(rms) or rms < 1e-3:
            continue
        cycle = cycle / rms
        cycles.append(cycle)
        accepted_intervals.append(interval)
    if len(cycles) < 3:
        raise ValueError("selected window has fewer than three usable detail cycles")

    # The App retains at most the latest eight accepted real cycles.
    cycles = cycles[-8:]
    accepted_intervals = accepted_intervals[-8:]
    strongest = int(np.argmax(np.abs(cycles[0])))
    if cycles[0][strongest] < 0.0:
        cycles[0] = -cycles[0]
    aligned = cycles
    for _ in range(2):
        reference = np.median(np.vstack(aligned), axis=0)
        aligned = [align_cycle(cycle, reference)[0] for cycle in aligned]

    median_template = np.median(np.vstack(aligned), axis=0)
    robust = (np.roll(median_template, 1) + 2.0 * median_template + np.roll(median_template, -1)) / 4.0
    similarities = []
    for first in range(len(aligned)):
        peer_values = [
            abs(normalized_correlation(aligned[first], aligned[second]))
            for second in range(len(aligned))
            if second != first
        ]
        similarities.append(float(np.mean(peer_values)))
    medoid = aligned[int(np.argmax(similarities))].copy()
    if normalized_correlation(medoid, robust) < 0.0:
        medoid = -medoid

    display = 0.25 * robust + 0.75 * medoid
    display = display - float(np.mean(display))
    scale = float(np.max(np.abs(display)))
    if scale > 1e-9:
        display = display / scale
    edge_value = float((display[0] + display[-1]) / 2.0)
    display[0] = edge_value
    display[-1] = edge_value
    quality = float(np.mean([abs(normalized_correlation(cycle, display)) for cycle in aligned]))
    strongest = int(np.argmax(np.abs(display)))
    if display[strongest] < 0.0:
        display = -display
        strongest = int(np.argmax(np.abs(display)))
    display = circular_shift(display, int(64 * 0.30) - strongest)
    return display, float(np.median(accepted_intervals)) / 1000.0, len(aligned), quality


def repeat_template(template: np.ndarray, period_s: float, duration_s: float, fs_hz: float) -> Dict[str, np.ndarray]:
    times = np.arange(0.0, duration_s + 0.5 / fs_hz, 1.0 / fs_hz)
    phase_points = (np.mod(times, period_s) / period_s) * len(template)
    extended = np.concatenate((template, template[:1]))
    values = np.interp(phase_points, np.arange(len(template) + 1, dtype=float), extended)
    cycle_index = np.floor(times / period_s).astype(int)
    boundaries = np.zeros(len(times), dtype=int)
    boundaries[0] = 1
    boundaries[1:] = (cycle_index[1:] != cycle_index[:-1]).astype(int)
    return {
        "time_s": times,
        "enhanced_repeated": values,
        "cycle_index": cycle_index,
        "cycle_boundary": boundaries,
    }


def write_origin_csv(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    fields = (
        "Time_s",
        "Mixed_xDc",
        "Mixed_normalized",
        "CleanHeart",
        "CleanHeart_normalized",
        "HeartDetail_1_10Hz",
        "HeartDetail_normalized",
        "Enhanced75_normalized",
        "BeatBoundary",
        "CycleIndex",
        "LoggedBPM",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for index in range(len(arrays["time_s"])):
            writer.writerow(
                (
                    f"{arrays['time_s'][index]:.3f}",
                    f"{arrays['xDc'][index]:.7g}",
                    f"{arrays['mixed_normalized'][index]:.7g}",
                    f"{arrays['cleanHeart'][index]:.7g}",
                    f"{arrays['clean_normalized'][index]:.7g}",
                    f"{arrays['heartTemplateInput'][index]:.7g}",
                    f"{arrays['detail_normalized'][index]:.7g}",
                    f"{arrays['enhanced_normalized'][index]:.7g}",
                    int(arrays["cleanPeak"][index] > 0.5),
                    int(arrays["cycle_index"][index]),
                    "" if not math.isfinite(arrays["bpm"][index]) else f"{arrays['bpm'][index]:.3f}",
                )
            )


def write_repeated_origin_csv(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Time_s", "Enhanced75Repeated", "CycleBoundary", "CycleIndex"))
        for index in range(len(arrays["time_s"])):
            writer.writerow(
                (
                    f"{arrays['time_s'][index]:.3f}",
                    f"{arrays['enhanced_repeated'][index]:.7g}",
                    int(arrays["cycle_boundary"][index]),
                    int(arrays["cycle_index"][index]),
                )
            )


def write_template_csv(path: Path, template: np.ndarray, period_s: float) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("TemplatePoint", "Phase_0_1", "TimeInCycle_s", "Amplitude"))
        for index, value in enumerate(template):
            phase = index / len(template)
            writer.writerow((index, f"{phase:.7f}", f"{phase * period_s:.7f}", f"{value:.7g}"))


def svg_polyline(times: np.ndarray, values: np.ndarray, left: float, top: float, width: float, height: float) -> str:
    x_max = max(float(times[-1]), 1e-9)
    points = []
    for time_s, value in zip(times, values):
        x = left + float(time_s) / x_max * width
        y = top + (1.2 - float(value)) / 2.4 * height
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def write_selected_svg(path: Path, result: WindowResult, arrays: Dict[str, np.ndarray]) -> None:
    width, height = 1400, 620
    left, top, plot_w, plot_h = 120, 125, 1210, 360
    times = arrays["time_s"]
    enhanced = arrays["enhanced_normalized"]
    x_max = float(times[-1])
    peak_times = times[arrays["cleanPeak"] > 0.5]
    verticals = "\n".join(
        f'<line x1="{left + float(value) / x_max * plot_w:.2f}" y1="{top}" '
        f'x2="{left + float(value) / x_max * plot_w:.2f}" y2="{top + plot_h}" class="beat"/>'
        for value in peak_times
    )
    x_grid = "\n".join(
        f'<line x1="{left + second / x_max * plot_w:.2f}" y1="{top}" '
        f'x2="{left + second / x_max * plot_w:.2f}" y2="{top + plot_h}" class="grid"/>'
        for second in range(0, int(math.floor(x_max)) + 1)
    )
    x_labels = "\n".join(
        f'<text x="{left + second / x_max * plot_w:.2f}" y="{top + plot_h + 34}" class="tick" text-anchor="middle">{second}</text>'
        for second in range(0, int(math.floor(x_max)) + 1)
    )
    y_grid_parts = []
    for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
        y = top + (1.2 - value) / 2.4 * plot_h
        y_grid_parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>')
        y_grid_parts.append(f'<text x="{left - 16}" y="{y + 6:.2f}" class="tick" text-anchor="end">{value:+.1f}</text>')
    polyline = svg_polyline(times, enhanced, left, top, plot_w, plot_h)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; fill: #202124; }}
.title {{ font-size: 30px; font-weight: 500; }} .subtitle {{ font-size: 19px; }}
.tick {{ font-size: 16px; fill: #5f6368; }} .axis-label {{ font-size: 19px; fill: #4b4f52; }}
.grid {{ stroke: #dfe3e8; stroke-width: 1; }} .border {{ fill: none; stroke: #9aa0a6; stroke-width: 1.4; }}
.beat {{ stroke: #f28c45; stroke-width: 1.6; opacity: 0.72; }}
.wave {{ fill: none; stroke: #298fff; stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
.note {{ font-size: 17px; fill: #5f6368; }}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="48" class="title">周期增强心搏展示信号（10 秒多周期）</text>
<text x="{left}" y="84" class="subtitle">{result.dataset.position} · {result.dataset.path.name} · {result.beat_count} 个检测边界 · BPM 中位数 {result.median_bpm:.1f}</text>
{x_grid}
{''.join(y_grid_parts)}
{verticals}
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="border"/>
<polyline points="{polyline}" class="wave"/>
{x_labels}
<text x="{left + plot_w / 2}" y="{top + plot_h + 74}" class="axis-label" text-anchor="middle">时间（s）</text>
<text x="32" y="{top + plot_h / 2}" class="axis-label" text-anchor="middle" transform="rotate(-90 32 {top + plot_h / 2})">归一化幅值</text>
<text x="{left}" y="{height - 38}" class="note">75%代表周期保形增强；橙线为真实检测边界。该图用于周期形态展示，不作为原始连续心搏或 BPM 计算输入。</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def write_repeated_svg(
    path: Path,
    result: WindowResult,
    arrays: Dict[str, np.ndarray],
    retained_cycles: int,
    template_quality: float,
    period_s: float,
) -> None:
    width, height = 1400, 620
    left, top, plot_w, plot_h = 120, 125, 1210, 360
    times = arrays["time_s"]
    values = arrays["enhanced_repeated"]
    x_max = float(times[-1])
    boundary_times = times[arrays["cycle_boundary"] > 0]
    verticals = "\n".join(
        f'<line x1="{left + float(value) / x_max * plot_w:.2f}" y1="{top}" '
        f'x2="{left + float(value) / x_max * plot_w:.2f}" y2="{top + plot_h}" class="beat"/>'
        for value in boundary_times
    )
    x_grid = "\n".join(
        f'<line x1="{left + second / x_max * plot_w:.2f}" y1="{top}" '
        f'x2="{left + second / x_max * plot_w:.2f}" y2="{top + plot_h}" class="grid"/>'
        for second in range(0, 11)
    )
    x_labels = "\n".join(
        f'<text x="{left + second / x_max * plot_w:.2f}" y="{top + plot_h + 34}" class="tick" text-anchor="middle">{second}</text>'
        for second in range(0, 11)
    )
    y_grid_parts = []
    for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
        y = top + (1.2 - value) / 2.4 * plot_h
        y_grid_parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>')
        y_grid_parts.append(f'<text x="{left - 16}" y="{y + 6:.2f}" class="tick" text-anchor="end">{value:+.1f}</text>')
    polyline = svg_polyline(times, values, left, top, plot_w, plot_h)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; fill: #202124; }}
.title {{ font-size: 30px; font-weight: 500; }} .subtitle {{ font-size: 19px; }}
.tick {{ font-size: 16px; fill: #5f6368; }} .axis-label {{ font-size: 19px; fill: #4b4f52; }}
.grid {{ stroke: #dfe3e8; stroke-width: 1; }} .border {{ fill: none; stroke: #9aa0a6; stroke-width: 1.4; }}
.beat {{ stroke: #f28c45; stroke-width: 1.6; opacity: 0.72; }}
.wave {{ fill: none; stroke: #298fff; stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
.note {{ font-size: 17px; fill: #5f6368; }}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="48" class="title">代表性周期增强心搏展示信号（10 秒模板重复）</text>
<text x="{left}" y="84" class="subtitle">{result.dataset.position} · {retained_cycles} 个真实周期构造 · 模板一致性 {template_quality:.2f} · 周期 {period_s:.3f}s（约 {60.0 / period_s:.1f} BPM）</text>
{x_grid}
{''.join(y_grid_parts)}
{verticals}
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="border"/>
<polyline points="{polyline}" class="wave"/>
{x_labels}
<text x="{left + plot_w / 2}" y="{top + plot_h + 74}" class="axis-label" text-anchor="middle">时间（s）</text>
<text x="32" y="{top + plot_h / 2}" class="axis-label" text-anchor="middle" transform="rotate(-90 32 {top + plot_h / 2})">归一化幅值</text>
<text x="{left}" y="{height - 38}" class="note">从真实周期构造75%代表周期保形模板后重复显示；橙线为模板周期边界。非原始连续心搏，不参与 BPM 计算。</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def write_comparison_svg(path: Path, windows: Sequence[WindowResult]) -> None:
    width = 1400
    left, plot_w = 120, 1210
    panel_h, panel_gap = 170, 82
    top = 100
    height = top + len(windows) * (panel_h + panel_gap) + 20
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#202124}.title{font-size:28px;font-weight:500}.label{font-size:18px}.tick{font-size:14px;fill:#5f6368}.grid{stroke:#dfe3e8;stroke-width:1}.border{fill:none;stroke:#9aa0a6;stroke-width:1.2}.wave{fill:none;stroke:#298fff;stroke-width:2.2;stroke-linejoin:round;stroke-linecap:round}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="46" class="title">三位置后段最佳 10 秒连续 cleanHeart 周期性比较</text>',
    ]
    for panel, result in enumerate(windows):
        panel_top = top + panel * (panel_h + panel_gap)
        start, end = result.start_index, result.end_index
        times = result.dataset.values["time_ms"][start:end]
        times = (times - times[0]) / 1000.0
        signal = robust_normalize(result.dataset.values["cleanHeart"][start:end])
        x_max = float(times[-1])
        for second in range(0, int(math.floor(x_max)) + 1):
            x = left + second / x_max * plot_w
            pieces.append(f'<line x1="{x:.2f}" y1="{panel_top}" x2="{x:.2f}" y2="{panel_top + panel_h}" class="grid"/>')
            pieces.append(f'<text x="{x:.2f}" y="{panel_top + panel_h + 25}" class="tick" text-anchor="middle">{second}</text>')
        zero_y = panel_top + panel_h / 2
        pieces.append(f'<line x1="{left}" y1="{zero_y}" x2="{left + plot_w}" y2="{zero_y}" class="grid"/>')
        pieces.append(f'<rect x="{left}" y="{panel_top}" width="{plot_w}" height="{panel_h}" class="border"/>')
        pieces.append(f'<polyline points="{svg_polyline(times, signal, left, panel_top, plot_w, panel_h)}" class="wave"/>')
        pieces.append(
            f'<text x="{left}" y="{panel_top - 16}" class="label">{result.dataset.position} · 综合分 {result.score:.3f} · '
            f'连续心搏自相关 {result.clean_autocorrelation:.3f} · RR-CV {result.rr_cv:.3f} · {result.median_bpm:.1f} BPM</text>'
        )
    pieces.append('</svg>')
    path.write_text("\n".join(pieces), encoding="utf-8")


def write_report(path: Path, windows: Sequence[WindowResult], selected: WindowResult) -> None:
    ranking = sorted(windows, key=lambda item: item.score, reverse=True)
    rows = "\n".join(
        f"| {index} | {item.dataset.position} | `{item.dataset.path.name}` | "
        f"{item.start_s:.2f}–{item.end_s:.2f} | {item.beat_count} | {item.median_bpm:.1f} | "
        f"{item.clean_autocorrelation:.3f} | {item.detail_autocorrelation:.3f} | "
        f"{item.rr_cv:.3f} | {item.template_quality:.3f} | **{item.score:.3f}** |"
        for index, item in enumerate(ranking, start=1)
    )
    text = f"""# 2026-07-16 三位置心搏周期性筛选与 Origin 导出

## 输入位置

- `152843`：胸口；
- `153115`：腹部；
- `153454`：人体倚靠椅背。

## 选择原则

只在每份记录后 60% 范围内搜索连续 10 秒窗口，并用未经周期重构的 `cleanHeart` 与 `heartTemplateInput` 评价周期性。综合分由以下内部指标组成：

- `cleanHeart` 周期自相关：40%；
- 1–10 Hz 心搏细节输入周期自相关：30%；
- 真实检测峰间期稳定性：20%；
- App 模板一致性：10%。

`heartTemplateEnhanced` 不参与位置和时间窗选择，避免用增强后的规则性反向挑选数据。当前没有同步 ECG，因此这些分数只能用于三份数据的内部比较，不能作为心率绝对准确度证明。

## 排名

| 排名 | 位置 | 文件 | 记录内窗口（s） | 边界数 | BPM中位数 | cleanHeart自相关 | 细节自相关 | RR-CV | 模板质量 | 综合分 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

## 最终选择

选择 **{selected.dataset.position}** 的 `{selected.dataset.path.name}`，窗口位于记录开始后的 **{selected.start_s:.2f}–{selected.end_s:.2f} 秒**。该窗口综合分最高，为 **{selected.score:.3f}**，包含 **{selected.beat_count}** 个真实检测边界，日志 BPM 中位数为 **{selected.median_bpm:.1f} BPM**。

若目标是得到与先前“四周期代表模板图”相同但更长的展示效果，Origin 主展示建议使用 `selected_repeated_template_origin_10s.csv` 的：

- X：`Time_s`；
- Y：`Enhanced75Repeated`；
- `CycleBoundary=1` 的时间点可添加为竖直参考线。

该主展示曲线由所选窗口中的真实周期构造一个固定代表模板，再按真实中位周期重复到 10 秒，因此不是未经处理的连续心搏。`selected_origin_10s.csv` 另外保留了同一真实 10 秒窗口的混合信号、连续 `cleanHeart`、1–10 Hz 细节信号和日志内连续增强信号，可作为处理链证据。

### Origin 绘图设置

1. 导入 `selected_repeated_template_origin_10s.csv`；
2. 将 A 列 `Time_s` 设为 X，将 B 列 `Enhanced75Repeated` 设为 Y；
3. 使用二维折线图，X 轴设为 `0–10 s`、主刻度 `1 s`；
4. Y 轴设为 `-1.2–1.2`、主刻度 `0.5`；
5. 建议使用蓝色实线、线宽 `1.5–2 pt`；
6. C 列 `CycleBoundary=1` 对应的 A 列时间可以添加为橙色竖直参考线；
7. 图注必须写明“固定代表模板重复展示，非原始连续心搏，不参与 BPM”。

`selected_representative_template_64points.csv` 保存了实际使用的单周期 64 点模板，便于复核重复曲线并在 Origin 中单独绘制代表性单周期。

若论文需要同时证明处理链，应另外绘制 `Mixed_normalized` 和 `CleanHeart_normalized`，并把增强曲线标注为“75%代表周期保形增强展示信号（非原始连续心搏，不参与BPM）”。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/research/20260716_three_position_periodicity"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [load_dataset(*specification) for specification in DATASETS]
    windows = [best_late_window(dataset, window_s=10.0) for dataset in datasets]
    selected = max(windows, key=lambda item: (round(item.score, 4), item.start_s))
    arrays = selected_arrays(selected)
    template, period_s, retained_cycles, template_quality = representative_template(selected)
    repeated = repeat_template(template, period_s, duration_s=10.0, fs_hz=selected.dataset.fs_hz)

    write_metrics(args.output_dir / "position_window_metrics.csv", windows, selected)
    write_origin_csv(args.output_dir / "selected_origin_10s.csv", arrays)
    write_selected_svg(args.output_dir / "selected_enhanced_10s.svg", selected, arrays)
    write_repeated_origin_csv(args.output_dir / "selected_repeated_template_origin_10s.csv", repeated)
    write_template_csv(args.output_dir / "selected_representative_template_64points.csv", template, period_s)
    write_repeated_svg(
        args.output_dir / "selected_repeated_template_10s.svg",
        selected,
        repeated,
        retained_cycles,
        template_quality,
        period_s,
    )
    write_comparison_svg(args.output_dir / "three_position_cleanheart_comparison.svg", windows)
    write_report(args.output_dir / "结果说明.md", windows, selected)

    print(f"selected={selected.dataset.dataset_id}")
    print(f"window={selected.start_s:.3f}-{selected.end_s:.3f}s")
    print(f"score={selected.score:.6f}")
    print(f"bpm={selected.median_bpm:.3f}")
    print(f"beats={selected.beat_count}")
    print(f"retained_cycles={retained_cycles}")
    print(f"template_quality={template_quality:.6f}")
    print(f"template_period_s={period_s:.6f}")


if __name__ == "__main__":
    main()
