"""Create paper-like but non-identical heartbeat cycles for all three positions.

The main display is cycle-adaptive: every output cycle comes from its matching
real 1-10 Hz detail cycle, phase-aligned and blended with a robust template.
Only two phase harmonics are retained to suppress burrs and extra peaks.  Real
RR timing and small cycle-to-cycle morphology/amplitude variations are kept.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from export_origin_periodic_heartbeat import (
    DATASETS,
    Dataset,
    WindowResult,
    align_cycle,
    autocorrelation_near_period,
    evaluate_window,
    load_dataset,
    normalized_correlation,
    robust_normalize,
    svg_polyline,
)


OUTPUT_DIR = Path("docs/research/20260716_position_adaptive_heartbeat")
TEMPLATE_POINTS = 64
TEMPLATE_WEIGHT = 0.68
PHASE_HARMONICS = 2


@dataclass
class PositionResult:
    dataset: Dataset
    base_window: WindowResult
    time_s: np.ndarray
    clean: np.ndarray
    detail: np.ndarray
    narrowband: np.ndarray
    adaptive: np.ndarray
    boundaries: np.ndarray
    cycle_index: np.ndarray
    rr_s: np.ndarray
    logged_bpm: np.ndarray
    retained_cycles: int
    template_quality: float
    median_rr_s: float
    adaptive_consistency: float
    narrowband_consistency: float
    adaptive_variation: float
    narrowband_variation: float
    adaptive_roughness: float
    narrowband_roughness: float
    adaptive_fidelity: float
    narrowband_fidelity: float


def harmonic_smooth(values: np.ndarray, harmonics: int = PHASE_HARMONICS) -> np.ndarray:
    spectrum = np.fft.rfft(values)
    spectrum[harmonics + 1 :] = 0.0
    output = np.fft.irfft(spectrum, n=len(values))
    output -= float(np.mean(output))
    rms = float(np.sqrt(np.mean(output * output)))
    return output / max(rms, 1e-9)


def normalize_shape(values: np.ndarray) -> np.ndarray:
    output = values - float(np.mean(values))
    scale = float(np.max(np.abs(output)))
    if scale > 1e-9:
        output = output / scale
    return output


def rotate_display_phase(values: np.ndarray, target_fraction: float = 0.27) -> Tuple[np.ndarray, int]:
    strongest = int(np.argmax(np.abs(values)))
    output = values.copy()
    if output[strongest] < 0.0:
        output = -output
        strongest = int(np.argmax(np.abs(output)))
    shift = int(round(len(output) * target_fraction)) - strongest
    return np.roll(output, shift), shift


def fir_bandpass(values: np.ndarray, fs_hz: float, low_hz: float, high_hz: float, center_hz: float) -> np.ndarray:
    taps = 151
    n = np.arange(taps, dtype=float) - (taps - 1) / 2.0
    high = 2.0 * high_hz / fs_hz * np.sinc(2.0 * high_hz / fs_hz * n)
    low = 2.0 * low_hz / fs_hz * np.sinc(2.0 * low_hz / fs_hz * n)
    kernel = (high - low) * np.hamming(taps)
    omega = 2.0 * math.pi * center_hz / fs_hz
    gain = abs(np.sum(kernel * np.exp(-1j * omega * np.arange(taps))))
    kernel = kernel / max(float(gain), 1e-9)
    return np.convolve(values, kernel, mode="same")


def extract_real_cycles(
    dataset: Dataset,
    start_ms: float,
    end_ms: float,
) -> Tuple[List[np.ndarray], List[float], List[float], np.ndarray]:
    time_ms = dataset.values["time_ms"]
    detail = dataset.values["heartTemplateInput"]
    all_peaks = time_ms[dataset.values["cleanPeak"] > 0.5]
    peaks = all_peaks[(all_peaks >= start_ms) & (all_peaks <= end_ms + 1600.0)]
    if len(peaks) < 5:
        raise ValueError(f"{dataset.dataset_id}: too few clean peaks")
    intervals = np.diff(peaks)
    plausible = intervals[(intervals >= 500.0) & (intervals <= 1500.0)]
    median_interval = float(np.median(plausible))

    phase = np.arange(TEMPLATE_POINTS, dtype=float) / TEMPLATE_POINTS
    cycles: List[np.ndarray] = []
    rms_values: List[float] = []
    intervals_ms: List[float] = []
    accepted_peaks: List[float] = []
    for first, second in zip(peaks[:-1], peaks[1:]):
        interval = float(second - first)
        if interval < 500.0 or interval > 1500.0:
            continue
        cycle = np.interp(first + phase * interval, time_ms, detail)
        cycle -= float(np.mean(cycle))
        rms = float(np.sqrt(np.mean(cycle * cycle)))
        if rms < 1e-3 or not math.isfinite(rms):
            continue
        cycles.append(cycle / rms)
        rms_values.append(rms)
        intervals_ms.append(interval)
        accepted_peaks.append(float(first))
    if len(cycles) < 5:
        raise ValueError(f"{dataset.dataset_id}: too few usable cycles")
    accepted_peaks.append(accepted_peaks[-1] + intervals_ms[-1])
    return cycles, rms_values, intervals_ms, np.asarray(accepted_peaks, dtype=float)


def align_and_smooth_cycles(cycles: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray, int, float]:
    aligned = [cycle.copy() for cycle in cycles]
    strongest = int(np.argmax(np.abs(aligned[0])))
    if aligned[0][strongest] < 0.0:
        aligned[0] = -aligned[0]
    for _ in range(2):
        reference = np.median(np.vstack(aligned), axis=0)
        aligned = [align_cycle(cycle, reference)[0] for cycle in aligned]

    smoothed = [harmonic_smooth(cycle) for cycle in aligned]
    robust = harmonic_smooth(np.median(np.vstack(smoothed), axis=0))
    robust, shift = rotate_display_phase(robust)
    smoothed = [np.roll(cycle, shift) for cycle in smoothed]
    smoothed = [(-cycle if normalized_correlation(cycle, robust) < 0.0 else cycle) for cycle in smoothed]
    quality = float(np.median([abs(normalized_correlation(cycle, robust)) for cycle in smoothed]))
    return smoothed, robust, shift, quality


def build_adaptive_signal(
    dataset: Dataset,
    start_ms: float,
    duration_s: float,
    cycles: Sequence[np.ndarray],
    robust: np.ndarray,
    rms_values: Sequence[float],
    intervals_ms: Sequence[float],
    peaks_ms: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_ms = dataset.values["time_ms"]
    begin = int(np.searchsorted(time_ms, start_ms, side="left"))
    end = int(np.searchsorted(time_ms, start_ms + duration_s * 1000.0, side="right"))
    output_time = time_ms[begin:end]
    adaptive = np.full(len(output_time), np.nan, dtype=float)
    cycle_index = np.full(len(output_time), -1, dtype=int)
    boundaries = np.zeros(len(output_time), dtype=int)
    rr_at_sample = np.full(len(output_time), np.nan, dtype=float)
    median_rms = float(np.median(rms_values))
    common_edge = float((robust[0] + robust[-1]) / 2.0)

    for index, (cycle, first, interval_ms, rms) in enumerate(zip(cycles, peaks_ms[:-1], intervals_ms, rms_values)):
        second = first + interval_ms
        local_mask = (output_time >= first) & (output_time < second)
        if not np.any(local_mask):
            continue
        shape = TEMPLATE_WEIGHT * robust + (1.0 - TEMPLATE_WEIGHT) * cycle
        shape = normalize_shape(shape)
        # Preserve subtle amplitude differences but return to a shared edge at
        # every real boundary so the output remains continuous.
        amplitude = float(np.clip(1.0 + 0.22 * (rms / median_rms - 1.0), 0.88, 1.12))
        phase_grid = np.arange(TEMPLATE_POINTS, dtype=float) / TEMPLATE_POINTS
        envelope = np.sin(math.pi * phase_grid) ** 2
        shape = shape * (1.0 + (amplitude - 1.0) * envelope)
        edge_points = 5
        for point in range(edge_points):
            blend = point / edge_points
            shape[point] = common_edge * (1.0 - blend) + shape[point] * blend
            shape[-point - 1] = common_edge * (1.0 - blend) + shape[-point - 1] * blend
        shape[0] = common_edge
        shape[-1] = common_edge

        local_times = output_time[local_mask]
        phase = (local_times - first) / interval_ms
        extended = np.concatenate((shape, shape[:1]))
        values = np.interp(phase * TEMPLATE_POINTS, np.arange(TEMPLATE_POINTS + 1), extended)
        adaptive[local_mask] = values
        cycle_index[local_mask] = index
        rr_at_sample[local_mask] = interval_ms / 1000.0
        boundary_position = int(np.searchsorted(output_time, first, side="left"))
        if 0 <= boundary_position < len(boundaries):
            boundaries[boundary_position] = 1

    finite = np.isfinite(adaptive)
    if not np.all(finite):
        valid_indices = np.flatnonzero(finite)
        if valid_indices.size:
            adaptive[~finite] = np.interp(np.flatnonzero(~finite), valid_indices, adaptive[finite])
        else:
            adaptive[:] = 0.0
    return output_time, adaptive, boundaries, cycle_index, rr_at_sample


def resampled_cycles(values: np.ndarray, time_ms: np.ndarray, peaks_ms: np.ndarray) -> List[np.ndarray]:
    cycles: List[np.ndarray] = []
    phase = np.arange(TEMPLATE_POINTS, dtype=float) / TEMPLATE_POINTS
    for first, second in zip(peaks_ms[:-1], peaks_ms[1:]):
        if second <= first:
            continue
        cycle = np.interp(first + phase * (second - first), time_ms, values)
        cycle -= float(np.mean(cycle))
        rms = float(np.sqrt(np.mean(cycle * cycle)))
        if rms > 1e-6:
            cycles.append(cycle / rms)
    return cycles


def cycle_metrics(values: np.ndarray, detail: np.ndarray, time_ms: np.ndarray, peaks_ms: np.ndarray) -> Tuple[float, float, float, float]:
    cycles = resampled_cycles(values, time_ms, peaks_ms)
    if len(cycles) < 3:
        return 0.0, 0.0, 1.0, 0.0
    reference = np.median(np.vstack(cycles), axis=0)
    aligned = [align_cycle(cycle, reference)[0] for cycle in cycles]
    consistency = float(np.median([abs(normalized_correlation(cycle, reference)) for cycle in aligned]))
    stack = np.vstack([normalize_shape(cycle) for cycle in aligned])
    variation = float(np.median(np.std(stack, axis=0)))
    normalized = robust_normalize(values)
    roughness = float(np.percentile(np.abs(np.diff(normalized, n=2)), 95))
    fidelity = abs(normalized_correlation(robust_normalize(values), robust_normalize(detail)))
    return consistency, variation, roughness, fidelity


def best_paperlike_window(dataset: Dataset, window_s: float = 10.0) -> WindowResult:
    time_ms = dataset.values["time_ms"]
    duration_s = float((time_ms[-1] - time_ms[0]) / 1000.0)
    search_start_ms = float(time_ms[0] + duration_s * 0.40 * 1000.0)
    first_index = int(np.searchsorted(time_ms, search_start_ms, side="left"))
    step = max(1, int(round(dataset.fs_hz * 0.5)))
    candidates: List[WindowResult] = []
    for begin in range(first_index, len(time_ms), step):
        end_ms = float(time_ms[begin]) + window_s * 1000.0
        end = int(np.searchsorted(time_ms, end_ms, side="right"))
        if end - begin < dataset.fs_hz * 9.2:
            continue
        candidate = evaluate_window(dataset, begin, end)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        raise ValueError(f"{dataset.dataset_id}: no valid paper-like candidate window")

    # Only the strongest underlying windows enter the display-oriented search.
    candidates = sorted(candidates, key=lambda item: item.score, reverse=True)[:40]
    best = candidates[0]
    best_score = -1.0
    for candidate in candidates:
        start, end = candidate.start_index, candidate.end_index
        local_time = time_ms[start:end]
        local_peaks = local_time[dataset.values["cleanPeak"][start:end] > 0.5]
        rr = np.diff(local_peaks) / 1000.0
        rr = rr[(rr >= 0.5) & (rr <= 1.5)]
        if rr.size < 7:
            continue
        median_rr = float(np.median(rr))
        f0 = 1.0 / median_rr
        filtered = fir_bandpass(
            dataset.values["heartTemplateInput"],
            dataset.fs_hz,
            max(0.65, f0 - 0.35),
            min(2.10, f0 + 0.35),
            f0,
        )[start:end]
        filtered = robust_normalize(filtered)
        detail = dataset.values["heartTemplateInput"][start:end]
        consistency, _, _, fidelity = cycle_metrics(filtered, detail, local_time, local_peaks)
        periodicity = autocorrelation_near_period(filtered, dataset.fs_hz, median_rr)
        amplitudes = []
        for first, second in zip(local_peaks[:-1], local_peaks[1:]):
            mask = (local_time >= first) & (local_time < second)
            if np.sum(mask) >= 10:
                amplitudes.append(float(np.ptp(filtered[mask])))
        amplitude_cv = float(np.std(amplitudes) / max(np.mean(amplitudes), 1e-9)) if amplitudes else 1.0
        subtle_variation_score = math.exp(-5.0 * abs(amplitude_cv - 0.16))
        score = (
            0.38 * periodicity
            + 0.28 * consistency
            + 0.14 * fidelity
            + 0.12 * subtle_variation_score
            + 0.08 * candidate.score
        )
        if score > best_score or (round(score, 5) == round(best_score, 5) and candidate.start_s > best.start_s):
            best = candidate
            best_score = score
    return best


def process_position(dataset: Dataset, base_window: WindowResult) -> PositionResult:
    time_ms = dataset.values["time_ms"]
    all_peaks = time_ms[dataset.values["cleanPeak"] > 0.5]
    desired_start = time_ms[0] + base_window.start_s * 1000.0
    start_candidates = all_peaks[all_peaks >= desired_start]
    if start_candidates.size == 0:
        raise ValueError(f"{dataset.dataset_id}: no output anchor peak")
    start_ms = float(start_candidates[0])
    duration_s = 10.0

    cycles, rms_values, intervals_ms, peaks_ms = extract_real_cycles(
        dataset,
        start_ms,
        start_ms + duration_s * 1000.0,
    )
    smoothed_cycles, robust, _, template_quality = align_and_smooth_cycles(cycles)
    output_time_ms, adaptive, boundaries, cycle_index, rr_at_sample = build_adaptive_signal(
        dataset,
        start_ms,
        duration_s,
        smoothed_cycles,
        robust,
        rms_values,
        intervals_ms,
        peaks_ms,
    )
    begin = int(np.searchsorted(time_ms, output_time_ms[0], side="left"))
    end = begin + len(output_time_ms)
    detail = dataset.values["heartTemplateInput"][begin:end]
    clean = dataset.values["cleanHeart"][begin:end]
    logged_bpm = dataset.values["bpm"][begin:end]

    median_rr_s = float(np.median(intervals_ms)) / 1000.0
    f0 = 1.0 / median_rr_s
    # A narrow, RR-centred continuous band is the closest analogue to the
    # paper's smooth heartbeat mode.  It intentionally suppresses SCG detail.
    low_hz = max(0.65, f0 - 0.35)
    high_hz = min(2.10, f0 + 0.35)
    full_narrowband = fir_bandpass(dataset.values["heartTemplateInput"], dataset.fs_hz, low_hz, high_hz, f0)
    narrowband = robust_normalize(full_narrowband[begin:end])
    adaptive = robust_normalize(adaptive)
    clean = robust_normalize(clean)
    detail = robust_normalize(detail)

    metric_peaks = peaks_ms[(peaks_ms >= output_time_ms[0]) & (peaks_ms <= output_time_ms[-1] + 1000.0)]
    adaptive_metrics = cycle_metrics(adaptive, detail, output_time_ms, metric_peaks)
    narrowband_metrics = cycle_metrics(narrowband, detail, output_time_ms, metric_peaks)
    return PositionResult(
        dataset=dataset,
        base_window=base_window,
        time_s=(output_time_ms - output_time_ms[0]) / 1000.0,
        clean=clean,
        detail=detail,
        narrowband=narrowband,
        adaptive=adaptive,
        boundaries=boundaries,
        cycle_index=cycle_index,
        rr_s=rr_at_sample,
        logged_bpm=logged_bpm,
        retained_cycles=len(smoothed_cycles),
        template_quality=template_quality,
        median_rr_s=median_rr_s,
        adaptive_consistency=adaptive_metrics[0],
        narrowband_consistency=narrowband_metrics[0],
        adaptive_variation=adaptive_metrics[1],
        narrowband_variation=narrowband_metrics[1],
        adaptive_roughness=adaptive_metrics[2],
        narrowband_roughness=narrowband_metrics[2],
        adaptive_fidelity=adaptive_metrics[3],
        narrowband_fidelity=narrowband_metrics[3],
    )


def write_origin_csv(path: Path, result: PositionResult) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "Time_s",
                "AdaptiveHeartbeat",
                "NarrowbandContinuous",
                "CleanHeartNormalized",
                "HeartDetailNormalized",
                "CycleBoundary",
                "CycleIndex",
                "RR_s",
                "LoggedBPM",
            )
        )
        for index in range(len(result.time_s)):
            writer.writerow(
                (
                    f"{result.time_s[index]:.3f}",
                    f"{result.adaptive[index]:.7g}",
                    f"{result.narrowband[index]:.7g}",
                    f"{result.clean[index]:.7g}",
                    f"{result.detail[index]:.7g}",
                    int(result.boundaries[index]),
                    int(result.cycle_index[index]),
                    "" if not math.isfinite(result.rr_s[index]) else f"{result.rr_s[index]:.3f}",
                    "" if not math.isfinite(result.logged_bpm[index]) else f"{result.logged_bpm[index]:.3f}",
                )
            )


def svg_panel(
    times: np.ndarray,
    values: np.ndarray,
    boundaries: np.ndarray,
    left: float,
    top: float,
    width: float,
    height: float,
    show_x_labels: bool,
) -> str:
    x_max = float(times[-1])
    grid_parts: List[str] = []
    for second in range(0, 11):
        x = left + second / x_max * width
        grid_parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + height}" class="grid"/>')
        if show_x_labels:
            grid_parts.append(f'<text x="{x:.2f}" y="{top + height + 26}" class="tick" text-anchor="middle">{second}</text>')
    for value in (-1.0, 0.0, 1.0):
        y = top + (1.2 - value) / 2.4 * height
        grid_parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + width}" y2="{y:.2f}" class="grid"/>')
        grid_parts.append(f'<text x="{left - 12}" y="{y + 5:.2f}" class="tick" text-anchor="end">{value:+.0f}</text>')
    for boundary_time in times[boundaries > 0]:
        x = left + float(boundary_time) / x_max * width
        grid_parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + height}" class="beat"/>')
    points = svg_polyline(times, values, left, top, width, height)
    grid_parts.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" class="border"/>')
    grid_parts.append(f'<polyline points="{points}" class="wave"/>')
    return "\n".join(grid_parts)


def svg_style() -> str:
    return '''<style>
text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#202124}.title{font-size:29px;font-weight:500}.subtitle{font-size:18px}.panel-label{font-size:18px;font-weight:500}.tick{font-size:14px;fill:#5f6368}.axis-label{font-size:18px;fill:#4b4f52}.grid{stroke:#dfe3e8;stroke-width:1}.border{fill:none;stroke:#9aa0a6;stroke-width:1.3}.beat{stroke:#f28c45;stroke-width:1.25;opacity:.55}.wave{fill:none;stroke:#298fff;stroke-width:2.7;stroke-linejoin:round;stroke-linecap:round}.note{font-size:16px;fill:#5f6368}
</style>'''


def write_position_svg(path: Path, result: PositionResult) -> None:
    width, height = 1400, 620
    left, top, plot_w, plot_h = 120, 125, 1210, 360
    panel = svg_panel(result.time_s, result.adaptive, result.boundaries, left, top, plot_w, plot_h, True)
    duration_start = result.base_window.start_s
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
{svg_style()}<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="48" class="title">{result.dataset.position}：逐周期自适应心搏增强（10秒）</text>
<text x="{left}" y="84" class="subtitle">后段最优窗口约 {duration_start:.1f}s 起 · {result.retained_cycles} 个真实周期 · 模板一致性 {result.template_quality:.2f} · 中位周期 {result.median_rr_s:.3f}s</text>
{panel}
<text x="{left + plot_w / 2}" y="{top + plot_h + 68}" class="axis-label" text-anchor="middle">时间（s）</text>
<text x="30" y="{top + plot_h / 2}" class="axis-label" text-anchor="middle" transform="rotate(-90 30 {top + plot_h / 2})">归一化幅值</text>
<text x="{left}" y="{height - 34}" class="note">每个周期均来自对应真实周期；保留实际RR间距及细微形态/幅值变化，抑制毛刺和额外小峰。展示增强，不参与BPM计算。</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def write_narrowband_position_svg(path: Path, result: PositionResult) -> None:
    width, height = 1400, 620
    left, top, plot_w, plot_h = 120, 125, 1210, 360
    no_boundaries = np.zeros_like(result.boundaries)
    panel = svg_panel(result.time_s, result.narrowband, no_boundaries, left, top, plot_w, plot_h, True)
    f0 = 1.0 / result.median_rr_s
    low_hz = max(0.65, f0 - 0.35)
    high_hz = min(2.10, f0 + 0.35)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
{svg_style()}<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="48" class="title">{result.dataset.position}：连续窄带心搏候选（后段最优10秒）</text>
<text x="{left}" y="84" class="subtitle">原记录 {result.base_window.start_s:.2f}–{result.base_window.end_s:.2f}s · RR中心频率 {f0:.2f}Hz · 离线零相位带宽 {low_hz:.2f}–{high_hz:.2f}Hz</text>
{panel}
<text x="{left + plot_w / 2}" y="{top + plot_h + 68}" class="axis-label" text-anchor="middle">时间（s）</text>
<text x="30" y="{top + plot_h / 2}" class="axis-label" text-anchor="middle" transform="rotate(-90 30 {top + plot_h / 2})">归一化幅值</text>
<text x="{left}" y="{height - 34}" class="note">真实连续信号的窄带平滑结果：保留自然幅值和周期变化，不复制模板；已抑制毛刺和额外小峰，不代表高保真SCG。</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def write_method_comparison_svg(path: Path, result: PositionResult) -> None:
    width, height = 1400, 880
    left, plot_w = 120, 1210
    top, panel_h, gap = 110, 180, 86
    signals = (
        ("原连续 cleanHeart", result.clean),
        ("连续窄带平滑（无模板重复）", result.narrowband),
        ("逐周期自适应增强（最终展示）", result.adaptive),
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_style(),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="48" class="title">{result.dataset.position}：毛刺/额外峰处理方法对比</text>',
        f'<text x="{left}" y="80" class="subtitle">蓝线均来自同一真实10秒窗口；最终曲线不是固定模板重复</text>',
    ]
    for index, (label, values) in enumerate(signals):
        panel_top = top + index * (panel_h + gap)
        parts.append(f'<text x="{left}" y="{panel_top - 15}" class="panel-label">{label}</text>')
        parts.append(svg_panel(result.time_s, values, result.boundaries, left, panel_top, plot_w, panel_h, index == 2))
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def write_all_positions_svg(path: Path, results: Sequence[PositionResult]) -> None:
    width = 1400
    left, plot_w = 120, 1210
    top, panel_h, gap = 100, 190, 88
    height = top + len(results) * (panel_h + gap) + 10
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_style(),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="48" class="title">胸口、腹部、椅背各自后段最优10秒：逐周期自适应增强</text>',
    ]
    for index, result in enumerate(results):
        panel_top = top + index * (panel_h + gap)
        parts.append(
            f'<text x="{left}" y="{panel_top - 16}" class="panel-label">{result.dataset.position} · '
            f'{result.retained_cycles}个真实周期 · 一致性{result.adaptive_consistency:.2f} · '
            f'变异度{result.adaptive_variation:.2f} · 中位RR {result.median_rr_s:.3f}s</text>'
        )
        parts.append(svg_panel(result.time_s, result.adaptive, result.boundaries, left, panel_top, plot_w, panel_h, index == len(results) - 1))
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def write_all_positions_narrowband_svg(path: Path, results: Sequence[PositionResult]) -> None:
    width = 1400
    left, plot_w = 120, 1210
    top, panel_h, gap = 100, 190, 88
    height = top + len(results) * (panel_h + gap) + 10
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_style(),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="48" class="title">胸口、腹部、椅背各自最优10秒：真实连续窄带心搏候选</text>',
    ]
    for index, result in enumerate(results):
        panel_top = top + index * (panel_h + gap)
        f0 = 1.0 / result.median_rr_s
        parts.append(
            f'<text x="{left}" y="{panel_top - 16}" class="panel-label">{result.dataset.position} · '
            f'原记录{result.base_window.start_s:.2f}–{result.base_window.end_s:.2f}s · '
            f'中心{f0:.2f}Hz · 一致性{result.narrowband_consistency:.2f} · '
            f'自然变异度{result.narrowband_variation:.2f}</text>'
        )
        parts.append(
            svg_panel(
                result.time_s,
                result.narrowband,
                np.zeros_like(result.boundaries),
                left,
                panel_top,
                plot_w,
                panel_h,
                index == len(results) - 1,
            )
        )
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def write_metrics_csv(path: Path, results: Sequence[PositionResult]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "position",
                "source_file",
                "best_window_start_s",
                "best_window_end_s",
                "retained_real_cycles",
                "median_rr_s",
                "template_quality",
                "adaptive_consistency",
                "adaptive_variation",
                "adaptive_roughness",
                "adaptive_fidelity_to_detail",
                "narrowband_consistency",
                "narrowband_variation",
                "narrowband_roughness",
                "narrowband_fidelity_to_detail",
            )
        )
        for result in results:
            writer.writerow(
                (
                    result.dataset.position,
                    result.dataset.path.name,
                    f"{result.base_window.start_s:.3f}",
                    f"{result.base_window.end_s:.3f}",
                    result.retained_cycles,
                    f"{result.median_rr_s:.6f}",
                    f"{result.template_quality:.6f}",
                    f"{result.adaptive_consistency:.6f}",
                    f"{result.adaptive_variation:.6f}",
                    f"{result.adaptive_roughness:.6f}",
                    f"{result.adaptive_fidelity:.6f}",
                    f"{result.narrowband_consistency:.6f}",
                    f"{result.narrowband_variation:.6f}",
                    f"{result.narrowband_roughness:.6f}",
                    f"{result.narrowband_fidelity:.6f}",
                )
            )


def write_report(path: Path, results: Sequence[PositionResult]) -> None:
    rows = "\n".join(
        f"| {result.dataset.position} | `{result.dataset.path.name}` | "
        f"{result.base_window.start_s:.2f}–{result.base_window.end_s:.2f} | "
        f"{result.retained_cycles} | {result.median_rr_s:.3f} | {result.template_quality:.3f} | "
        f"{result.adaptive_consistency:.3f} | {result.adaptive_variation:.3f} | "
        f"{result.adaptive_roughness:.3f} |"
        for result in results
    )
    text = f"""# 2026-07-16 三位置逐周期自适应心搏增强

## 本轮目标

上一版把同一个固定模板重复十秒，因此每个周期完全一致。本轮首先从每个位置分别搜索后段最优10秒，再比较两种不会固定重复同一模板的方法。

最终推荐使用 `NarrowbandContinuous`：以该窗口真实 RR 中位数为中心，使用中心频率±0.35 Hz的151阶对称FIR做离线零相位窄带平滑。它直接处理连续信号，不复制模板，因此保留自然的周期、幅值和相位变化，同时能显著去除毛刺和额外峰。

另保留 `AdaptiveHeartbeat` 作为对比：每个周期都来自对应真实周期，只保留前两个相位谐波，再使用 `68%稳健模板 + 32%当前真实周期`。它比窄带曲线更规则，但属于逐周期形态增强。

## 每个位置的后段最优窗口

| 位置 | 文件 | 原记录窗口（s） | 真实周期数 | 中位RR（s） | 模板质量 | 增强周期一致性 | 周期间变异度 | 毛刺指标 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Origin 数据

每个位置的 `*_origin_10s.csv` 包含：

- `NarrowbandContinuous`：**推荐主展示**，完全连续、无模板重构的窄带平滑候选；
- `AdaptiveHeartbeat`：逐周期自适应增强对比；
- `CleanHeartNormalized`：App 原连续心搏；
- `HeartDetailNormalized`：1–10 Hz 心搏细节输入；
- `CycleBoundary`、`CycleIndex`、`RR_s`：真实检测周期信息。

Origin 建议将 `Time_s` 设为 X、`NarrowbandContinuous` 设为 Y，X轴 `0–10 s`，Y轴 `-1.2–1.2`。图注建议写明“真实连续信号的RR中心窄带心搏候选；离线零相位平滑，不代表高保真SCG”。

## 能否达到论文效果

从外观上可以接近论文中的平滑 Heartbeat：毛刺和额外小峰显著减少，每个周期仍保留自然的小幅变化，不再完全相同。代价是削弱 SCG 的多峰机械细节。因此它适合作为“心率主周期/论文展示波形”，不能称为高保真 SCG，也不能替代同步 ECG 对准确性的验证。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = [load_dataset(*specification) for specification in DATASETS]
    windows = [best_paperlike_window(dataset, window_s=10.0) for dataset in datasets]
    results = [process_position(dataset, window) for dataset, window in zip(datasets, windows)]

    for result in results:
        prefix = result.dataset.dataset_id
        write_origin_csv(OUTPUT_DIR / f"{prefix}_origin_10s.csv", result)
        write_position_svg(OUTPUT_DIR / f"{prefix}_adaptive_10s.svg", result)
        write_narrowband_position_svg(OUTPUT_DIR / f"{prefix}_best_paperlike_10s.svg", result)
        write_method_comparison_svg(OUTPUT_DIR / f"{prefix}_method_comparison.svg", result)
    write_all_positions_svg(OUTPUT_DIR / "all_positions_best_adaptive_10s.svg", results)
    write_all_positions_narrowband_svg(OUTPUT_DIR / "all_positions_best_paperlike_10s.svg", results)
    write_metrics_csv(OUTPUT_DIR / "position_adaptive_metrics.csv", results)
    write_report(OUTPUT_DIR / "结果说明.md", results)

    for result in results:
        print(
            result.dataset.dataset_id,
            f"window={result.base_window.start_s:.2f}-{result.base_window.end_s:.2f}s",
            f"cycles={result.retained_cycles}",
            f"rr={result.median_rr_s:.3f}s",
            f"quality={result.template_quality:.3f}",
            f"adaptive_consistency={result.adaptive_consistency:.3f}",
            f"adaptive_variation={result.adaptive_variation:.3f}",
            f"adaptive_roughness={result.adaptive_roughness:.3f}",
        )


if __name__ == "__main__":
    main()
