"""Export morphology-constrained, non-identical heartbeat display traces.

The source cycles are taken from ``heartTemplateInput`` and divided by the
real ``cleanPeak`` boundaries.  Eight consecutive cycles are selected for
each sensor position, phase/polarity aligned, and combined with a robust
representative template.  The output keeps one dominant peak, recurrent
small morphology, actual RR timing and restrained beat-to-beat variation.

This is a display-only enhancement.  It is not a raw/extracted SCG waveform
and it never participates in BPM estimation.
"""

from __future__ import annotations

import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from export_origin_periodic_heartbeat import (
    Dataset,
    align_cycle,
    load_dataset,
    normalized_correlation,
)


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

OUTPUT_DIR = Path("docs/research/20260716_morphology_constrained_heartbeat")
VISUALIZATION_DIR = Path(
    r"C:\Users\lik\.codex\visualizations\2026\07\10\019f4bc1-dafe-72f0-ba31-1f73ab952889"
)
PHASE_POINTS = 96
DISPLAY_CYCLES = 8
BASE_WEIGHT = 0.82
REAL_WEIGHT = 1.0 - BASE_WEIGHT


@dataclass
class CycleGroup:
    cycles: List[np.ndarray]
    aligned: List[np.ndarray]
    starts_ms: np.ndarray
    intervals_ms: np.ndarray
    rms_values: np.ndarray
    base: np.ndarray
    robust: np.ndarray
    medoid: np.ndarray
    quality: float
    rr_cv: float
    score: float
    second_peak_ratio: float


@dataclass
class Result:
    dataset: Dataset
    group: CycleGroup
    time_s: np.ndarray
    enhanced: np.ndarray
    fixed: np.ndarray
    real_component: np.ndarray
    boundary: np.ndarray
    cycle_index: np.ndarray
    rr_s: np.ndarray
    amplitude_scale: np.ndarray
    cycle_shapes: np.ndarray
    cycle_correlations: np.ndarray
    second_peak_ratios: np.ndarray


def normalize_rms(values: np.ndarray) -> Tuple[np.ndarray, float]:
    centered = values - float(np.mean(values))
    rms = float(np.sqrt(np.mean(centered * centered)))
    if not math.isfinite(rms) or rms < 1e-9:
        return np.zeros_like(centered), 0.0
    return centered / rms, rms


def normalize_peak(values: np.ndarray) -> np.ndarray:
    output = values - float(np.mean(values))
    scale = float(np.max(np.abs(output)))
    return output / max(scale, 1e-9)


def display_scale(values: np.ndarray, negative_limit: float = 0.68) -> np.ndarray:
    """Normalize the positive main peak and softly compress deep troughs."""
    output = values.copy()
    positive = float(np.max(output))
    output /= max(positive, 1e-9)
    negative = output < 0.0
    output[negative] = -negative_limit * np.tanh((-output[negative]) / negative_limit)
    return output


def circular_smooth(values: np.ndarray, passes: int = 1) -> np.ndarray:
    output = values.copy()
    for _ in range(passes):
        output = (
            np.roll(output, 2)
            + 2.0 * np.roll(output, 1)
            + 3.0 * output
            + 2.0 * np.roll(output, -1)
            + np.roll(output, -2)
        ) / 9.0
    return output


def harmonic_residual(values: np.ndarray, harmonics: int = 7) -> np.ndarray:
    spectrum = np.fft.rfft(values)
    spectrum[harmonics + 1 :] = 0.0
    return np.fft.irfft(spectrum, n=len(values))


def local_maxima(values: np.ndarray) -> List[int]:
    return [
        index
        for index in range(1, len(values) - 1)
        if values[index] > values[index - 1] and values[index] >= values[index + 1]
    ]


def second_positive_peak_ratio(values: np.ndarray) -> float:
    peaks = sorted((float(values[index]), index) for index in local_maxima(values))[::-1]
    positive = [(value, index) for value, index in peaks if value > 0.0]
    if not positive:
        return 1.0
    main_value, main_index = positive[0]
    exclusion = max(4, len(values) // 10)
    secondary = [
        value
        for value, index in positive[1:]
        if abs(index - main_index) > exclusion
    ]
    return max(secondary, default=0.0) / max(main_value, 1e-9)


def orient_and_rotate(values: np.ndarray, target_fraction: float = 0.29) -> Tuple[np.ndarray, int]:
    output = values.copy()
    strongest = int(np.argmax(np.abs(output)))
    if output[strongest] < 0.0:
        output = -output
    main = int(np.argmax(output))
    shift = int(round(len(output) * target_fraction)) - main
    return np.roll(output, shift), shift


def close_seam(values: np.ndarray, target_edge: float | None = None) -> np.ndarray:
    output = values.copy()
    edge = float((output[0] + output[-1]) / 2.0) if target_edge is None else target_edge
    count = 6
    for point in range(count):
        blend = point / count
        output[point] = edge * (1.0 - blend) + output[point] * blend
        output[-point - 1] = edge * (1.0 - blend) + output[-point - 1] * blend
    output[0] = edge
    output[-1] = edge
    return output


def align_cycles(cycles: Sequence[np.ndarray]) -> List[np.ndarray]:
    aligned = [cycle.copy() for cycle in cycles]
    strongest = int(np.argmax(np.abs(aligned[0])))
    if aligned[0][strongest] < 0.0:
        aligned[0] = -aligned[0]
    for _ in range(3):
        reference = circular_smooth(np.median(np.vstack(aligned), axis=0), passes=1)
        aligned = [align_cycle(cycle, reference, maximum_shift=8)[0] for cycle in aligned]
    return aligned


def build_group(
    cycles: Sequence[np.ndarray],
    starts_ms: Sequence[float],
    intervals_ms: Sequence[float],
    rms_values: Sequence[float],
) -> CycleGroup:
    aligned = align_cycles(cycles)
    stack = np.vstack(aligned)
    median_template = np.median(stack, axis=0)
    robust = circular_smooth(median_template, passes=2)
    peer_scores = []
    for first in range(len(aligned)):
        values = [
            abs(normalized_correlation(aligned[first], aligned[second]))
            for second in range(len(aligned))
            if first != second
        ]
        peer_scores.append(float(np.mean(values)))
    medoid = circular_smooth(aligned[int(np.argmax(peer_scores))], passes=2)
    if normalized_correlation(medoid, robust) < 0.0:
        medoid = -medoid
    base = circular_smooth(normalize_peak(0.72 * medoid + 0.28 * robust), passes=1)
    base, shift = orient_and_rotate(base)
    robust = np.roll(robust, shift)
    medoid = np.roll(medoid, shift)
    shifted = [np.roll(cycle, shift) for cycle in aligned]
    shifted = [(-cycle if normalized_correlation(cycle, base) < 0.0 else cycle) for cycle in shifted]
    base = display_scale(close_seam(base))
    base = close_seam(base)
    quality = float(np.median([abs(normalized_correlation(cycle, base)) for cycle in shifted]))
    rr = np.asarray(intervals_ms, dtype=float)
    rr_cv = float(np.std(rr) / max(float(np.mean(rr)), 1e-9))
    second_ratio = second_positive_peak_ratio(base)
    dominance = math.exp(-5.0 * max(0.0, second_ratio - 0.36))
    score = 0.72 * quality + 0.20 * math.exp(-8.0 * rr_cv) + 0.08 * dominance
    return CycleGroup(
        cycles=list(cycles),
        aligned=shifted,
        starts_ms=np.asarray(starts_ms, dtype=float),
        intervals_ms=rr,
        rms_values=np.asarray(rms_values, dtype=float),
        base=base,
        robust=robust,
        medoid=medoid,
        quality=quality,
        rr_cv=rr_cv,
        score=score,
        second_peak_ratio=second_ratio,
    )


def choose_best_group(dataset: Dataset) -> CycleGroup:
    time_ms = dataset.values["time_ms"]
    detail = dataset.values["heartTemplateInput"]
    peaks = time_ms[dataset.values["cleanPeak"] > 0.5]
    if len(peaks) < DISPLAY_CYCLES + 1:
        raise ValueError(f"{dataset.dataset_id}: insufficient cleanPeak boundaries")
    duration_ms = float(time_ms[-1] - time_ms[0])
    search_after = float(time_ms[0] + duration_ms * 0.40)
    phase = np.arange(PHASE_POINTS, dtype=float) / PHASE_POINTS

    candidates: List[CycleGroup] = []
    for begin in range(0, len(peaks) - DISPLAY_CYCLES):
        group_peaks = peaks[begin : begin + DISPLAY_CYCLES + 1]
        if group_peaks[0] < search_after:
            continue
        intervals = np.diff(group_peaks)
        if np.any(intervals < 500.0) or np.any(intervals > 1500.0):
            continue
        cycles: List[np.ndarray] = []
        rms_values: List[float] = []
        valid = True
        for first, second in zip(group_peaks[:-1], group_peaks[1:]):
            sampled = np.interp(first + phase * (second - first), time_ms, detail)
            cycle, rms = normalize_rms(sampled)
            if rms <= 0.0:
                valid = False
                break
            cycles.append(cycle)
            rms_values.append(rms)
        if valid:
            candidates.append(build_group(cycles, group_peaks[:-1], intervals, rms_values))
    if not candidates:
        raise ValueError(f"{dataset.dataset_id}: no eight-cycle group passed RR constraints")
    return max(candidates, key=lambda item: (round(item.score, 7), item.starts_ms[0]))


def constrain_secondary_peaks(values: np.ndarray, target_index: int) -> np.ndarray:
    """Keep secondary positive peaks small without flattening stable shoulders."""
    output = values.copy()
    main = float(output[target_index])
    exclusion = max(6, len(output) // 9)
    cap = 0.42 * main
    for index in local_maxima(output):
        if abs(index - target_index) <= exclusion or output[index] <= cap:
            continue
        excess = float(output[index] - cap)
        width = 4.0
        positions = np.arange(len(output), dtype=float)
        distance = np.minimum(np.abs(positions - index), len(output) - np.abs(positions - index))
        output -= excess * 0.85 * np.exp(-0.5 * (distance / width) ** 2)
    return output


def build_result(dataset: Dataset, group: CycleGroup) -> Result:
    base = group.base.copy()
    main_index = int(np.argmax(base))
    common_edge = float((base[0] + base[-1]) / 2.0)
    median_rms = float(np.median(group.rms_values))
    sample_step_s = 1.0 / dataset.fs_hz

    times: List[float] = []
    enhanced: List[float] = []
    fixed: List[float] = []
    real_component: List[float] = []
    boundaries: List[int] = []
    indices: List[int] = []
    rr_values: List[float] = []
    amplitude_values: List[float] = []
    shapes: List[np.ndarray] = []
    correlations: List[float] = []
    secondary_ratios: List[float] = []
    elapsed = 0.0

    for cycle_index, (cycle, interval_ms, rms) in enumerate(
        zip(group.aligned, group.intervals_ms, group.rms_values)
    ):
        real_shape = normalize_peak(circular_smooth(cycle, passes=1))
        residual = harmonic_residual(real_shape - base, harmonics=7)
        shape = base + REAL_WEIGHT * residual
        shape = constrain_secondary_peaks(shape, main_index)
        shape = display_scale(close_seam(shape, common_edge))
        shape = close_seam(shape, common_edge)
        amplitude = float(np.clip(1.0 + 0.18 * (rms / median_rms - 1.0), 0.92, 1.08))
        shape *= amplitude
        shapes.append(shape.copy())
        correlations.append(abs(normalized_correlation(shape, base)))
        secondary_ratios.append(second_positive_peak_ratio(shape))

        interval_s = float(interval_ms / 1000.0)
        count = max(20, int(round(interval_s / sample_step_s)))
        local_phase = np.arange(count, dtype=float) / count
        phase_axis = np.arange(PHASE_POINTS + 1, dtype=float)
        shape_ext = np.concatenate((shape, shape[:1]))
        base_ext = np.concatenate((base, base[:1]))
        real_ext = np.concatenate((real_shape, real_shape[:1]))
        shape_samples = np.interp(local_phase * PHASE_POINTS, phase_axis, shape_ext)
        base_samples = np.interp(local_phase * PHASE_POINTS, phase_axis, base_ext)
        real_samples = np.interp(local_phase * PHASE_POINTS, phase_axis, real_ext)
        local_times = elapsed + local_phase * interval_s
        times.extend(local_times.tolist())
        enhanced.extend(shape_samples.tolist())
        fixed.extend(base_samples.tolist())
        real_component.extend(real_samples.tolist())
        boundaries.extend(([1] + [0] * (count - 1)))
        indices.extend([cycle_index + 1] * count)
        rr_values.extend([interval_s] * count)
        amplitude_values.extend([amplitude] * count)
        elapsed += interval_s

    # Add the final seam sample so Origin shows the complete eighth cycle.
    times.append(elapsed)
    enhanced.append(float(shapes[-1][0]))
    fixed.append(float(base[0]))
    real_component.append(float(normalize_peak(group.aligned[-1])[0]))
    boundaries.append(1)
    indices.append(DISPLAY_CYCLES)
    rr_values.append(float(group.intervals_ms[-1] / 1000.0))
    amplitude_values.append(amplitude_values[-1])
    return Result(
        dataset=dataset,
        group=group,
        time_s=np.asarray(times),
        enhanced=np.asarray(enhanced),
        fixed=np.asarray(fixed),
        real_component=np.asarray(real_component),
        boundary=np.asarray(boundaries, dtype=int),
        cycle_index=np.asarray(indices, dtype=int),
        rr_s=np.asarray(rr_values),
        amplitude_scale=np.asarray(amplitude_values),
        cycle_shapes=np.vstack(shapes),
        cycle_correlations=np.asarray(correlations),
        second_peak_ratios=np.asarray(secondary_ratios),
    )


def polyline_points(
    times: np.ndarray,
    values: np.ndarray,
    left: float,
    top: float,
    width: float,
    height: float,
    x_max: float,
) -> str:
    xs = left + times / max(x_max, 1e-9) * width
    ys = top + (1.18 - values) / 2.36 * height
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))


def panel_svg(
    result: Result,
    left: float,
    top: float,
    width: float,
    height: float,
    values: np.ndarray,
    title: str,
    subtitle: str,
    color: str = "#2f8cff",
    show_labels: bool = True,
) -> str:
    x_max = float(result.time_s[-1])
    lines = [
        f'<text x="{left:.1f}" y="{top - 44:.1f}" class="panel-title">{html.escape(title)}</text>',
        f'<text x="{left:.1f}" y="{top - 15:.1f}" class="subtitle">{html.escape(subtitle)}</text>',
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" height="{height:.1f}" class="plot"/>',
    ]
    for fraction in (0.25, 0.50, 0.75):
        x = left + width * fraction
        lines.append(f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{top + height:.1f}" class="grid"/>')
    for value in (-0.5, 0.0, 0.5):
        y = top + (1.18 - value) / 2.36 * height
        lines.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" y2="{y:.1f}" class="grid"/>')
    zero_y = top + 0.5 * height
    lines.append(f'<line x1="{left:.1f}" y1="{zero_y:.1f}" x2="{left + width:.1f}" y2="{zero_y:.1f}" class="zero"/>')
    boundary_times = result.time_s[result.boundary > 0]
    for boundary_time in boundary_times:
        x = left + float(boundary_time) / max(x_max, 1e-9) * width
        lines.append(f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{top + height:.1f}" class="boundary"/>')
    points = polyline_points(result.time_s, values, left, top, width, height, x_max)
    lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3.1" stroke-linejoin="round" stroke-linecap="round"/>')
    if show_labels:
        for tick in range(0, int(math.floor(x_max)) + 1):
            x = left + tick / max(x_max, 1e-9) * width
            lines.append(f'<text x="{x:.1f}" y="{top + height + 30:.1f}" class="axis" text-anchor="middle">{tick}</text>')
        lines.append(f'<text x="{left + width / 2:.1f}" y="{top + height + 65:.1f}" class="axis" text-anchor="middle">时间 (s)</text>')
        lines.append(f'<text x="{left - 55:.1f}" y="{top + height / 2:.1f}" class="axis" text-anchor="middle" transform="rotate(-90 {left - 55:.1f} {top + height / 2:.1f})">归一化幅值</text>')
    return "\n".join(lines)


def svg_header(width: int, height: int) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; fill: #19212b; }}
  .title {{ font-size: 31px; font-weight: 700; }}
  .panel-title {{ font-size: 24px; font-weight: 650; }}
  .subtitle {{ font-size: 17px; fill: #5a6573; }}
  .axis {{ font-size: 16px; fill: #66717f; }}
  .warning {{ font-size: 17px; fill: #b33a32; font-weight: 600; }}
  .plot {{ fill: #ffffff; stroke: #ff975f; stroke-width: 1.4; }}
  .grid {{ stroke: #e8edf3; stroke-width: 1; }}
  .zero {{ stroke: #87919d; stroke-width: 1.2; }}
  .boundary {{ stroke: #ff8c42; stroke-width: 1.35; opacity: 0.72; }}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>'''


def write_final_svg(path: Path, result: Result) -> None:
    duration = float(result.time_s[-1])
    median_bpm = 60.0 / float(np.median(result.group.intervals_ms) / 1000.0)
    body = [svg_header(1440, 700)]
    body.append(f'<text x="90" y="58" class="title">{html.escape(result.dataset.position)}：8 个真实心搏周期的形态约束增强</text>')
    subtitle = (
        f"实际时长 {duration:.2f} s；周期中位数 {median_bpm:.1f} BPM；"
        f"模板一致性 {result.group.quality:.3f}；周期并非固定复制"
    )
    body.append(panel_svg(result, 115, 150, 1235, 390, result.enhanced, "一个稳定主峰 + 重复出现的小峰 + 轻微周期间差异", subtitle))
    body.append('<text x="115" y="650" class="warning">展示用形态增强信号：来源为 heartTemplateInput + cleanPeak；非原始/高保真 SCG，不参与 BPM。</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def write_comparison_svg(path: Path, result: Result) -> None:
    body = [svg_header(1440, 1030)]
    body.append(f'<text x="90" y="58" class="title">{html.escape(result.dataset.position)}：固定模板复制与真实周期轻微变化对比</text>')
    body.append(panel_svg(result, 115, 145, 1235, 300, result.fixed, "A. 固定代表模板重复（仅作对照）", "每个周期完全相同，最规整但会掩盖真实差异", color="#7f8aa5"))
    body.append(panel_svg(result, 115, 590, 1235, 300, result.enhanced, "B. 最终结果：模板约束 + 当前真实周期残差", f"模板占 {BASE_WEIGHT:.0%}，真实周期形态占 {REAL_WEIGHT:.0%}；RR 和幅值差异保留", color="#2f8cff"))
    body.append('<text x="115" y="985" class="warning">B 图不是人工按固定 BPM 生成：8 个周期均由该记录中的真实峰值边界和真实周期形态得到。</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def write_all_svg(path: Path, results: Sequence[Result]) -> None:
    body = [svg_header(1500, 1420)]
    body.append('<text x="90" y="60" class="title">胸口、腹部、椅背：各自最优 8 个真实周期的心搏形态增强</text>')
    for index, result in enumerate(results):
        top = 160 + index * 410
        median_bpm = 60.0 / float(np.median(result.group.intervals_ms) / 1000.0)
        subtitle = f"{result.dataset.position}｜{result.time_s[-1]:.2f} s｜{median_bpm:.1f} BPM 周期宽度｜一致性 {result.group.quality:.3f}"
        body.append(panel_svg(result, 130, top, 1280, 260, result.enhanced, f"{index + 1}. {result.dataset.position}", subtitle, show_labels=index == len(results) - 1))
    body.append('<text x="130" y="1375" class="warning">所有曲线均保留一个主峰、稳定小峰与轻微真实变化；用于论文形态展示，不替代原始信号或正式 BPM 通道。</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def write_origin_csv(path: Path, result: Result) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "Time_s",
                "MorphologyEnhanced",
                "FixedRepresentativeTemplate",
                "AlignedRealCycle",
                "CycleBoundary",
                "CycleIndex",
                "RR_s",
                "AmplitudeScale",
                "SourceLabel",
                "BoundaryLabel",
            )
        )
        for index in range(len(result.time_s)):
            writer.writerow(
                (
                    f"{result.time_s[index]:.4f}",
                    f"{result.enhanced[index]:.8g}",
                    f"{result.fixed[index]:.8g}",
                    f"{result.real_component[index]:.8g}",
                    int(result.boundary[index]),
                    int(result.cycle_index[index]),
                    f"{result.rr_s[index]:.4f}",
                    f"{result.amplitude_scale[index]:.5f}",
                    "heartTemplateInput",
                    "cleanPeak",
                )
            )


def write_cycle_csv(path: Path, result: Result) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        fields = ["Phase", "BaseTemplate"] + [f"EnhancedCycle_{index + 1}" for index in range(DISPLAY_CYCLES)]
        writer.writerow(fields)
        for point in range(PHASE_POINTS):
            writer.writerow(
                [f"{point / PHASE_POINTS:.6f}", f"{result.group.base[point]:.8g}"]
                + [f"{result.cycle_shapes[index, point]:.8g}" for index in range(DISPLAY_CYCLES)]
            )


def write_metrics(path: Path, results: Sequence[Result]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "Position",
                "Dataset",
                "Start_s",
                "Duration_s",
                "Median_RR_s",
                "RR_CV",
                "Equivalent_BPM_from_RR",
                "Template_Quality",
                "Median_Cycle_Correlation",
                "Median_SecondPeak_Ratio",
                "Cycle_Variation_RMS",
                "Source_Label",
                "Boundary_Label",
                "Use",
            )
        )
        for result in results:
            start_s = float((result.group.starts_ms[0] - result.dataset.values["time_ms"][0]) / 1000.0)
            median_rr = float(np.median(result.group.intervals_ms) / 1000.0)
            variation = float(np.mean(np.std(result.cycle_shapes, axis=0)))
            writer.writerow(
                (
                    result.dataset.position,
                    result.dataset.path.name,
                    f"{start_s:.3f}",
                    f"{result.time_s[-1]:.3f}",
                    f"{median_rr:.4f}",
                    f"{result.group.rr_cv:.5f}",
                    f"{60.0 / median_rr:.2f}",
                    f"{result.group.quality:.5f}",
                    f"{np.median(result.cycle_correlations):.5f}",
                    f"{np.median(result.second_peak_ratios):.5f}",
                    f"{variation:.5f}",
                    "heartTemplateInput",
                    "cleanPeak",
                    "Display only; not BPM/SCG source",
                )
            )


def write_report(path: Path, results: Sequence[Result]) -> None:
    lines = [
        "# 三位置心搏形态约束增强结果",
        "",
        "## 本轮目标",
        "",
        "在甲方认可的代表性四周期形态基础上，改成每个位置各自选择连续 8 个真实周期。每周期保留一个稳定主峰、重复出现的小峰、真实 RR 宽度和轻微形态/幅值差异，避免把同一个模板机械复制 8 次。",
        "",
        "## 数据标签与算法边界",
        "",
        "- 周期内原始形态来源：`heartTemplateInput`。",
        "- 心搏边界来源：`cleanPeak`。",
        "- 最终列：`MorphologyEnhanced`。",
        "- 固定模板对照列：`FixedRepresentativeTemplate`。",
        "- 真实周期贡献列：`AlignedRealCycle`。",
        "- 该信号只用于论文形态展示，不替代 `cleanHeart`，不参与 BPM，也不能称为原始或高保真 SCG。",
        "",
        "## 处理步骤",
        "",
        "1. 在每份记录后 60% 区间搜索连续 8 个有效 `cleanPeak` 周期。",
        "2. 将每个 `heartTemplateInput` 周期重采样到 96 个相位点，去均值并按 RMS 归一化。",
        "3. 对周期做相位、极性对齐，选取周期一致性与 RR 稳定性综合最优的一组。",
        "4. 用真实周期中值与最具代表性的真实周期建立形态基准；基准自身保留重复出现的小峰。",
        f"5. 每个输出周期由 {BASE_WEIGHT:.0%} 形态基准与 {REAL_WEIGHT:.0%} 当前真实周期残差组成，并保留实际 RR 与压缩后的真实幅值差异。",
        "6. 对非主峰区域只做上限约束，压制偶发大毛刺，不删除稳定小峰。",
        "",
        "## 每个位置的选择结果",
        "",
        "|位置|数据文件|起点 (s)|8 周期时长 (s)|RR 中位数 (s)|等效 BPM|模板一致性|次峰/主峰中位比|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        start_s = float((result.group.starts_ms[0] - result.dataset.values["time_ms"][0]) / 1000.0)
        median_rr = float(np.median(result.group.intervals_ms) / 1000.0)
        lines.append(
            f"|{result.dataset.position}|`{result.dataset.path.name}`|{start_s:.2f}|{result.time_s[-1]:.2f}|{median_rr:.3f}|{60.0 / median_rr:.1f}|{result.group.quality:.3f}|{np.median(result.second_peak_ratios):.3f}|"
        )
    lines.extend(
        [
            "",
            "## Origin 绘图",
            "",
            "导入对应的 `*_origin_morphology_8cycles.csv`，设置 A 列 `Time_s` 为 X、B 列 `MorphologyEnhanced` 为 Y，绘制 Line 图。E 列 `CycleBoundary=1` 是周期边界，可用于加竖线；C、D 列用于审计固定模板与真实周期贡献，不作为最终主曲线。",
            "",
            "## 结论",
            "",
            "本轮实现的是“真实周期约束下的论文展示波形”：规律性比直接连续提取波形强，但不是八个完全相同的复制周期；小峰只保留在多个周期重复出现、且幅度低于主峰的位置。它解决的是展示形态问题，不等于已经获得可做 AO/IC/AC 标注的高保真 SCG。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_inline_html(path: Path, results: Sequence[Result]) -> None:
    cards = []
    colors = ("#2f8cff", "#6857d9", "#0f9f83")
    for result, color in zip(results, colors):
        width, height = 880, 240
        left, top, plot_width, plot_height = 48, 25, 790, 170
        x_max = float(result.time_s[-1])
        points = polyline_points(result.time_s, result.enhanced, left, top, plot_width, plot_height, x_max)
        boundaries = []
        for boundary_time in result.time_s[result.boundary > 0]:
            x = left + float(boundary_time) / max(x_max, 1e-9) * plot_width
            boundaries.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" class="beat"/>')
        median_rr = float(np.median(result.group.intervals_ms) / 1000.0)
        cards.append(
            f'''<article class="mc-card">
  <div class="mc-head"><div><h3>{html.escape(result.dataset.position)}</h3><p>8 个真实周期 · {result.time_s[-1]:.2f} s · {60.0 / median_rr:.1f} BPM 周期宽度</p></div><span>一致性 {result.group.quality:.3f}</span></div>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(result.dataset.position)}形态增强心搏波形">
    <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" class="frame"/>
    <line x1="{left}" y1="{top + plot_height / 2}" x2="{left + plot_width}" y2="{top + plot_height / 2}" class="zero"/>
    {''.join(boundaries)}
    <polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
    <text x="{left}" y="{height - 12}" class="axis">0 s</text><text x="{left + plot_width}" y="{height - 12}" text-anchor="end" class="axis">{x_max:.1f} s</text>
  </svg>
</article>'''
        )
    fragment = f'''<div id="morphology-constrained-heartbeats" class="mc-root">
  <style>
    #morphology-constrained-heartbeats {{ color: var(--codex-color-text-primary, #17202b); font-family: ui-sans-serif, system-ui, "Microsoft YaHei", sans-serif; max-width: 980px; margin: 0 auto; }}
    #morphology-constrained-heartbeats * {{ box-sizing: border-box; }}
    #morphology-constrained-heartbeats .mc-intro {{ margin: 0 0 14px; color: var(--codex-color-text-secondary, #5f6b78); line-height: 1.55; }}
    #morphology-constrained-heartbeats .mc-grid {{ display: grid; gap: 14px; }}
    #morphology-constrained-heartbeats .mc-card {{ background: var(--codex-color-background-secondary, #f7f8fb); border: 1px solid var(--codex-color-border, #dfe4ea); border-radius: 14px; padding: 16px; }}
    #morphology-constrained-heartbeats .mc-head {{ display: flex; align-items: start; justify-content: space-between; gap: 18px; }}
    #morphology-constrained-heartbeats h3 {{ margin: 0; font-size: 18px; }}
    #morphology-constrained-heartbeats p {{ margin: 4px 0 0; color: var(--codex-color-text-secondary, #687380); font-size: 13px; }}
    #morphology-constrained-heartbeats .mc-head span {{ white-space: nowrap; color: var(--codex-color-text-secondary, #687380); font-size: 12px; border: 1px solid var(--codex-color-border, #dfe4ea); border-radius: 99px; padding: 4px 8px; }}
    #morphology-constrained-heartbeats svg {{ display: block; width: 100%; height: auto; margin-top: 8px; }}
    #morphology-constrained-heartbeats .frame {{ fill: var(--codex-color-background-primary, #fff); stroke: #ff975f; stroke-width: 1.2; }}
    #morphology-constrained-heartbeats .zero {{ stroke: var(--codex-color-text-tertiary, #8b95a1); stroke-width: 1; }}
    #morphology-constrained-heartbeats .beat {{ stroke: #ff8c42; stroke-width: 1; opacity: .65; }}
    #morphology-constrained-heartbeats .axis {{ fill: var(--codex-color-text-secondary, #687380); font-size: 13px; }}
    #morphology-constrained-heartbeats .mc-note {{ margin: 14px 0 0; padding: 12px 14px; border-left: 3px solid #c4493f; background: color-mix(in srgb, #c4493f 8%, transparent); color: var(--codex-color-text-secondary, #5f6b78); line-height: 1.55; font-size: 13px; }}
  </style>
  <p class="mc-intro">每个位置都从同一份记录中选取连续 8 个真实心搏周期。主峰由代表性模板约束，重复的小峰被保留，偶发大毛刺被压制，同时保留 RR、幅值与形态的轻微周期间差异。</p>
  <div class="mc-grid">{''.join(cards)}</div>
  <p class="mc-note">展示信号来源：<code>heartTemplateInput</code> + <code>cleanPeak</code>。它只用于论文形态展示，不是原始/高保真 SCG，也不参与 BPM。</p>
</div>'''
    path.write_text(fragment, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Result] = []
    for dataset_id, position, source in DATASETS:
        dataset = load_dataset(dataset_id, position, source)
        group = choose_best_group(dataset)
        result = build_result(dataset, group)
        results.append(result)
        write_origin_csv(OUTPUT_DIR / f"{dataset_id}_origin_morphology_8cycles.csv", result)
        write_cycle_csv(OUTPUT_DIR / f"{dataset_id}_phase_cycles.csv", result)
        write_final_svg(OUTPUT_DIR / f"{dataset_id}_morphology_8cycles.svg", result)
        write_comparison_svg(OUTPUT_DIR / f"{dataset_id}_fixed_vs_varying.svg", result)
    write_all_svg(OUTPUT_DIR / "all_positions_morphology_8cycles.svg", results)
    write_metrics(OUTPUT_DIR / "morphology_metrics.csv", results)
    write_report(OUTPUT_DIR / "结果说明与Origin使用方法.md", results)
    write_inline_html(VISUALIZATION_DIR / "morphology-heartbeats.html", results)
    for result in results:
        median_rr = float(np.median(result.group.intervals_ms) / 1000.0)
        print(
            f"{result.dataset.position}: start={(result.group.starts_ms[0] - result.dataset.values['time_ms'][0]) / 1000.0:.2f}s "
            f"duration={result.time_s[-1]:.2f}s bpm={60.0 / median_rr:.1f} "
            f"quality={result.group.quality:.3f} second={np.median(result.second_peak_ratios):.3f} "
            f"cycle_corr={np.median(result.cycle_correlations):.3f}"
        )


if __name__ == "__main__":
    main()
