"""Build an auditable, display-only four-cycle heartbeat morphology.

The reconstructed waveform is derived from real detected mechanical cycles,
but it is not a raw/extracted SCG signal and never participates in BPM.  The
script searches one common parameter set on historical human recordings and
saves every intermediate signal, retained cycle, figure and metric.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from reproduce_paper_methods import band_power, finite, longest_continuous_slice, rounded


@dataclass(frozen=True)
class DisplayDataset:
    dataset_id: str
    label: str
    path: Path
    weight: float


DATASETS = (
    DisplayDataset(
        "chairback_192603",
        "latest seated chair-back",
        Path("F:/Users/lik/Documents/xwechat_files/wxid_dkahkwnvrf9422_e9bf/msg/file/2026-07/sensor_20260715_192603.csv"),
        0.40,
    ),
    DisplayDataset(
        "chairback_155119",
        "seated chair-back",
        Path("F:/Users/lik/Documents/xwechat_files/wxid_dkahkwnvrf9422_e9bf/msg/file/2026-07/sensor_20260715_155119.csv"),
        0.30,
    ),
    DisplayDataset(
        "chest_175614",
        "chest held with both hands",
        Path("F:/Users/lik/Documents/xwechat_files/wxid_dkahkwnvrf9422_e9bf/msg/file/2026-07/sensor_20260715_175614.csv"),
        0.20,
    ),
    DisplayDataset(
        "abdomen_195248",
        "abdomen",
        Path("F:/Users/lik/Documents/xwechat_files/wxid_dkahkwnvrf9422_e9bf/msg/file/2026-07/sensor_20260713_195248.csv"),
        0.10,
    ),
)


def read_dataset(item: DisplayDataset) -> dict:
    if not item.path.exists():
        raise FileNotFoundError(item.path)
    wanted = [
        "time_ms",
        "xF",
        "cleanHeart",
        "heartTemplateEnhanced",
        "heartTemplateQuality",
        "heartTemplateCycles",
        "heartTemplateReady",
        "bpm",
    ]
    header = pd.read_csv(item.path, nrows=0).columns.tolist()
    frame = pd.read_csv(item.path, usecols=[key for key in wanted if key in header], low_memory=False)
    time_ms = finite(frame["time_ms"].to_numpy())
    dt = np.diff(time_ms)
    valid_dt = dt[(dt > 0) & (dt < 200)]
    fs = 1000.0 / float(np.median(valid_dt))
    values = {
        key: finite(frame[key].to_numpy())
        for key in frame.columns
        if key != "time_ms"
    }
    values["time_ms"] = time_ms
    values["fs"] = fs
    values["slice"] = longest_continuous_slice(time_ms, fs)
    bpm = values.get("bpm", np.array([]))
    bpm = bpm[(bpm >= 40) & (bpm <= 180)]
    values["reference_bpm"] = float(np.median(bpm)) if len(bpm) else 75.0
    return values


def detail_band(values: np.ndarray, fs: float, highpass: float, lowpass: float, order: int) -> np.ndarray:
    sos = signal.butter(order, [highpass, lowpass], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, values)


def normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = first - np.mean(first)
    b = second - np.mean(second)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-9 else 0.0


def detect_centers(values: np.ndarray, fs: float, reference_bpm: float, start: int, end: int) -> np.ndarray:
    local = values[start:end]
    analytic = np.abs(signal.hilbert(local))
    envelope_sos = signal.butter(2, min(3.0, fs * 0.20), btype="lowpass", fs=fs, output="sos")
    envelope = signal.sosfiltfilt(envelope_sos, analytic)
    expected = fs * 60.0 / reference_bpm
    spread = float(np.median(np.abs(envelope - np.median(envelope))) * 1.4826)
    peaks, _ = signal.find_peaks(
        envelope,
        distance=max(1, int(round(expected * 0.62))),
        prominence=max(spread * 0.25, np.std(envelope) * 0.06, 1e-9),
    )
    refinement = max(1, int(round(expected * 0.16)))
    refined = []
    for peak in peaks:
        left = max(0, peak - refinement)
        right = min(len(local), peak + refinement + 1)
        refined.append(start + left + int(np.argmax(np.abs(local[left:right]))))
    refined.sort()
    deduplicated: list[int] = []
    for center in refined:
        if not deduplicated or center - deduplicated[-1] >= expected * 0.45:
            deduplicated.append(center)
        elif abs(values[center]) > abs(values[deduplicated[-1]]):
            deduplicated[-1] = center
    return np.asarray(deduplicated, dtype=int)


def resample_cycle(values: np.ndarray, left: int, right: int, points: int) -> np.ndarray | None:
    if left < 0 or right > len(values) or right - left < 8:
        return None
    source = signal.detrend(values[left:right], type="linear")
    target = signal.resample(source, points)
    target -= np.mean(target)
    rms = float(np.sqrt(np.mean(target * target)))
    if not np.isfinite(rms) or rms < 1e-9:
        return None
    return target / rms


def shift_and_orient(cycle: np.ndarray, reference: np.ndarray, max_shift: int) -> tuple[np.ndarray, float]:
    best_cycle = cycle.copy()
    best_quality = -2.0
    for shift in range(-max_shift, max_shift + 1):
        shifted = np.roll(cycle, shift)
        quality = normalized_correlation(shifted, reference)
        if quality < 0:
            shifted = -shifted
            quality = -quality
        if quality > best_quality:
            best_cycle = shifted
            best_quality = quality
    return best_cycle, best_quality


def seam_close(values: np.ndarray) -> np.ndarray:
    """Remove only the end-to-end offset so template repeats do not jump."""
    output = values.copy()
    output -= np.linspace(output[0], output[-1], len(output))
    output -= np.mean(output)
    return output


def build_template(
    values: np.ndarray,
    fs: float,
    reference_bpm: float,
    start: int,
    end: int,
    max_shift: int,
    smoothing_points: int,
    template_points: int = 96,
    max_cycles: int = 10,
) -> dict:
    expected = fs * 60.0 / reference_bpm
    target_index = int(round(template_points * 0.30))
    centers = detect_centers(values, fs, reference_bpm, start, end)
    cycles = []
    accepted_centers = []
    pre = int(round(expected * 0.30))
    post = int(round(expected * 0.70))
    for center in centers:
        cycle = resample_cycle(values, center - pre, center + post, template_points)
        if cycle is None:
            continue
        strongest = int(np.argmax(np.abs(cycle)))
        cycle = np.roll(cycle, target_index - strongest)
        if cycle[target_index] < 0:
            cycle = -cycle
        cycles.append(cycle)
        accepted_centers.append(center)
    if len(cycles) < 3:
        return {
            "quality": 0.0,
            "cycles": np.zeros((0, template_points)),
            "allCycles": np.asarray(cycles),
            "template": np.zeros(template_points),
            "centers": np.asarray(accepted_centers, dtype=int),
            "retained": 0,
            "detected": len(cycles),
        }
    aligned = [cycle.copy() for cycle in cycles]
    for _ in range(3):
        reference = np.median(np.asarray(aligned), axis=0)
        aligned = [shift_and_orient(cycle, reference, max_shift)[0] for cycle in aligned]
    preliminary = np.median(np.asarray(aligned), axis=0)
    correlations = np.asarray([normalized_correlation(cycle, preliminary) for cycle in aligned])
    keep_count = min(max_cycles, len(aligned))
    threshold = max(0.25, float(np.quantile(correlations, 0.35)))
    ranked = np.argsort(correlations)[::-1]
    kept_indices = [int(index) for index in ranked if correlations[index] >= threshold][:keep_count]
    if len(kept_indices) < 3:
        kept_indices = [int(index) for index in ranked[: min(3, len(ranked))]]
    retained_cycles = np.asarray([aligned[index] for index in kept_indices])
    template = np.median(retained_cycles, axis=0)
    if smoothing_points >= 5:
        points = smoothing_points if smoothing_points % 2 == 1 else smoothing_points + 1
        template = signal.savgol_filter(template, points, 2, mode="wrap")
    template = seam_close(template)
    strongest = int(np.argmax(np.abs(template)))
    template = np.roll(template, target_index - strongest)
    retained_cycles = np.asarray([np.roll(cycle, target_index - strongest) for cycle in retained_cycles])
    if template[target_index] < 0:
        template = -template
        retained_cycles = -retained_cycles
    template = seam_close(template)
    scale = max(float(np.max(np.abs(template))), 1e-9)
    template /= scale
    retained_cycles /= scale
    quality = float(np.median([abs(normalized_correlation(cycle, template)) for cycle in retained_cycles]))
    return {
        "quality": quality,
        "cycles": retained_cycles,
        "allCycles": np.asarray(aligned),
        "template": template,
        "centers": np.asarray(accepted_centers, dtype=int),
        "retained": len(retained_cycles),
        "detected": len(aligned),
    }


def morphology_metrics(template: np.ndarray) -> dict:
    peaks, _ = signal.find_peaks(template, prominence=0.08)
    troughs, _ = signal.find_peaks(-template, prominence=0.08)
    derivative = np.diff(template)
    return {
        "positivePeaks": int(len(peaks)),
        "negativeTroughs": int(len(troughs)),
        "slope95": float(np.percentile(np.abs(derivative), 95)),
        "peakToPeak": float(np.max(template) - np.min(template)),
    }


def select_four_cycle_window(values: np.ndarray, fs: float, reference_bpm: float, start: int, end: int) -> tuple[int, int]:
    period = max(1, int(round(fs * 60.0 / reference_bpm)))
    width = period * 4
    best = (-2.0, start)
    for index in range(start, max(start + 1, end - width + 1), max(1, int(fs * 0.25))):
        local = values[index : index + width]
        if len(local) < width:
            continue
        score = normalized_correlation(local[:-period], local[period:])
        if score > best[0]:
            best = (score, index)
    return best[1], min(end, best[1] + width)


def normalize_for_plot(values: np.ndarray) -> np.ndarray:
    centered = values - np.median(values)
    scale = np.percentile(np.abs(centered), 95)
    return centered / max(float(scale), 1e-9)


def plot_method_comparison(
    output: Path,
    item: DisplayDataset,
    data: dict,
    detail: np.ndarray,
    result: dict,
) -> None:
    fs = data["fs"]
    start, end, _ = data["slice"]
    warm = min(end, start + int(fs * 15))
    left, right = select_four_cycle_window(detail, fs, data["reference_bpm"], warm, end)
    time = (data["time_ms"][left:right] - data["time_ms"][left]) / 1000.0
    old_enhanced = data.get("heartTemplateEnhanced", np.zeros_like(detail))
    period = 60.0 / data["reference_bpm"]
    repeated = np.tile(result["template"], 4)
    repeated_time = np.linspace(0.0, period * 4.0, len(repeated), endpoint=False)
    phase = np.linspace(0.0, 1.0, len(result["template"]), endpoint=False)

    fig, axes = plt.subplots(5, 1, figsize=(13, 11), constrained_layout=True)
    axes[0].plot(time, normalize_for_plot(data.get("cleanHeart", detail)[left:right]), linewidth=1.0)
    axes[0].set_ylabel("cleanHeart")
    axes[1].plot(time, normalize_for_plot(old_enhanced[left:right]), linewidth=1.0)
    axes[1].set_ylabel("Existing enhanced")
    axes[2].plot(time, normalize_for_plot(detail[left:right]), linewidth=1.0)
    axes[2].set_ylabel("Detail heart band")
    for cycle in result["cycles"]:
        axes[3].plot(phase, cycle, color="#888888", alpha=0.28, linewidth=0.8)
    axes[3].plot(phase, result["template"], color="#1f4e79", linewidth=2.0, label="robust template")
    axes[3].set_ylabel("Aligned cycles")
    axes[3].legend(loc="upper right")
    axes[4].plot(repeated_time, repeated, color="#1f4e79", linewidth=1.5)
    axes[4].set_ylabel("4-cycle display")
    axes[4].set_xlabel("Time (s)")
    for boundary in np.arange(0.0, period * 4.01, period):
        axes[4].axvline(boundary, color="#d99090", linewidth=0.8, alpha=0.6)
    for axis in axes:
        axis.axhline(0.0, color="#777777", linewidth=0.5)
        axis.grid(alpha=0.17)
    axes[0].set_title(
        f"{item.dataset_id}: real-cycle morphology enhancement; display-only, not raw SCG/BPM\n"
        f"retained {result['retained']}/{result['detected']} cycles; template consistency {result['quality']:.2f}"
    )
    fig.savefig(output / f"{item.dataset_id}_display_method_comparison.png", dpi=180)
    fig.savefig(output / f"{item.dataset_id}_display_method_comparison.svg")
    plt.close(fig)


def plot_paper_four_cycles(output: Path, item: DisplayDataset, data: dict, result: dict) -> None:
    period = 60.0 / data["reference_bpm"]
    repeated = np.tile(result["template"], 4)
    repeated_time = np.linspace(0.0, period * 4.0, len(repeated), endpoint=False)
    fig, axis = plt.subplots(figsize=(13, 4.2), constrained_layout=True)
    axis.plot(repeated_time, repeated, color="#222222", linewidth=1.35, label="enhanced heartbeat display")
    for cycle in range(4):
        rectangle = plt.Rectangle(
            (cycle * period, -1.08), period, 2.16,
            fill=False, edgecolor="#d99a9a", linewidth=0.9, alpha=0.75,
        )
        axis.add_patch(rectangle)
    axis.axhline(0.0, color="#777777", linewidth=0.55)
    axis.set_xlim(0.0, period * 4.0)
    axis.set_ylim(-1.15, 1.15)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Normalized amplitude")
    axis.set_title(
        f"Beat-synchronous robust template repeated for 4 cycles ({data['reference_bpm']:.1f} BPM timing)"
    )
    axis.text(
        0.01, 0.03,
        "DISPLAY-ONLY PERIOD ENHANCEMENT · NOT RAW/EXTRACTED SCG · NOT USED FOR BPM",
        transform=axis.transAxes, fontsize=9, color="#9a2020",
    )
    axis.grid(alpha=0.16)
    axis.legend(loc="upper right")
    fig.savefig(output / f"{item.dataset_id}_paper_four_cycle_display.png", dpi=200)
    fig.savefig(output / f"{item.dataset_id}_paper_four_cycle_display.svg")
    plt.close(fig)


def write_report(output: Path, selected: dict, metrics: pd.DataFrame) -> None:
    rows = []
    for _, row in metrics.iterrows():
        rows.append(
            f"| {row['dataset']} | {row['internalReferenceBpm']:.1f} | {int(row['detectedCycles'])} | "
            f"{int(row['retainedCycles'])} | {row['templateConsistency']:.3f} | "
            f"{int(row['positivePeaks'])}/{int(row['negativeTroughs'])} |"
        )
    report = f"""# 论文展示用心搏周期形态增强实验

## 本轮目标

本轮不增加人体门控，不修改正式 BPM，也不把波形当作原始 SCG。目标仅是从真实采集信号中选取真实机械心搏周期，进行同步对齐、异常周期剔除和稳健模板叠加，得到约 4 个清楚、宽度合适的论文展示周期。

## 选中方法

- 从 `xF` 取得 `{selected['highpassHz']}–{selected['lowpassHz']}` Hz 零相位细节通道；
- 使用日志内部 BPM 只确定预期周期范围，不生成心率数值；
- 在真实包络峰附近寻找机械冲击中心，提取周期；
- 周期基线移除、幅值归一化、极性统一和最多 ±{selected['maxShiftPoints']} 点相位对齐；
- 剔除与初始中位模板差异大的周期，保留不超过 10 个周期；
- 使用逐点中位数生成代表性模板，仅做 `{selected['smoothingPoints']}` 点轻度形态平滑；
- 将代表性真实模板重复 4 次作为展示图，并在图中明确标注“非原始波形、不参与 BPM”。

## 结果

| 数据 | 日志内部 BPM（仅定周期宽度） | 检出周期 | 保留周期 | 模板一致性 | 正峰/负谷数 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 如何理解

新图会比连续 `cleanHeart` 更有规律，主要来自三个可解释操作：保留到 {selected['lowpassHz']} Hz 的较快机械细节、剔除不一致周期、将真实周期稳健叠加后重复显示。它适合作为论文中的“心搏同步叠加平均/周期增强展示信号”。

它不能写成“原始心跳波形”或“高保真 SCG”，也不能用重复后的波形重新计算 BPM。论文中应同时给出一小段连续提取波形，以及这张增强模板图，避免读者误以为四个周期是未经处理的连续原始数据。

## 已保存内容

- `display_parameter_search.csv`：全部参数搜索；
- `display_template_metrics.csv`：逐数据模板指标；
- `derived_signals/*_display_template.csv`：全部保留周期、模板及四周期展示数值；
- `*_display_method_comparison.png/.svg`：现有波形、细节通道、周期叠加和新展示波形；
- `*_paper_four_cycle_display.png/.svg`：论文式四周期展示图；
- `display_template_results.json`：机器可读参数和指标。

## 下一步

先确认最新坐姿椅背四周期图片的形态是否满足展示要求。确认后再把同一套“细节通道 + 稳健模板”移植到 App 的周期增强卡片；正式 `cleanHeart`、BPM 和睡眠模型保持不变。
"""
    (output / "论文展示用心搏周期形态增强实验.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    derived_dir = args.output / "derived_signals"
    derived_dir.mkdir(parents=True, exist_ok=True)

    loaded = {item.dataset_id: read_dataset(item) for item in DATASETS}
    search_rows = []
    result_cache: dict[tuple, tuple[np.ndarray, dict]] = {}
    for highpass in (0.6, 0.8, 1.0):
        for lowpass in (6.0, 8.0, 10.0, 12.0):
            for order in (2, 3):
                for max_shift in (2, 4, 6):
                    for smoothing in (0, 5):
                        weighted_score = 0.0
                        minimum_quality = 1.0
                        dataset_results = {}
                        for item in DATASETS:
                            data = loaded[item.dataset_id]
                            start, end, _ = data["slice"]
                            warm = min(end, start + int(data["fs"] * 15))
                            cache_key = (item.dataset_id, highpass, lowpass, order)
                            cached = result_cache.get(cache_key)
                            if cached is None:
                                detail = detail_band(data["xF"], data["fs"], highpass, lowpass, order)
                            else:
                                detail = cached[0]
                            result = build_template(
                                detail, data["fs"], data["reference_bpm"], warm, end,
                                max_shift=max_shift, smoothing_points=smoothing,
                            )
                            result_cache[cache_key] = (detail, result)
                            detail_total = band_power(detail[warm:end], data["fs"], 0.7, 12.0) + 1e-12
                            detail_ratio = band_power(detail[warm:end], data["fs"], 4.0, 12.0) / detail_total
                            coverage = min(result["retained"] / 8.0, 1.0)
                            local_score = 0.78 * result["quality"] + 0.12 * coverage + 0.10 * min(detail_ratio / 0.35, 1.0)
                            weighted_score += item.weight * local_score
                            minimum_quality = min(minimum_quality, result["quality"])
                            dataset_results[item.dataset_id] = {
                                "quality": result["quality"],
                                "detected": result["detected"],
                                "retained": result["retained"],
                                "detailRatio": detail_ratio,
                            }
                        search_rows.append({
                            "highpassHz": highpass,
                            "lowpassHz": lowpass,
                            "filterOrder": order,
                            "maxShiftPoints": max_shift,
                            "smoothingPoints": smoothing,
                            "weightedDisplayScore": weighted_score,
                            "minimumTemplateConsistency": minimum_quality,
                            "datasetResults": json.dumps(rounded(dataset_results), ensure_ascii=False),
                        })
    search_rows.sort(key=lambda row: (row["weightedDisplayScore"], row["minimumTemplateConsistency"]), reverse=True)
    selected = search_rows[0]
    pd.DataFrame(search_rows).to_csv(
        args.output / "display_parameter_search.csv", index=False, encoding="utf-8-sig"
    )

    metric_rows = []
    detailed = {}
    for item in DATASETS:
        data = loaded[item.dataset_id]
        start, end, gaps = data["slice"]
        warm = min(end, start + int(data["fs"] * 15))
        detail = detail_band(
            data["xF"], data["fs"], selected["highpassHz"], selected["lowpassHz"],
            int(selected["filterOrder"]),
        )
        result = build_template(
            detail, data["fs"], data["reference_bpm"], warm, end,
            max_shift=int(selected["maxShiftPoints"]),
            smoothing_points=int(selected["smoothingPoints"]),
        )
        morphology = morphology_metrics(result["template"])
        metric = {
            "dataset": item.dataset_id,
            "label": item.label,
            "internalReferenceBpm": data["reference_bpm"],
            "detectedCycles": result["detected"],
            "retainedCycles": result["retained"],
            "templateConsistency": result["quality"],
            **morphology,
        }
        metric_rows.append(metric)
        detailed[item.dataset_id] = {
            "source": str(item.path),
            "sampleRateHz": data["fs"],
            "gapCount": gaps,
            **metric,
        }

        phase = np.linspace(0.0, 1.0, len(result["template"]), endpoint=False)
        template_frame = pd.DataFrame({"phase": phase, "robust_template": result["template"]})
        for index, cycle in enumerate(result["cycles"], start=1):
            template_frame[f"retained_cycle_{index}"] = cycle
        period = 60.0 / data["reference_bpm"]
        repeated = np.tile(result["template"], 4)
        display_frame = pd.DataFrame({
            "display_time_s": np.linspace(0.0, period * 4.0, len(repeated), endpoint=False),
            "display_value": repeated,
            "display_cycle_index": np.repeat(np.arange(1, 5), len(result["template"])),
        })
        template_frame.to_csv(
            derived_dir / f"{item.dataset_id}_retained_cycles_and_template.csv",
            index=False, encoding="utf-8-sig", float_format="%.7f",
        )
        display_frame.to_csv(
            derived_dir / f"{item.dataset_id}_four_cycle_display.csv",
            index=False, encoding="utf-8-sig", float_format="%.7f",
        )
        pd.DataFrame({
            "time_ms": data["time_ms"].astype(np.int64),
            "time_s": (data["time_ms"] - data["time_ms"][0]) / 1000.0,
            "xF": data["xF"],
            "cleanHeart": data.get("cleanHeart", np.zeros_like(detail)),
            "existingHeartTemplateEnhanced": data.get("heartTemplateEnhanced", np.zeros_like(detail)),
            "detailPreservingHeart": detail,
        }).to_csv(
            derived_dir / f"{item.dataset_id}_continuous_display_inputs.csv",
            index=False, encoding="utf-8-sig", float_format="%.6f",
        )
        plot_method_comparison(args.output, item, data, detail, result)
        plot_paper_four_cycles(args.output, item, data, result)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "display_template_metrics.csv", index=False, encoding="utf-8-sig")
    results = {
        "purpose": "display-only morphology enhancement; never used for BPM",
        "selected": selected,
        "datasets": detailed,
        "disclosure": "Four cycles repeat one robust template derived from retained real cycles; they are not four untouched consecutive raw beats.",
    }
    (args.output / "display_template_results.json").write_text(
        json.dumps(rounded(results), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(args.output, selected, metrics)
    print(json.dumps(rounded({"selected": selected, "metrics": metric_rows}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
