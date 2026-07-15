"""Compare sharper, display-only heartbeat templates without altering BPM.

The accepted robust median template remains the baseline.  Sharper variants
blend it with the medoid: the retained real cycle having the highest average
correlation to the other retained cycles.  Repeating any variant four times is
strictly a morphology display, never an untouched continuous SCG recording.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from display_template_morphology_study import DATASETS, normalized_correlation, rounded, seam_close


BLENDS = (0.0, 0.25, 0.50, 0.75, 1.0)
RECOMMENDED_BLEND = 0.75


def load_templates(input_dir: Path, dataset_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = input_dir / "derived_signals" / f"{dataset_id}_retained_cycles_and_template.csv"
    frame = pd.read_csv(path)
    robust = frame["robust_template"].to_numpy(dtype=float)
    cycle_columns = [key for key in frame.columns if key.startswith("retained_cycle_")]
    cycles = frame[cycle_columns].to_numpy(dtype=float).T
    if len(cycles) < 3:
        raise ValueError(f"{dataset_id}: fewer than three retained cycles")
    return robust, cycles


def medoid_cycle(cycles: np.ndarray) -> tuple[np.ndarray, int, float]:
    count = len(cycles)
    similarities = np.eye(count, dtype=float)
    for first in range(count):
        for second in range(first + 1, count):
            value = abs(normalized_correlation(cycles[first], cycles[second]))
            similarities[first, second] = value
            similarities[second, first] = value
    means = (np.sum(similarities, axis=1) - 1.0) / max(count - 1, 1)
    index = int(np.argmax(means))
    medoid = cycles[index].copy()
    if normalized_correlation(medoid, np.median(cycles, axis=0)) < 0:
        medoid = -medoid
    return medoid, index, float(means[index])


def normalize_template(values: np.ndarray) -> np.ndarray:
    output = seam_close(values)
    scale = max(float(np.max(np.abs(output))), 1e-9)
    return output / scale


def blend_template(robust: np.ndarray, medoid: np.ndarray, blend: float) -> np.ndarray:
    return normalize_template((1.0 - blend) * robust + blend * medoid)


def metrics(values: np.ndarray, cycles: np.ndarray, baseline_slope: float) -> dict:
    consistency = float(np.median([abs(normalized_correlation(cycle, values)) for cycle in cycles]))
    slope95 = float(np.percentile(np.abs(np.diff(values)), 95))
    curvature95 = float(np.percentile(np.abs(np.diff(values, n=2)), 95))
    peaks, _ = signal.find_peaks(values, prominence=0.08)
    troughs, _ = signal.find_peaks(-values, prominence=0.08)
    return {
        "templateConsistency": consistency,
        "slope95": slope95,
        "relativeSharpness": slope95 / max(baseline_slope, 1e-9),
        "curvature95": curvature95,
        "positivePeaks": int(len(peaks)),
        "negativeTroughs": int(len(troughs)),
        "seamDifference": float(abs(values[0] - values[-1])),
    }


def four_cycle(values: np.ndarray) -> np.ndarray:
    return np.tile(values, 4)


def plot_comparison(
    output: Path,
    dataset_id: str,
    bpm: float,
    templates: dict[float, np.ndarray],
    variant_metrics: dict[float, dict],
) -> None:
    phase = np.linspace(0.0, 1.0, len(templates[0.0]), endpoint=False)
    period = 60.0 / bpm
    display_time = np.linspace(0.0, period * 4.0, len(templates[0.0]) * 4, endpoint=False)
    colors = {0.0: "#777777", 0.75: "#1f4e79", 1.0: "#9b3f3f"}
    labels = {
        0.0: "robust median baseline",
        0.75: "75% representative-cycle blend",
        1.0: "100% representative real cycle",
    }
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), constrained_layout=True)
    for blend in (0.0, 0.75, 1.0):
        item = variant_metrics[blend]
        axes[0].plot(
            phase, templates[blend], color=colors[blend], linewidth=1.5,
            label=f"{labels[blend]} · Q={item['templateConsistency']:.3f} · sharp={item['relativeSharpness']:.2f}x",
        )
    axes[0].set_ylabel("One cycle")
    axes[0].set_xlabel("Normalized phase")
    axes[0].legend(loc="upper right")
    for axis, blend in zip(axes[1:], (0.0, 0.75, 1.0)):
        axis.plot(display_time, four_cycle(templates[blend]), color=colors[blend], linewidth=1.25)
        for boundary in np.arange(0.0, period * 4.01, period):
            axis.axvline(boundary, color="#d99a9a", linewidth=0.7, alpha=0.55)
        axis.set_ylabel(labels[blend])
    axes[-1].set_xlabel("Time (s)")
    for axis in axes:
        axis.axhline(0.0, color="#777777", linewidth=0.5)
        axis.grid(alpha=0.16)
    axes[0].set_title(
        f"{dataset_id}: sharper real-cycle morphology variants\n"
        "DISPLAY ONLY · NOT RAW/EXTRACTED SCG · NOT USED FOR BPM"
    )
    fig.savefig(output / f"{dataset_id}_sharpness_comparison.png", dpi=190)
    fig.savefig(output / f"{dataset_id}_sharpness_comparison.svg")
    plt.close(fig)


def plot_standalone(
    output: Path,
    dataset_id: str,
    bpm: float,
    values: np.ndarray,
    blend: float,
    suffix: str,
    label: str,
) -> None:
    period = 60.0 / bpm
    repeated = four_cycle(values)
    time = np.linspace(0.0, period * 4.0, len(repeated), endpoint=False)
    fig, axis = plt.subplots(figsize=(13, 4.2), constrained_layout=True)
    axis.plot(time, repeated, color="#222222", linewidth=1.25, label=label)
    for cycle in range(4):
        axis.add_patch(plt.Rectangle(
            (cycle * period, -1.08), period, 2.16,
            fill=False, edgecolor="#d99a9a", linewidth=0.9, alpha=0.75,
        ))
    axis.axhline(0.0, color="#777777", linewidth=0.55)
    axis.set_xlim(0.0, period * 4.0)
    axis.set_ylim(-1.15, 1.15)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Normalized amplitude")
    axis.set_title(f"Sharper beat-synchronous display: {blend:.0%} representative real-cycle content")
    axis.text(
        0.01, 0.03,
        "DISPLAY-ONLY PERIOD ENHANCEMENT · NOT FOUR UNTOUCHED CONSECUTIVE BEATS · NOT USED FOR BPM",
        transform=axis.transAxes, fontsize=9, color="#9a2020",
    )
    axis.grid(alpha=0.16)
    axis.legend(loc="upper right")
    fig.savefig(output / f"{dataset_id}_{suffix}.png", dpi=200)
    fig.savefig(output / f"{dataset_id}_{suffix}.svg")
    plt.close(fig)


def write_report(output: Path, rows: pd.DataFrame, medoid_rows: list[dict]) -> None:
    latest = rows[rows["dataset"] == "chairback_192603"].set_index("medoidBlend")
    report = f"""# 心搏展示模板峰谷锐化对比实验

## 目的

在已确认符合要求的稳健四周期模板基础上，观察更尖锐的峰谷形态。正式 `cleanHeart`、BPM、睡眠模型和上一轮图片均未修改。

## 方法

上一轮保留了 10 个最一致的真实周期。本轮从中选择“代表性真实周期”（与其余保留周期平均相关性最高的 medoid），分别生成：

- 0%：上一轮逐点中位数稳健模板；
- 25%/50%/75%：稳健模板与代表性真实周期混合；
- 100%：代表性真实周期保形模板。

没有使用人工高斯峰、固定正弦波或目标图片拟合。每个模板只做端点闭合和幅值归一化，然后重复 4 次用于展示。

## 最新坐姿椅背结果

| 代表性周期占比 | 模板一致性 | 相对峰谷锐度 | 正峰/负谷 | 周期接缝差 |
|---:|---:|---:|---:|---:|
| 0% | {latest.loc[0.0, 'templateConsistency']:.3f} | {latest.loc[0.0, 'relativeSharpness']:.2f}× | {int(latest.loc[0.0, 'positivePeaks'])}/{int(latest.loc[0.0, 'negativeTroughs'])} | {latest.loc[0.0, 'seamDifference']:.6f} |
| 50% | {latest.loc[0.5, 'templateConsistency']:.3f} | {latest.loc[0.5, 'relativeSharpness']:.2f}× | {int(latest.loc[0.5, 'positivePeaks'])}/{int(latest.loc[0.5, 'negativeTroughs'])} | {latest.loc[0.5, 'seamDifference']:.6f} |
| 75% | {latest.loc[0.75, 'templateConsistency']:.3f} | {latest.loc[0.75, 'relativeSharpness']:.2f}× | {int(latest.loc[0.75, 'positivePeaks'])}/{int(latest.loc[0.75, 'negativeTroughs'])} | {latest.loc[0.75, 'seamDifference']:.6f} |
| 100% | {latest.loc[1.0, 'templateConsistency']:.3f} | {latest.loc[1.0, 'relativeSharpness']:.2f}× | {int(latest.loc[1.0, 'positivePeaks'])}/{int(latest.loc[1.0, 'negativeTroughs'])} | {latest.loc[1.0, 'seamDifference']:.6f} |

推荐先采用 **75% 代表性真实周期 + 25% 稳健模板**：最新椅背峰谷锐度提高到 {latest.loc[0.75, 'relativeSharpness']:.2f} 倍，同时模板一致性仍为 {latest.loc[0.75, 'templateConsistency']:.3f}。100% 版本更尖锐，可作为对照，但对单次代表周期依赖更强。

## 论文标注

可称为“代表性真实心搏周期保形增强/周期模板展示信号”。不能称为四个未经处理的连续原始心搏，也不能用重复后的波形重新计算 BPM。

## 已保存内容

- `sharpness_variant_metrics.csv`：四个位置、五种锐化比例；
- `derived_signals/*_sharpness_variants.csv`：各模板及四周期数值；
- `*_sharpness_comparison.png/.svg`：0%、75%、100% 同图对比；
- `*_balanced_sharp_four_cycle.png/.svg`：推荐 75% 版本；
- `*_maximum_sharp_four_cycle.png/.svg`：100% 尖锐对照；
- `sharpness_results.json`：代表周期编号、相关性和完整指标。
"""
    (output / "心搏展示模板峰谷锐化对比实验.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/research/display_template_morphology_study"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    derived_dir = args.output / "derived_signals"
    derived_dir.mkdir(parents=True, exist_ok=True)

    previous_metrics = pd.read_csv(args.input / "display_template_metrics.csv").set_index("dataset")
    rows = []
    medoid_rows = []
    detailed = {}
    for item in DATASETS:
        robust, cycles = load_templates(args.input, item.dataset_id)
        medoid, medoid_index, medoid_similarity = medoid_cycle(cycles)
        baseline = normalize_template(robust)
        baseline_slope = float(np.percentile(np.abs(np.diff(baseline)), 95))
        templates = {blend: blend_template(baseline, medoid, blend) for blend in BLENDS}
        variant_metrics = {}
        for blend, template in templates.items():
            item_metrics = metrics(template, cycles, baseline_slope)
            variant_metrics[blend] = item_metrics
            rows.append({
                "dataset": item.dataset_id,
                "medoidBlend": blend,
                "medoidCycleNumber": medoid_index + 1,
                "medoidMeanSimilarity": medoid_similarity,
                **item_metrics,
            })
        bpm = float(previous_metrics.loc[item.dataset_id, "internalReferenceBpm"])
        medoid_rows.append({
            "dataset": item.dataset_id,
            "medoidCycleNumber": medoid_index + 1,
            "medoidMeanSimilarity": medoid_similarity,
        })
        detailed[item.dataset_id] = {
            "sourceRetainedCycles": str(
                args.input / "derived_signals" / f"{item.dataset_id}_retained_cycles_and_template.csv"
            ),
            "internalReferenceBpm": bpm,
            "medoidCycleNumber": medoid_index + 1,
            "medoidMeanSimilarity": medoid_similarity,
            "variants": {str(blend): variant_metrics[blend] for blend in BLENDS},
        }

        phase = np.linspace(0.0, 1.0, len(robust), endpoint=False)
        period = 60.0 / bpm
        template_frame = pd.DataFrame({"phase": phase})
        for blend, template in templates.items():
            template_frame[f"template_blend_{int(round(blend * 100)):03d}"] = template
        template_frame.to_csv(
            derived_dir / f"{item.dataset_id}_sharpness_templates.csv",
            index=False, encoding="utf-8-sig", float_format="%.7f",
        )
        display_frame = pd.DataFrame({
            "display_time_s": np.linspace(0.0, period * 4.0, len(robust) * 4, endpoint=False),
            "cycle_index": np.repeat(np.arange(1, 5), len(robust)),
        })
        for blend, template in templates.items():
            display_frame[f"display_blend_{int(round(blend * 100)):03d}"] = four_cycle(template)
        display_frame.to_csv(
            derived_dir / f"{item.dataset_id}_sharpness_four_cycle_variants.csv",
            index=False, encoding="utf-8-sig", float_format="%.7f",
        )
        plot_comparison(args.output, item.dataset_id, bpm, templates, variant_metrics)
        plot_standalone(
            args.output, item.dataset_id, bpm, templates[RECOMMENDED_BLEND],
            RECOMMENDED_BLEND, "balanced_sharp_four_cycle",
            "75% representative real cycle + 25% robust median",
        )
        plot_standalone(
            args.output, item.dataset_id, bpm, templates[1.0],
            1.0, "maximum_sharp_four_cycle", "representative real medoid cycle",
        )

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(args.output / "sharpness_variant_metrics.csv", index=False, encoding="utf-8-sig")
    result = {
        "purpose": "display-only peak/trough sharpness comparison; formal BPM unchanged",
        "recommendedBlend": RECOMMENDED_BLEND,
        "blendDefinition": "(1-blend)*robustMedian + blend*representativeRealMedoidCycle",
        "datasets": detailed,
        "disclosure": "Each four-cycle trace repeats one derived template; it is not four untouched consecutive beats.",
    }
    (args.output / "sharpness_results.json").write_text(
        json.dumps(rounded(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(args.output, metrics_frame, medoid_rows)
    print(json.dumps(rounded({
        "recommendedBlend": RECOMMENDED_BLEND,
        "medoids": medoid_rows,
        "latestChairback": metrics_frame[metrics_frame["dataset"] == "chairback_192603"].to_dict("records"),
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
