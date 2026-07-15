"""Replay a real-time feasible respiration-cancellation heart candidate.

This study deliberately keeps the Android formal waveform/BPM path unchanged.
It compares three signals on exactly the same historical recordings:

1. the logged current-app ``cleanHeart`` baseline;
2. the previous offline, zero-phase respiration-regression candidate;
3. a causal, fixed-latency adaptive-noise-cancellation candidate.

Every search result, derived signal and figure is written to the requested
output directory so that each iteration remains auditable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from reproduce_paper_methods import (
    candidate_score,
    load_manifest,
    method_metrics,
    read_log,
    rounded,
)
from respiration_cancellation_study import build_candidate


OFFLINE_PARAMETERS = {
    "respCutoffHz": 0.6,
    "regressionWindowSeconds": 12.0,
    "nonlinearBasis": True,
    "heartHighpassHz": 0.8,
    "heartLowpassHz": 6.0,
}


def causal_lowpass(values: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    sos = signal.butter(4, cutoff, btype="lowpass", fs=fs, output="sos")
    return signal.sosfilt(sos, values)


def causal_heart_band(
    values: np.ndarray, fs: float, highpass: float, lowpass: float
) -> np.ndarray:
    """A causal band-pass suitable for later Kotlin biquad implementation."""
    sos = signal.butter(2, [highpass, lowpass], btype="bandpass", fs=fs, output="sos")
    return signal.sosfilt(sos, values)


def nlms_respiration_cancel(
    mixed: np.ndarray,
    respiration_reference: np.ndarray,
    fs: float,
    delay_seconds: float,
    taps: int,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal adaptive cancellation with an explicit fixed output latency.

    At processing sample ``n`` the filter predicts ``mixed[n-delay]`` from the
    current and past causal respiration reference.  The delay makes phase
    alignment possible without reading future samples.  NLMS is used because
    it maps directly to a small real-time FIR adaptive filter.
    """
    delay = max(0, int(round(delay_seconds * fs)))
    residual = np.zeros_like(mixed, dtype=float)
    predicted = np.zeros_like(mixed, dtype=float)
    delayed_mixed = np.zeros_like(mixed, dtype=float)
    delayed_mixed[delay:] = mixed[: len(mixed) - delay] if delay else mixed
    weights = np.zeros(taps, dtype=float)
    final_weights = np.zeros((len(mixed), taps), dtype=float)
    first = max(taps - 1, delay)
    epsilon = max(float(np.var(respiration_reference[: max(first + 1, int(fs * 5))])) * 1e-5, 1e-8)
    for index in range(first, len(mixed)):
        reference_vector = respiration_reference[index - taps + 1 : index + 1][::-1]
        estimate = float(np.dot(weights, reference_vector))
        error = float(delayed_mixed[index] - estimate)
        power = float(np.dot(reference_vector, reference_vector))
        weights += step_size * error * reference_vector / (epsilon + power)
        # Very small leakage prevents unbounded coefficients after a contact change.
        weights *= 0.999999
        predicted[index] = estimate
        residual[index] = error
        final_weights[index] = weights
    return residual, predicted, delayed_mixed, final_weights


def build_causal_candidate(
    mixed: np.ndarray,
    fs: float,
    resp_cutoff: float,
    taps: int,
    step_size: float,
    delay_seconds: float,
    highpass: float,
    lowpass: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    respiration = causal_lowpass(mixed, fs, resp_cutoff)
    residual, predicted, delayed_mixed, weights = nlms_respiration_cancel(
        mixed, respiration, fs, delay_seconds, taps, step_size
    )
    heart = causal_heart_band(residual, fs, highpass, lowpass)
    return heart, respiration, predicted, delayed_mixed, weights


def normalize_for_plot(values: np.ndarray) -> np.ndarray:
    centered = values - np.median(values)
    scale = np.percentile(np.abs(centered), 95)
    return centered / max(float(scale), 1e-9)


def quiet_window(values: np.ndarray, fs: float, start: int, end: int, seconds: float = 12.0) -> tuple[int, int]:
    width = min(end - start, int(round(seconds * fs)))
    if width <= 0:
        return start, end
    choices = []
    for index in range(start, max(start + 1, end - width + 1), max(1, int(fs))):
        local = values[index : index + width]
        choices.append((float(np.std(np.diff(local))), index))
    chosen = min(choices)[1] if choices else start
    return chosen, min(end, chosen + width)


def plot_comparison(
    output: Path,
    dataset_id: str,
    time_ms: np.ndarray,
    delayed_mixed: np.ndarray,
    causal_respiration: np.ndarray,
    causal_prediction: np.ndarray,
    causal_residual: np.ndarray,
    causal_heart: np.ndarray,
    offline_heart: np.ndarray,
    app_clean: np.ndarray,
    start: int,
    end: int,
    fs: float,
) -> None:
    left, right = quiet_window(delayed_mixed, fs, start, end)
    t = (time_ms[left:right] - time_ms[left]) / 1000.0
    panels = [
        (delayed_mixed, "Delayed mixed"),
        (causal_respiration, "Causal respiration"),
        (causal_prediction, "NLMS prediction"),
        (causal_residual, "Subtraction residual"),
        (causal_heart, "Causal candidate"),
        (offline_heart, "Offline candidate"),
        (app_clean, "Current cleanHeart"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 13.5), sharex=True)
    for axis, (values, label) in zip(axes, panels):
        axis.plot(t, normalize_for_plot(values[left:right]), linewidth=0.9)
        axis.axhline(0.0, color="#777", linewidth=0.5)
        axis.set_ylabel(label, fontsize=8.5, labelpad=8)
        axis.grid(alpha=0.18)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{dataset_id}: causal / offline / current comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / f"{dataset_id}_causal_comparison.png", dpi=180)
    fig.savefig(output / f"{dataset_id}_causal_comparison.svg")
    plt.close(fig)


def best_four_cycle_window(
    values: np.ndarray,
    fs: float,
    reference_bpm: float,
    start: int,
    end: int,
) -> tuple[int, int, float]:
    period = max(1, int(round(60.0 * fs / reference_bpm)))
    width = period * 4
    best = (-2.0, start)
    for index in range(start, max(start + 1, end - width + 1), max(1, int(fs * 0.25))):
        local = values[index : index + width]
        correlations = []
        for offset in range(period, len(local), period):
            first = local[: len(local) - offset] - np.mean(local[: len(local) - offset])
            second = local[offset:] - np.mean(local[offset:])
            denominator = np.linalg.norm(first) * np.linalg.norm(second)
            if denominator > 1e-9:
                correlations.append(float(np.dot(first, second) / denominator))
        score = float(np.mean(correlations)) if correlations else -2.0
        if score > best[0]:
            best = (score, index)
    return best[1], min(end, best[1] + width), best[0]


def plot_four_cycle_zoom(
    output: Path,
    dataset_id: str,
    time_ms: np.ndarray,
    causal_heart: np.ndarray,
    offline_heart: np.ndarray,
    app_clean: np.ndarray,
    reference_bpm: float,
    start: int,
    end: int,
    fs: float,
) -> None:
    left, right, score = best_four_cycle_window(
        causal_heart, fs, reference_bpm, start, end
    )
    t = (time_ms[left:right] - time_ms[left]) / 1000.0
    panels = [
        (causal_heart, "Causal candidate"),
        (offline_heart, "Offline zero-phase candidate"),
        (app_clean, "Current app cleanHeart"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for axis, (values, label) in zip(axes, panels):
        axis.plot(t, normalize_for_plot(values[left:right]), linewidth=1.2)
        axis.axhline(0.0, color="#777", linewidth=0.5)
        axis.set_ylabel(label, fontsize=9)
        axis.grid(alpha=0.18)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"{dataset_id}: causal best four-cycle view, ref={reference_bpm:.1f} BPM, repeat={score:.2f}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output / f"{dataset_id}_causal_four_cycle_zoom.png", dpi=180)
    fig.savefig(output / f"{dataset_id}_causal_four_cycle_zoom.svg")
    plt.close(fig)


def dominant_error(metric: dict, reference: float | None) -> float:
    if reference is None or metric["autocorrBpm"] is None:
        return 1.0
    return abs(metric["autocorrBpm"] - reference) / reference


def acceptance(per_dataset: dict[str, dict], human_ids: list[str], desk_ids: list[str], references: dict[str, float | None]) -> dict:
    errors = [dominant_error(per_dataset[key], references[key]) for key in human_ids]
    human_periodicity = [per_dataset[key]["periodicity"] for key in human_ids]
    desk_periodicity = [per_dataset[key]["periodicity"] for key in desk_ids]
    return {
        "maxHumanDominantBpmRelativeError": max(errors),
        "minHumanTemplateQuality": min(per_dataset[key]["templateQuality"] for key in human_ids),
        "minHumanPeriodicity": min(human_periodicity),
        "maxDeskPeriodicity": max(desk_periodicity),
        "accepted": (
            max(errors) <= 0.15
            and min(per_dataset[key]["templateQuality"] for key in human_ids) >= 0.35
            and max(desk_periodicity) < min(human_periodicity)
        ),
    }


def write_report(
    output: Path,
    selected: dict,
    accepted_count: int,
    comparison: pd.DataFrame,
    references: dict[str, float | None],
) -> None:
    humans = [key for key, value in references.items() if value is not None]
    rows = []
    for dataset_id in humans:
        local = comparison[comparison["dataset"] == dataset_id].set_index("method")
        rows.append(
            "| {dataset} | {reference:.1f} | {app:.1f} | {offline:.1f} | {causal:.1f} | {quality:.2f} |".format(
                dataset=dataset_id,
                reference=references[dataset_id],
                app=local.loc["currentAppCleanHeart", "autocorrBpm"],
                offline=local.loc["offlineZeroPhaseCandidate", "autocorrBpm"],
                causal=local.loc["causalNlmsCandidate", "autocorrBpm"],
                quality=local.loc["causalNlmsCandidate", "templateQuality"],
            )
        )
    verdict = (
        "通过了预先定义的跨位置门槛，可进入 App 的 B 观察通道。"
        if selected["accepted"]
        else "没有通过预先定义的跨位置门槛，因此本轮不能替换 App 正式 cleanHeart/BPM。"
    )
    text = f"""# 因果呼吸相减实时可行性实验

## 目的

把上一轮需要未来数据的零相位呼吸建模相减，改写成可实时运行的因果版本，并在同一批胸口、腹部、椅背和三段桌面负对照日志上回放。Android 正式通道未修改。

## 实时方法

- 四阶因果低通产生呼吸参考；
- NLMS 自适应 FIR 根据呼吸参考估计混合信号中的呼吸贡献；
- 使用固定延迟补偿因果低通相位，不读取未来样本；
- 相减后通过二阶因果带通生成心搏候选。

选中参数：呼吸低通 `{selected['respCutoffHz']}` Hz，NLMS `{selected['taps']}` taps，步长 `{selected['stepSize']}`，固定延迟 `{selected['delaySeconds']}` s，心搏带通 `{selected['heartHighpassHz']}–{selected['heartLowpassHz']}` Hz。

## 同一数据结果

| 数据 | 日志内部 BPM 中位数（仅内部参照） | 当前 cleanHeart | 离线零相位候选 | 因果候选 | 因果模板一致性 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 门槛与结论

门槛在搜索前固定：三个有人的位置主周期误差均不超过 15%，最低模板一致性不低于 0.35，并且三段桌面负对照的最高周期性低于有人数据的最低周期性。

- 满足门槛的参数数：**{accepted_count}**；
- 选中参数最大人体主周期相对误差：**{selected['maxHumanDominantBpmRelativeError']:.1%}**；
- 最低人体模板一致性：**{selected['minHumanTemplateQuality']:.3f}**；
- 人体最低周期性 / 桌面最高周期性：**{selected['minHumanPeriodicity']:.3f} / {selected['maxDeskPeriodicity']:.3f}**。

**结论：{verdict}** 即使某一位置的图形更规整，也不能据此宣称已经得到真实 SCG；这里仍是用于心率周期观察的机械心搏候选信号。

## 已保存证据

- `causal_parameter_search.csv`：全部参数和门槛结果；
- `causal_comparison_metrics.csv`：三个方法逐数据指标；
- `derived_signals/*.csv`：逐采样派生信号；
- `*_causal_comparison.png/.svg`：七层完整处理链；
- `*_causal_four_cycle_zoom.png/.svg`：人体数据四周期视图；
- `causal_results.json`：参数、指标和结论的机器可读版本。

## 下一步

只有本轮通过门槛时，才把因果候选加入 App 的 B 观察/日志通道；B 通道先不参与正式 BPM、人体在位和睡眠模型。随后用同一次采集同时保存 A/B，确认椅背场景与桌面负对照后再决定是否替换。
"""
    (output / "因果呼吸相减实时可行性实验.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    derived_dir = args.output / "derived_signals"
    derived_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_manifest(args.manifest)
    loaded = {item.dataset_id: read_log(item) for item in datasets}
    human_ids = [item.dataset_id for item in datasets if item.use == "primary_human"]
    desk_ids = [item.dataset_id for item in datasets if item.use == "negative_control"]
    references: dict[str, float | None] = {}
    for item in datasets:
        bpm = loaded[item.dataset_id].get("bpm", np.array([]))
        bpm = bpm[(bpm >= 40) & (bpm <= 180)]
        references[item.dataset_id] = (
            float(np.median(bpm)) if item.use == "primary_human" and len(bpm) else None
        )

    residual_cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    rows = []
    all_metrics: dict[str, dict[str, dict]] = {}
    for resp_cutoff in (0.4, 0.5, 0.6):
        for taps in (8, 16, 32):
            for step_size in (0.01, 0.03, 0.1):
                for delay_seconds in (0.25, 0.5, 0.75, 1.0):
                    base_key = (resp_cutoff, taps, step_size, delay_seconds)
                    for highpass in (0.7, 0.8):
                        for lowpass in (4.0, 6.0, 8.0):
                            key = (
                                f"resp{resp_cutoff:g}_taps{taps}_mu{step_size:g}_"
                                f"delay{delay_seconds:g}_bp{highpass:g}-{lowpass:g}"
                            )
                            per_dataset = {}
                            for item in datasets:
                                data = loaded[item.dataset_id]
                                cache_key = (item.dataset_id,) + base_key
                                cached = residual_cache.get(cache_key)
                                if cached is None:
                                    respiration = causal_lowpass(data["xF"], data["fs"], resp_cutoff)
                                    cached = nlms_respiration_cancel(
                                        data["xF"], respiration, data["fs"], delay_seconds, taps, step_size
                                    )[:3] + (respiration,)
                                    residual_cache[cache_key] = cached
                                residual = cached[0]
                                heart = causal_heart_band(residual, data["fs"], highpass, lowpass)
                                start, end, _ = data["slice"]
                                warm = min(end, start + int(data["fs"] * 20))
                                per_dataset[item.dataset_id] = method_metrics(
                                    heart[warm:end], data["fs"], references[item.dataset_id]
                                )
                            gate = acceptance(per_dataset, human_ids, desk_ids, references)
                            score = candidate_score(per_dataset, human_ids, desk_ids)
                            all_metrics[key] = per_dataset
                            rows.append({
                                "key": key,
                                "respCutoffHz": resp_cutoff,
                                "taps": taps,
                                "stepSize": step_size,
                                "delaySeconds": delay_seconds,
                                "heartHighpassHz": highpass,
                                "heartLowpassHz": lowpass,
                                "score": score,
                                **gate,
                            })
    rows.sort(key=lambda row: row["score"], reverse=True)
    selected = next((row for row in rows if row["accepted"]), rows[0])
    accepted_count = sum(bool(row["accepted"]) for row in rows)

    comparison_rows = []
    detailed = {}
    for item in datasets:
        data = loaded[item.dataset_id]
        start, end, gaps = data["slice"]
        warm = min(end, start + int(data["fs"] * 20))
        causal_heart, respiration, predicted, delayed_mixed, weights = build_causal_candidate(
            data["xF"], data["fs"], selected["respCutoffHz"], int(selected["taps"]),
            selected["stepSize"], selected["delaySeconds"], selected["heartHighpassHz"],
            selected["heartLowpassHz"],
        )
        causal_residual = delayed_mixed - predicted
        offline_heart, offline_respiration, offline_predicted = build_candidate(
            data["xF"], data["fs"], OFFLINE_PARAMETERS["respCutoffHz"],
            OFFLINE_PARAMETERS["regressionWindowSeconds"], OFFLINE_PARAMETERS["nonlinearBasis"],
            OFFLINE_PARAMETERS["heartHighpassHz"], OFFLINE_PARAMETERS["heartLowpassHz"],
        )
        app_clean = data.get("cleanHeart", np.zeros_like(data["xF"]))
        methods = {
            "currentAppCleanHeart": method_metrics(app_clean[warm:end], data["fs"], references[item.dataset_id]),
            "offlineZeroPhaseCandidate": method_metrics(offline_heart[warm:end], data["fs"], references[item.dataset_id]),
            "causalNlmsCandidate": method_metrics(causal_heart[warm:end], data["fs"], references[item.dataset_id]),
        }
        for method_name, metric in methods.items():
            comparison_rows.append({
                "dataset": item.dataset_id,
                "method": method_name,
                "internalReferenceBpm": metric["internalReferenceBpm"],
                "autocorrBpm": metric["autocorrBpm"],
                "periodicity": metric["periodicity"],
                "referencePeriodicity": metric["referencePeriodicity"],
                "templateQuality": metric["templateQuality"],
                "rrCv": metric["rrCv"],
                "respLeakPercent": metric["respLeakPercent"],
                "detailPercent": metric["detailPercent"],
                "noisePercent": metric["noisePercent"],
            })
        detailed[item.dataset_id] = {
            "source": str(item.path),
            "sampleRateHz": data["fs"],
            "gapCount": gaps,
            "methods": methods,
            "finalAdaptiveWeights": weights[-1].tolist(),
        }
        pd.DataFrame({
            "time_ms": data["time_ms"].astype(np.int64),
            "time_s": (data["time_ms"] - data["time_ms"][0]) / 1000.0,
            "mixed_xF": data["xF"],
            "causal_delayed_mixed": delayed_mixed,
            "causal_respiration_reference": respiration,
            "causal_fitted_respiration": predicted,
            "causal_residual": causal_residual,
            "causal_nlms_heart": causal_heart,
            "offline_respiration_reference": offline_respiration,
            "offline_fitted_respiration": offline_predicted,
            "offline_zero_phase_heart": offline_heart,
            "current_app_cleanHeart": app_clean,
        }).to_csv(
            derived_dir / f"{item.dataset_id}_causal_respiration_cancellation.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
        )
        plot_comparison(
            args.output, item.dataset_id, data["time_ms"], delayed_mixed, respiration,
            predicted, causal_residual, causal_heart, offline_heart, app_clean,
            warm, end, data["fs"],
        )
        if references[item.dataset_id] is not None:
            plot_four_cycle_zoom(
                args.output, item.dataset_id, data["time_ms"], causal_heart,
                offline_heart, app_clean, references[item.dataset_id], warm, end, data["fs"],
            )

    search_frame = pd.DataFrame(rows)
    comparison_frame = pd.DataFrame(comparison_rows)
    search_frame.to_csv(args.output / "causal_parameter_search.csv", index=False, encoding="utf-8-sig")
    comparison_frame.to_csv(args.output / "causal_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    result = {
        "baselineDefinition": {
            "currentApp": "logged cleanHeart: two 0.8 Hz HP + two 4 Hz LP + MA3",
            "offlineUpperBound": OFFLINE_PARAMETERS,
            "internalBpmReference": references,
            "warning": "Log BPM is an internal comparator, not synchronized ECG/PPG truth.",
        },
        "selected": selected,
        "acceptedCandidateCount": accepted_count,
        "topTen": rows[:10],
        "datasets": detailed,
    }
    (args.output / "causal_results.json").write_text(
        json.dumps(rounded(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(args.output, selected, accepted_count, comparison_frame, references)
    print(json.dumps(rounded({
        "selected": selected,
        "acceptedCandidateCount": accepted_count,
        "topFive": rows[:5],
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
