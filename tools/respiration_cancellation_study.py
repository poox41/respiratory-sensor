"""Evaluate respiration-model subtraction before heart-band extraction.

This is an offline study. It never changes the Android formal signal path.
Every candidate is evaluated on chest/abdomen/chair-back recordings and on all
desk negative controls from the shared dataset manifest.
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


def zero_phase_filter(values: np.ndarray, fs: float, cutoff, btype: str, order: int = 2) -> np.ndarray:
    sos = signal.butter(order, cutoff, btype=btype, fs=fs, output="sos")
    return signal.sosfiltfilt(sos, values)


def respiration_reference(values: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    return zero_phase_filter(values, fs, cutoff, "lowpass", order=4)


def rolling_respiration_projection(
    mixed: np.ndarray,
    respiration: np.ndarray,
    fs: float,
    window_seconds: float,
    nonlinear: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate breathing contribution with overlap-weighted local regression."""
    derivative = np.gradient(respiration)
    centered_square = respiration * respiration
    centered_square -= np.median(centered_square)
    interaction = respiration * derivative
    basis = [np.ones_like(respiration), respiration, derivative]
    if nonlinear:
        basis.extend([centered_square, interaction])
    design = np.column_stack(basis)

    width = max(int(round(window_seconds * fs)), int(fs * 4))
    hop = max(1, width // 4)
    prediction_sum = np.zeros_like(mixed)
    weight_sum = np.zeros_like(mixed)
    starts = list(range(0, max(1, len(mixed) - width + 1), hop))
    last = max(0, len(mixed) - width)
    if not starts or starts[-1] != last:
        starts.append(last)
    for start in starts:
        end = min(len(mixed), start + width)
        local_x = design[start:end]
        local_y = mixed[start:end]
        if len(local_y) < 8:
            continue
        ridge = np.eye(local_x.shape[1]) * 1e-5
        ridge[0, 0] = 0.0
        coefficients = np.linalg.solve(local_x.T @ local_x + ridge, local_x.T @ local_y)
        estimate = local_x @ coefficients
        weights = np.hanning(len(local_y))
        if not np.any(weights):
            weights = np.ones(len(local_y))
        weights = np.maximum(weights, 0.05)
        prediction_sum[start:end] += estimate * weights
        weight_sum[start:end] += weights
    prediction = prediction_sum / np.maximum(weight_sum, 1e-9)
    return mixed - prediction, prediction


def heart_band(values: np.ndarray, fs: float, highpass: float, lowpass: float) -> np.ndarray:
    return zero_phase_filter(values, fs, [highpass, lowpass], "bandpass", order=2)


def build_candidate(
    mixed: np.ndarray,
    fs: float,
    resp_cutoff: float,
    window_seconds: float,
    nonlinear: bool,
    highpass: float,
    lowpass: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    respiration = respiration_reference(mixed, fs, resp_cutoff)
    residual, predicted = rolling_respiration_projection(
        mixed, respiration, fs, window_seconds, nonlinear
    )
    heart = heart_band(residual, fs, highpass, lowpass)
    return heart, respiration, predicted


def select_display_window(mixed: np.ndarray, fs: float, start: int, end: int, seconds: int = 12) -> tuple[int, int]:
    width = int(fs * seconds)
    candidates = []
    for index in range(start, max(start + 1, end - width), max(1, int(fs))):
        segment = mixed[index : index + width]
        candidates.append((float(np.std(np.diff(segment))), index))
    chosen = min(candidates)[1] if candidates else start
    return chosen, min(end, chosen + width)


def normalize_for_plot(values: np.ndarray) -> np.ndarray:
    centered = values - np.median(values)
    scale = np.percentile(np.abs(centered), 95)
    return centered / max(float(scale), 1e-9)


def plot_comparison(
    output: Path,
    dataset_id: str,
    time_ms: np.ndarray,
    mixed: np.ndarray,
    respiration: np.ndarray,
    predicted: np.ndarray,
    residual: np.ndarray,
    selected_heart: np.ndarray,
    app_clean: np.ndarray,
    start: int,
    end: int,
    fs: float,
) -> None:
    left, right = select_display_window(mixed, fs, start, end)
    t = (time_ms[left:right] - time_ms[left]) / 1000.0
    panels = [
        (mixed, "Mixed xF"),
        (respiration, "Zero-phase respiration estimate"),
        (predicted, "Locally fitted breathing contribution"),
        (residual, "Mixed minus fitted breathing"),
        (selected_heart, "Residual heart-band candidate"),
        (app_clean, "Current app cleanHeart"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 12), sharex=True)
    for axis, (values, label) in zip(axes, panels):
        axis.plot(t, normalize_for_plot(values[left:right]), linewidth=0.9)
        axis.axhline(0.0, color="#777", linewidth=0.5)
        axis.set_ylabel(label, fontsize=9)
        axis.grid(alpha=0.18)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{dataset_id}: respiration modeling and subtraction", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / f"{dataset_id}_respiration_subtraction.png", dpi=180)
    fig.savefig(output / f"{dataset_id}_respiration_subtraction.svg")
    plt.close(fig)


def plot_four_cycle_zoom(
    output: Path,
    dataset_id: str,
    time_ms: np.ndarray,
    mixed: np.ndarray,
    selected_heart: np.ndarray,
    app_clean: np.ndarray,
    reference_bpm: float,
    start: int,
    end: int,
    fs: float,
) -> None:
    period_samples = max(1, int(round(60.0 * fs / reference_bpm)))
    width = period_samples * 4
    best_score = -1.0
    best_start = start
    for index in range(start, max(start + 1, end - width), max(1, int(fs * 0.5))):
        local = selected_heart[index : index + width]
        if len(local) <= period_samples:
            continue
        first = local[:-period_samples] - np.mean(local[:-period_samples])
        second = local[period_samples:] - np.mean(local[period_samples:])
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        score = float(np.dot(first, second) / denominator) if denominator > 1e-9 else -1.0
        if score > best_score:
            best_score = score
            best_start = index
    right = min(end, best_start + width)
    t = (time_ms[best_start:right] - time_ms[best_start]) / 1000.0
    panels = [
        (mixed, "Mixed xF"),
        (selected_heart, "Respiration-subtracted heart candidate"),
        (app_clean, "Current app cleanHeart"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True)
    for axis, (values, label) in zip(axes, panels):
        axis.plot(t, normalize_for_plot(values[best_start:right]), linewidth=1.1)
        axis.axhline(0.0, color="#777", linewidth=0.5)
        axis.set_ylabel(label, fontsize=9)
        axis.grid(alpha=0.18)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"{dataset_id}: best four-cycle view at internal reference {reference_bpm:.1f} BPM",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output / f"{dataset_id}_four_cycle_zoom.png", dpi=180)
    fig.savefig(output / f"{dataset_id}_four_cycle_zoom.svg")
    plt.close(fig)


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
        values = loaded[item.dataset_id].get("bpm", np.array([]))
        values = values[(values >= 40) & (values <= 180)]
        references[item.dataset_id] = (
            float(np.median(values)) if item.use == "primary_human" and len(values) else None
        )

    grid = []
    for resp_cutoff in (0.35, 0.4, 0.5, 0.6):
        for window_seconds in (8.0, 12.0, 20.0, 30.0):
            for nonlinear in (False, True):
                for highpass in (0.6, 0.7, 0.8):
                    for lowpass in (4.0, 6.0, 8.0, 10.0):
                        grid.append((resp_cutoff, window_seconds, nonlinear, highpass, lowpass))

    rows = []
    all_metrics: dict[str, dict[str, dict]] = {}
    residual_cache: dict[tuple, np.ndarray] = {}
    for resp_cutoff, window_seconds, nonlinear, highpass, lowpass in grid:
        key = (
            f"resp{resp_cutoff:g}_win{window_seconds:g}_"
            f"{'nonlinear' if nonlinear else 'linear'}_bp{highpass:g}-{lowpass:g}"
        )
        per_dataset = {}
        for item in datasets:
            data = loaded[item.dataset_id]
            start, end, _ = data["slice"]
            warm = min(end, start + int(data["fs"] * 15))
            cache_key = (item.dataset_id, resp_cutoff, window_seconds, nonlinear)
            residual = residual_cache.get(cache_key)
            if residual is None:
                respiration = respiration_reference(data["xF"], data["fs"], resp_cutoff)
                residual, _ = rolling_respiration_projection(
                    data["xF"], respiration, data["fs"], window_seconds, nonlinear
                )
                residual_cache[cache_key] = residual
            heart = heart_band(residual, data["fs"], highpass, lowpass)
            per_dataset[item.dataset_id] = method_metrics(
                heart[warm:end], data["fs"], references[item.dataset_id]
            )
        score = candidate_score(per_dataset, human_ids, desk_ids)
        max_desk_periodicity = max(per_dataset[key_]["periodicity"] for key_ in desk_ids)
        min_human_template = min(per_dataset[key_]["templateQuality"] for key_ in human_ids)
        dominant_errors = []
        for key_ in human_ids:
            metric = per_dataset[key_]
            dominant_errors.append(
                abs(metric["autocorrBpm"] - references[key_]) / references[key_]
                if metric["autocorrBpm"] is not None else 1.0
            )
        accepted = (
            max(dominant_errors) <= 0.15
            and min_human_template >= 0.35
            and max_desk_periodicity < min(per_dataset[key_]["periodicity"] for key_ in human_ids)
        )
        all_metrics[key] = per_dataset
        rows.append({
            "key": key,
            "respCutoffHz": resp_cutoff,
            "regressionWindowSeconds": window_seconds,
            "nonlinearBasis": nonlinear,
            "heartHighpassHz": highpass,
            "heartLowpassHz": lowpass,
            "score": score,
            "accepted": accepted,
            "maxHumanDominantBpmRelativeError": max(dominant_errors),
            "minHumanTemplateQuality": min_human_template,
            "maxDeskPeriodicity": max_desk_periodicity,
        })
    rows.sort(key=lambda row: row["score"], reverse=True)
    selected = next((row for row in rows if row["accepted"]), rows[0])
    selected_key = selected["key"]

    comparison_rows = []
    detailed = {}
    for item in datasets:
        data = loaded[item.dataset_id]
        start, end, gaps = data["slice"]
        warm = min(end, start + int(data["fs"] * 15))
        selected_heart, respiration, predicted = build_candidate(
            data["xF"], data["fs"], selected["respCutoffHz"],
            selected["regressionWindowSeconds"], selected["nonlinearBasis"],
            selected["heartHighpassHz"], selected["heartLowpassHz"],
        )
        residual = data["xF"] - predicted
        direct_subtract = data["xF"] - respiration
        direct_heart = heart_band(
            direct_subtract, data["fs"], selected["heartHighpassHz"], selected["heartLowpassHz"]
        )
        app_clean = data.get("cleanHeart", np.zeros_like(data["xF"]))
        methods = {
            "directRespSubtract": method_metrics(
                direct_heart[warm:end], data["fs"], references[item.dataset_id]
            ),
            "rollingProjectionSubtract": method_metrics(
                selected_heart[warm:end], data["fs"], references[item.dataset_id]
            ),
            "currentAppCleanHeart": method_metrics(
                app_clean[warm:end], data["fs"], references[item.dataset_id]
            ),
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
        }
        pd.DataFrame({
            "time_ms": data["time_ms"].astype(np.int64),
            "time_s": (data["time_ms"] - data["time_ms"][0]) / 1000.0,
            "mixed_xF": data["xF"],
            "respiration_reference": respiration,
            "fitted_breathing_contribution": predicted,
            "mixed_minus_fitted_breathing": residual,
            "direct_resp_subtract_heart": direct_heart,
            "rolling_projection_heart": selected_heart,
            "current_app_cleanHeart": app_clean,
        }).to_csv(
            derived_dir / f"{item.dataset_id}_respiration_cancellation.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
        )
        plot_comparison(
            args.output, item.dataset_id, data["time_ms"], data["xF"],
            respiration, predicted, residual, selected_heart, app_clean,
            warm, end, data["fs"],
        )
        if references[item.dataset_id] is not None:
            plot_four_cycle_zoom(
                args.output, item.dataset_id, data["time_ms"], data["xF"],
                selected_heart, app_clean, references[item.dataset_id],
                warm, end, data["fs"],
            )

    pd.DataFrame(rows).to_csv(args.output / "respiration_cancellation_parameter_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparison_rows).to_csv(args.output / "respiration_cancellation_metrics.csv", index=False, encoding="utf-8-sig")
    result = {
        "baselineDefinition": {
            "appWaveform": "cleanHeart: two 0.8 Hz HP biquads + two 4 Hz LP biquads + MA3",
            "internalBpmReference": references,
            "warning": "The log BPM median is an internal comparator, not ECG/PPG ground truth.",
        },
        "selected": selected,
        "acceptedCandidateCount": sum(bool(row["accepted"]) for row in rows),
        "topTen": rows[:10],
        "datasets": detailed,
    }
    (args.output / "respiration_cancellation_results.json").write_text(
        json.dumps(rounded(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(rounded({
        "selected": selected,
        "acceptedCandidateCount": result["acceptedCandidateCount"],
        "topFive": rows[:5],
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
