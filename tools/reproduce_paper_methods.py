"""Reproduce the three papers' signal-processing ideas on historical logs.

The two vehicle papers are documented as behavior-classification methods and
are therefore not forced into the vital-sign pipeline. This script focuses on
the reproducible Chapter-3 methods in Wang's thesis and compares them with the
signals already logged by the Android app.
"""

from __future__ import annotations

import argparse
import csv
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


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    label: str
    placement: str
    path: Path
    use: str


def finite(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def load_manifest(path: Path) -> list[Dataset]:
    rows: list[Dataset] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            source = Path(row["source_file"])
            if not source.exists():
                raise FileNotFoundError(source)
            rows.append(Dataset(row["id"], row["label"], row["placement"], source, row["use"]))
    return rows


def longest_continuous_slice(time_ms: np.ndarray, fs: float) -> tuple[int, int, int]:
    expected_ms = 1000.0 / fs
    cuts = np.flatnonzero(np.diff(time_ms) > expected_ms * 1.8) + 1
    edges = np.concatenate(([0], cuts, [len(time_ms)]))
    spans = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]
    start, end = max(spans, key=lambda pair: pair[1] - pair[0])
    return start, end, len(cuts)


def read_log(dataset: Dataset) -> dict:
    wanted = ["time_ms", "rawX", "xF", "resp", "hrRaw", "cleanHeart", "bpm", "rpm"]
    header = pd.read_csv(dataset.path, nrows=0).columns.tolist()
    available = [name for name in wanted if name in header]
    frame = pd.read_csv(dataset.path, usecols=available, low_memory=False)
    time_ms = finite(frame["time_ms"].to_numpy())
    dt = np.diff(time_ms)
    valid_dt = dt[(dt > 0) & (dt < 200)]
    fs = 1000.0 / float(np.median(valid_dt))
    result = {name: finite(frame[name].to_numpy()) for name in available if name != "time_ms"}
    result["time_ms"] = time_ms
    result["fs"] = fs
    result["slice"] = longest_continuous_slice(time_ms, fs)
    return result


def causal_moving_average(values: np.ndarray, points: int) -> np.ndarray:
    return signal.lfilter(np.ones(points, dtype=float) / points, [1.0], values)


def thesis_aligned_difference(values: np.ndarray, points: int, delay: int) -> np.ndarray:
    """Delay the raw signal to the center of a causal MA, then subtract MA."""
    average = causal_moving_average(values, points)
    delayed_raw = np.zeros_like(values)
    if delay == 0:
        delayed_raw[:] = values
    else:
        delayed_raw[delay:] = values[:-delay]
    return delayed_raw - average


def app_direction_difference(values: np.ndarray, points: int = 10, delay: int = 5) -> np.ndarray:
    """The timing direction used by the currently restored Kotlin class."""
    average = causal_moving_average(values, points)
    delayed_average = np.zeros_like(values)
    delayed_average[delay:] = average[:-delay]
    return values - delayed_average


def lowpass(values: np.ndarray, cutoff: float | None, fs: float, order: int = 2) -> np.ndarray:
    if cutoff is None:
        return values.copy()
    sos = signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    return signal.sosfilt(sos, values)


def wang_respiration(values: np.ndarray, fs: float) -> np.ndarray:
    # 633 taps reproduces the documented long FIR closely. Kaiser beta 7.86 is
    # the usual approximation for ~80 dB stop-band attenuation.
    taps = signal.firwin(633, cutoff=0.5, window=("kaiser", 7.86), fs=fs)
    return signal.lfilter(taps, [1.0], values)


def wang_narrow_fir(values: np.ndarray, fs: float) -> np.ndarray:
    taps = signal.firwin(633, [0.7, 2.0], pass_zero=False, window=("kaiser", 7.86), fs=fs)
    return signal.lfilter(taps, [1.0], values)


def band_power(values: np.ndarray, fs: float, low: float, high: float) -> float:
    if len(values) < int(fs * 8):
        return 0.0
    freq, psd = signal.welch(values, fs=fs, nperseg=min(len(values), int(fs * 16)))
    mask = (freq >= low) & (freq < high)
    return float(np.trapz(psd[mask], freq[mask])) if np.any(mask) else 0.0


def autocorrelation_bpm(values: np.ndarray, fs: float) -> tuple[float | None, float]:
    centered = values - np.median(values)
    scale = float(np.sqrt(np.mean(centered * centered)))
    if scale < 1e-9:
        return None, 0.0
    centered /= scale
    min_lag, max_lag = int(round(fs * 0.5)), int(round(fs * 1.5))
    corr = signal.fftconvolve(centered, centered[::-1], mode="full")[len(centered) - 1 :]
    normalization = np.arange(len(centered), 0, -1, dtype=float)
    corr /= normalization
    corr /= max(float(corr[0]), 1e-9)
    local = corr[min_lag : max_lag + 1]
    if not len(local):
        return None, 0.0
    peaks, _ = signal.find_peaks(local, distance=max(1, int(fs * 0.25)))
    index = int(peaks[np.argmax(local[peaks])]) if len(peaks) else int(np.argmax(local))
    lag = min_lag + index
    return 60.0 * fs / lag, float(np.clip(local[index], 0.0, 1.0))


def reference_periodicity(values: np.ndarray, fs: float, reference_bpm: float | None) -> tuple[float | None, float | None, float | None]:
    """Measure support near an internal BPM reference without calling it truth."""
    if reference_bpm is None or not 40 <= reference_bpm <= 120:
        return None, None, None
    centered = values - np.median(values)
    rms = float(np.sqrt(np.mean(centered * centered)))
    if rms < 1e-9:
        return None, 0.0, None
    centered /= rms
    corr = signal.fftconvolve(centered, centered[::-1], mode="full")[len(centered) - 1 :]
    corr /= np.arange(len(centered), 0, -1, dtype=float)
    corr /= max(float(corr[0]), 1e-9)
    expected_lag = 60.0 * fs / reference_bpm
    # This is support at the internal reference period, not another broad peak
    # search. A broad ±20% search can simply land on a respiratory harmonic.
    center = int(round(expected_lag))
    low = max(1, center - 2)
    high = min(len(corr) - 1, center + 2)
    local = corr[low : high + 1]
    if not len(local):
        return None, 0.0, None
    peaks, _ = signal.find_peaks(local)
    index = int(peaks[np.argmax(local[peaks])]) if len(peaks) else int(np.argmax(local))
    lag = low + index
    bpm = 60.0 * fs / lag
    quality = float(np.clip(local[index], 0.0, 1.0))
    error = abs(bpm - reference_bpm) / reference_bpm
    return bpm, quality, error


def window_stability(values: np.ndarray, fs: float, seconds: int = 20) -> tuple[list[float], float | None]:
    width = int(round(fs * seconds))
    bpms: list[float] = []
    for start in range(0, len(values) - width + 1, width):
        bpm, quality = autocorrelation_bpm(values[start : start + width], fs)
        if bpm is not None and 40 <= bpm <= 120 and quality >= 0.08:
            bpms.append(float(bpm))
    if len(bpms) < 2:
        return bpms, None
    return bpms, float(np.std(bpms) / max(np.mean(bpms), 1e-9))


def beat_template_metrics(values: np.ndarray, fs: float) -> tuple[float, float, int, np.ndarray, np.ndarray]:
    centered = values - np.median(values)
    analytic = np.abs(signal.hilbert(centered))
    envelope = lowpass(analytic, min(2.5, fs * 0.2), fs, order=2)
    floor = float(np.median(envelope))
    spread = float(np.median(np.abs(envelope - floor))) * 1.4826
    peaks, _ = signal.find_peaks(
        envelope,
        distance=max(1, int(fs * 0.5)),
        prominence=max(spread * 0.35, np.std(envelope) * 0.08, 1e-9),
    )
    if len(peaks) < 4:
        return 0.0, 0.0, 0, np.zeros(64), peaks
    rr = np.diff(peaks) / fs
    valid_rr = rr[(rr >= 0.5) & (rr <= 1.5)]
    rr_cv = float(np.std(valid_rr) / np.mean(valid_rr)) if len(valid_rr) >= 3 else 1.0
    cycles = []
    for first, second in zip(peaks[:-1], peaks[1:]):
        duration = (second - first) / fs
        if not 0.5 <= duration <= 1.5 or second - first < 5:
            continue
        source = centered[first : second + 1]
        target = np.interp(np.linspace(0, len(source) - 1, 64), np.arange(len(source)), source)
        target -= np.mean(target)
        rms = float(np.sqrt(np.mean(target * target)))
        if rms > 1e-9:
            cycles.append(target / rms)
    if len(cycles) < 3:
        return 0.0, rr_cv, len(cycles), np.zeros(64), peaks
    matrix = np.asarray(cycles)
    template = np.median(matrix, axis=0)
    quality_values = []
    for cycle in matrix:
        coefficient = np.corrcoef(cycle, template)[0, 1]
        if np.isfinite(coefficient):
            quality_values.append(abs(float(coefficient)))
    quality = float(np.median(quality_values)) if quality_values else 0.0
    template /= max(float(np.max(np.abs(template))), 1e-9)
    return quality, rr_cv, len(cycles), template, peaks


def method_metrics(values: np.ndarray, fs: float, reference_bpm: float | None = None) -> dict:
    values = finite(values)
    total = band_power(values, fs, 0.08, 15.0) + 1e-12
    resp = band_power(values, fs, 0.08, 0.6) / total
    heart = band_power(values, fs, 0.7, 4.0) / total
    detail = band_power(values, fs, 4.0, 10.0) / total
    noise = band_power(values, fs, 10.0, min(20.0, fs / 2 - 0.1)) / total
    bpm, periodicity = autocorrelation_bpm(values, fs)
    reference_peak_bpm, reference_quality, reference_error = reference_periodicity(values, fs, reference_bpm)
    window_bpms, window_cv = window_stability(values, fs)
    template_quality, rr_cv, cycles, template, peaks = beat_template_metrics(values, fs)
    return {
        "rms": float(np.sqrt(np.mean(values * values))),
        "respLeakPercent": resp * 100.0,
        "heartBandPercent": heart * 100.0,
        "detailPercent": detail * 100.0,
        "noisePercent": noise * 100.0,
        "autocorrBpm": bpm,
        "periodicity": periodicity,
        "internalReferenceBpm": reference_bpm,
        "referencePeakBpm": reference_peak_bpm,
        "referencePeriodicity": reference_quality,
        "referenceRelativeError": reference_error,
        "windowBpms": window_bpms,
        "windowBpmCv": window_cv,
        "templateQuality": template_quality,
        "rrCv": rr_cv,
        "cycles": cycles,
        "template": template.tolist(),
        "peakIndices": peaks.tolist(),
    }


def candidate_score(per_dataset: dict[str, dict], human_ids: list[str], desk_ids: list[str]) -> float:
    human = [per_dataset[key] for key in human_ids]
    periodicity = min(item["periodicity"] for item in human)
    template = min(item["templateQuality"] for item in human)
    leakage = max(item["respLeakPercent"] for item in human) / 100.0
    noise = max(item["noisePercent"] for item in human) / 100.0
    cvs = [item["windowBpmCv"] if item["windowBpmCv"] is not None else 0.5 for item in human]
    stability = max(cvs)
    physiological = sum(
        item["autocorrBpm"] is not None and 40 <= item["autocorrBpm"] <= 120 for item in human
    ) / len(human)
    human_rms = max(np.median([x["rms"] for x in human]), 1e-9)
    desk_penalty = max(
        per_dataset[desk_id]["periodicity"] * min(1.0, per_dataset[desk_id]["rms"] / human_rms)
        for desk_id in desk_ids
    )
    reference_quality = min(item["referencePeriodicity"] or 0.0 for item in human)
    reference_error = max(item["referenceRelativeError"] if item["referenceRelativeError"] is not None else 1.0 for item in human)
    dominant_error = max(
        abs(item["autocorrBpm"] - item["internalReferenceBpm"]) / item["internalReferenceBpm"]
        if item["autocorrBpm"] is not None and item["internalReferenceBpm"] is not None else 1.0
        for item in human
    )
    detail = min(item["detailPercent"] / 25.0 for item in human)
    return float(
        0.12 * periodicity
        + 0.15 * template
        + 0.15 * (1.0 - min(stability, 1.0))
        + 0.08 * physiological
        + 0.30 * reference_quality
        + 0.10 * min(detail, 1.0)
        - 0.25 * min(reference_error, 1.0)
        - 0.20 * min(dominant_error, 1.0)
        - 0.15 * min(leakage, 1.0)
        - 0.10 * min(noise, 1.0)
        - 0.10 * desk_penalty
    )


def plot_overview(output: Path, dataset: Dataset, t: np.ndarray, series: dict[str, np.ndarray], start: int, fs: float) -> None:
    width = int(fs * 12)
    candidates = range(start, max(start + 1, len(t) - width), max(1, int(fs)))
    chosen = min(candidates, key=lambda i: np.std(np.diff(series["mixed"][i : i + width])))
    end = min(len(t), chosen + width)
    rel = (t[chosen:end] - t[chosen]) / 1000.0
    panels = [
        ("mixed", "Mixed mechanical signal xF"),
        ("wang_resp", "Wang respiration FIR (delay not removed)"),
        ("app_diff", "Current app-direction difference"),
        ("wang_diff", "Wang center-aligned difference (10 points)"),
        ("tuned", "Selected aligned difference"),
        ("app_clean", "Current app cleanHeart"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 12), sharex=True)
    for axis, (key, title) in zip(axes, panels):
        values = series[key][chosen:end]
        scale = np.percentile(np.abs(values - np.median(values)), 95)
        normalized = (values - np.median(values)) / max(scale, 1e-9)
        axis.plot(rel, normalized, linewidth=0.9)
        axis.axhline(0, color="#888", linewidth=0.5)
        axis.set_ylabel(title, fontsize=9)
        axis.grid(alpha=0.18)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{dataset.dataset_id}: same-data method comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / f"{dataset.dataset_id}_method_overview.png", dpi=180)
    fig.savefig(output / f"{dataset.dataset_id}_method_overview.svg")
    plt.close(fig)


def plot_templates(output: Path, dataset: Dataset, metrics: dict[str, dict]) -> None:
    methods = ["app_clean", "app_diff", "wang_fir", "wang_diff", "tuned"]
    fig, axes = plt.subplots(1, len(methods), figsize=(16, 3.5), sharey=True)
    x = np.linspace(0, 1, 64, endpoint=False)
    for axis, key in zip(axes, methods):
        item = metrics[key]
        axis.plot(x, item["template"], linewidth=1.6)
        axis.axhline(0, color="#888", linewidth=0.5)
        axis.set_title(
            f"{key}\nQ={item['templateQuality']:.2f}, leak={item['respLeakPercent']:.1f}%",
            fontsize=9,
        )
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Normalized amplitude")
    fig.suptitle(f"{dataset.dataset_id}: beat-synchronous representative cycles (not raw waveform)")
    fig.tight_layout()
    fig.savefig(output / f"{dataset.dataset_id}_templates.png", dpi=180)
    fig.savefig(output / f"{dataset.dataset_id}_templates.svg")
    plt.close(fig)


def rounded(value):
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 6)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = load_manifest(args.manifest)
    loaded = {item.dataset_id: read_log(item) for item in datasets}
    humans = [item.dataset_id for item in datasets if item.use == "primary_human"]
    desks = [item.dataset_id for item in datasets if item.use == "negative_control"]
    internal_references: dict[str, float | None] = {}
    for item in datasets:
        bpm_values = loaded[item.dataset_id].get("bpm", np.array([]))
        bpm_values = bpm_values[(bpm_values >= 40) & (bpm_values <= 180)]
        internal_references[item.dataset_id] = (
            float(np.median(bpm_values)) if item.use == "primary_human" and len(bpm_values) else None
        )

    # Search only the Wang-aligned-difference family. Vehicle-paper CNN/LSTM
    # methods have incompatible labels and are intentionally not fabricated.
    candidates = []
    for points in (6, 8, 10, 12, 14, 16, 20):
        center = int(round((points - 1) / 2))
        for offset in (-2, -1, 0, 1, 2):
            delay = max(0, center + offset)
            for cutoff in (None, 4.0, 6.0, 8.0, 10.0, 12.0):
                candidates.append((points, delay, cutoff))

    search_rows = []
    candidate_results: dict[str, dict[str, dict]] = {}
    for points, delay, cutoff in candidates:
        key = f"ma{points}_delay{delay}_lp{cutoff if cutoff is not None else 'none'}"
        per_dataset = {}
        for dataset in datasets:
            data = loaded[dataset.dataset_id]
            start, end, _ = data["slice"]
            warmup = int(data["fs"] * 15)
            values = thesis_aligned_difference(data["xF"], points, delay)
            values = lowpass(values, cutoff, data["fs"])
            per_dataset[dataset.dataset_id] = method_metrics(
                values[min(end, start + warmup) : end], data["fs"], internal_references[dataset.dataset_id]
            )
        score = candidate_score(per_dataset, humans, desks)
        candidate_results[key] = per_dataset
        search_rows.append({"key": key, "points": points, "delay": delay, "lowpassHz": cutoff, "score": score})
    search_rows.sort(key=lambda row: row["score"], reverse=True)
    selected = search_rows[0]
    selected_key = selected["key"]

    detailed = {}
    summary_rows = []
    derived_dir = args.output / "derived_signals"
    derived_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        data = loaded[dataset.dataset_id]
        fs = data["fs"]
        start, end, gaps = data["slice"]
        warm_start = min(end, start + int(fs * 15))
        mixed = data["xF"]
        tuned = thesis_aligned_difference(mixed, int(selected["points"]), int(selected["delay"]))
        tuned = lowpass(tuned, selected["lowpassHz"], fs)
        series = {
            "mixed": mixed,
            "wang_resp": wang_respiration(mixed, fs),
            "app_diff": data.get("hrRaw", app_direction_difference(mixed)),
            "wang_diff": thesis_aligned_difference(mixed, 10, 5),
            "tuned": tuned,
            "wang_fir": wang_narrow_fir(mixed, fs),
            "app_clean": data.get("cleanHeart", np.zeros_like(mixed)),
        }
        metrics = {
            key: method_metrics(values[warm_start:end], fs, internal_references[dataset.dataset_id])
            for key, values in series.items()
            if key != "mixed" and key != "wang_resp"
        }
        formal_bpm = data.get("bpm", np.array([]))
        formal_bpm = formal_bpm[(formal_bpm >= 40) & (formal_bpm <= 180)]
        detailed[dataset.dataset_id] = {
            "label": dataset.label,
            "placement": dataset.placement,
            "source": str(dataset.path),
            "sampleRateHz": fs,
            "samples": int(len(mixed)),
            "longestContinuousSamples": int(end - start),
            "gapCount": gaps,
            "formalAppBpmMedian": float(np.median(formal_bpm)) if len(formal_bpm) else None,
            "methods": metrics,
        }
        pd.DataFrame({
            "time_ms": data["time_ms"].astype(np.int64),
            "time_s": (data["time_ms"] - data["time_ms"][0]) / 1000.0,
            "mixed_xF": series["mixed"],
            "wang_resp_fir": series["wang_resp"],
            "current_app_direction_diff": series["app_diff"],
            "wang_aligned_diff_ma10_delay5": series["wang_diff"],
            f"selected_{selected_key}": series["tuned"],
            "wang_narrow_fir": series["wang_fir"],
            "current_app_cleanHeart": series["app_clean"],
        }).to_csv(
            derived_dir / f"{dataset.dataset_id}_derived_signals.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
        )
        for key, item in metrics.items():
            summary_rows.append({
                "dataset": dataset.dataset_id,
                "label": dataset.label,
                "method": key,
                "autocorrBpm": item["autocorrBpm"],
                "periodicity": item["periodicity"],
                "internalReferenceBpm": item["internalReferenceBpm"],
                "referencePeakBpm": item["referencePeakBpm"],
                "referencePeriodicity": item["referencePeriodicity"],
                "referenceRelativeError": item["referenceRelativeError"],
                "windowBpmCv": item["windowBpmCv"],
                "templateQuality": item["templateQuality"],
                "rrCv": item["rrCv"],
                "respLeakPercent": item["respLeakPercent"],
                "heartBandPercent": item["heartBandPercent"],
                "detailPercent": item["detailPercent"],
                "noisePercent": item["noisePercent"],
                "rms": item["rms"],
            })
        plot_overview(args.output, dataset, data["time_ms"], series, warm_start, fs)
        plot_templates(args.output, dataset, metrics)

    pd.DataFrame(search_rows).to_csv(args.output / "wang_parameter_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(args.output / "method_metrics.csv", index=False, encoding="utf-8-sig")
    result = {
        "selectedAlignedDifference": selected,
        "selectionStatement": "Selected by cross-position self-consistency and desk negative control; not ECG accuracy.",
        "datasets": detailed,
        "topTenCandidates": search_rows[:10],
    }
    (args.output / "reproduction_results.json").write_text(
        json.dumps(rounded(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(rounded({"selected": selected, "topFive": search_rows[:5]}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
