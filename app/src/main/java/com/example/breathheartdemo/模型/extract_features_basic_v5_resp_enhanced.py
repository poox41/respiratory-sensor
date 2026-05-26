#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


# =========================
# Basic stats
# =========================

def safe_mean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if len(x) else np.nan


def safe_std(x: np.ndarray) -> float:
    return float(np.nanstd(x)) if len(x) else np.nan


def robust_range(x: np.ndarray) -> float:
    if len(x) == 0:
        return np.nan
    return float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5))


def zero_crossing_rate(x: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    return float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1])))


def flat_ratio(x: np.ndarray, eps_scale: float = 1e-3) -> float:
    if len(x) < 2:
        return 1.0
    scale = float(np.nanstd(x))
    eps = max(scale * eps_scale, 1e-12)
    return float(np.mean(np.abs(np.diff(x)) < eps))


# =========================
# ECG filtering + R peaks
# =========================

def bandpass_filter(
    x: np.ndarray,
    fs: float,
    low: float = 5.0,
    high: float = 20.0,
    order: int = 3,
) -> np.ndarray:
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq

    if low_n <= 0 or high_n >= 1 or low_n >= high_n:
        return x

    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, x)


def detect_ecg_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    if len(ecg) < int(fs * 3):
        return np.array([], dtype=int)

    x = ecg.astype(np.float64)
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    x = x - np.nanmedian(x)

    try:
        xf = bandpass_filter(x, fs, low=5.0, high=20.0, order=3)
    except Exception:
        xf = x

    y = xf ** 2

    if not np.isfinite(y).all() or np.nanstd(y) < 1e-12:
        return np.array([], dtype=int)

    min_dist = max(1, int(fs * 0.45))

    p90 = np.nanpercentile(y, 90)
    p50 = np.nanpercentile(y, 50)
    prominence = max((p90 - p50) * 0.35, np.nanstd(y) * 0.25, 1e-12)

    peaks, _ = find_peaks(
        y,
        distance=min_dist,
        prominence=prominence,
    )

    return peaks.astype(int)


# =========================
# HR + HRV
# =========================

def estimate_hr(ecg: np.ndarray, fs: float) -> Dict[str, float]:
    peaks = detect_ecg_peaks(ecg, fs)

    if len(peaks) < 2:
        return {
            "hr_mean": np.nan,
            "hr_std": np.nan,
            "ecg_peak_count": float(len(peaks)),
        }

    rr_all = np.diff(peaks) / fs
    rr = rr_all[(rr_all >= 0.4) & (rr_all <= 1.5)]

    if len(rr) < 2:
        return {
            "hr_mean": np.nan,
            "hr_std": np.nan,
            "ecg_peak_count": float(len(peaks)),
        }

    hr = 60.0 / np.clip(rr, 1e-6, None)

    return {
        "hr_mean": float(np.mean(hr)),
        "hr_std": float(np.std(hr)),
        "ecg_peak_count": float(len(peaks)),
    }


def estimate_hrv(ecg: np.ndarray, fs: float) -> Dict[str, float]:
    peaks = detect_ecg_peaks(ecg, fs)

    rr_all = np.diff(peaks) / fs if len(peaks) >= 2 else np.array([])
    rr = rr_all[(rr_all >= 0.4) & (rr_all <= 1.5)] if len(rr_all) else np.array([])

    if len(rr) < 2:
        return {
            "rr_mean": np.nan,
            "rr_std": np.nan,
            "rr_median": np.nan,
            "rr_iqr": np.nan,
            "sdnn": np.nan,
            "rmssd": np.nan,
            "pnn20": np.nan,
            "pnn50": np.nan,
            "hr_min": np.nan,
            "hr_max": np.nan,
            "hr_range": np.nan,
            "rr_valid_count": float(len(rr)),
            "rr_all_count": float(len(rr_all)),
            "rr_valid_ratio": np.nan,
            "rr_invalid_ratio": np.nan,
        }

    diff = np.diff(rr)
    hr = 60.0 / np.clip(rr, 1e-6, None)
    valid_ratio = len(rr) / max(len(rr_all), 1)

    return {
        "rr_mean": float(np.mean(rr)),
        "rr_std": float(np.std(rr)),
        "rr_median": float(np.median(rr)),
        "rr_iqr": float(np.percentile(rr, 75) - np.percentile(rr, 25)),
        "sdnn": float(np.std(rr)),
        "rmssd": float(np.sqrt(np.mean(diff ** 2))) if len(diff) else np.nan,
        "pnn20": float(np.mean(np.abs(diff) > 0.02)) if len(diff) else np.nan,
        "pnn50": float(np.mean(np.abs(diff) > 0.05)) if len(diff) else np.nan,
        "hr_min": float(np.min(hr)),
        "hr_max": float(np.max(hr)),
        "hr_range": float(np.max(hr) - np.min(hr)),
        "rr_valid_count": float(len(rr)),
        "rr_all_count": float(len(rr_all)),
        "rr_valid_ratio": float(valid_ratio),
        "rr_invalid_ratio": float(1.0 - valid_ratio),
    }


# =========================
# Resp basic + enhanced
# =========================

def detect_resp_peaks(resp: np.ndarray, fs: float) -> np.ndarray:
    if len(resp) < int(fs * 5):
        return np.array([], dtype=int)

    x = resp.astype(np.float64)
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    x = x - np.nanmedian(x)

    min_dist = max(1, int(fs * 1.0))

    prominence = max(
        np.nanstd(x) * 0.15,
        (np.nanpercentile(x, 90) - np.nanpercentile(x, 50)) * 0.20,
        1e-12,
    )

    peaks, _ = find_peaks(
        x,
        distance=min_dist,
        prominence=prominence,
    )

    return peaks.astype(int)


def estimate_resp_rate(resp: np.ndarray, fs: float) -> Dict[str, float]:
    peaks = detect_resp_peaks(resp, fs)

    if len(peaks) < 2:
        return {
            "resp_rate_mean": np.nan,
            "resp_rate_std": np.nan,
            "resp_peak_count": float(len(peaks)),
            "resp_cycle_std": np.nan,
            "resp_cycle_iqr": np.nan,
        }

    cycle = np.diff(peaks) / fs
    cycle = cycle[(cycle >= 1.0) & (cycle <= 10.0)]

    if len(cycle) < 2:
        return {
            "resp_rate_mean": np.nan,
            "resp_rate_std": np.nan,
            "resp_peak_count": float(len(peaks)),
            "resp_cycle_std": np.nan,
            "resp_cycle_iqr": np.nan,
        }

    rate = 60.0 / np.clip(cycle, 1e-6, None)

    return {
        "resp_rate_mean": float(np.mean(rate)),
        "resp_rate_std": float(np.std(rate)),
        "resp_peak_count": float(len(peaks)),
        "resp_cycle_std": float(np.std(cycle)),
        "resp_cycle_iqr": float(np.percentile(cycle, 75) - np.percentile(cycle, 25)),
    }


def estimate_resp_extra(resp: np.ndarray) -> Dict[str, float]:
    if len(resp) < 2:
        return {
            "resp_amp_iqr": np.nan,
            "resp_diff_std": np.nan,
            "resp_diff_range": np.nan,
        }

    d = np.diff(resp)

    return {
        "resp_amp_iqr": float(np.percentile(resp, 75) - np.percentile(resp, 25)),
        "resp_diff_std": float(np.std(d)),
        "resp_diff_range": robust_range(d),
    }


def estimate_resp_enhanced(resp: np.ndarray, resp_fs: float, ecg: np.ndarray, ecg_fs: float) -> Dict[str, float]:
    """
    v5 enhanced respiratory features.
    Keep v4.1 intact, only add these columns.
    """
    out = {
        "resp_rate_median": np.nan,
        "resp_rate_iqr": np.nan,
        "resp_rate_min": np.nan,
        "resp_rate_max": np.nan,
        "resp_rate_range": np.nan,
        "resp_rate_cv": np.nan,

        "resp_cycle_mean": np.nan,
        "resp_cycle_std_enh": np.nan,
        "resp_cycle_cv": np.nan,
        "resp_cycle_rmssd": np.nan,
        "resp_cycle_pnn20": np.nan,

        "resp_amp_mean": np.nan,
        "resp_amp_std": np.nan,
        "resp_amp_cv": np.nan,
        "resp_amp_iqr_enh": np.nan,
        "resp_amp_range": np.nan,

        "resp_low_amp_ratio": np.nan,
        "resp_flat_ratio_strict": np.nan,
        "resp_pause_count": np.nan,
        "resp_pause_ratio": np.nan,

        "hr_resp_ratio": np.nan,
    }

    if len(resp) < int(resp_fs * 5):
        return out

    x = resp.astype(np.float64)
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    x = x - np.nanmedian(x)

    # ---------- respiratory cycle features ----------
    peaks = detect_resp_peaks(x, resp_fs)

    if len(peaks) >= 2:
        cycle_all = np.diff(peaks) / resp_fs
        cycle = cycle_all[(cycle_all >= 1.0) & (cycle_all <= 10.0)]

        if len(cycle) >= 2:
            rate = 60.0 / np.clip(cycle, 1e-6, None)
            diff_cycle = np.diff(cycle)

            out.update({
                "resp_rate_median": float(np.median(rate)),
                "resp_rate_iqr": float(np.percentile(rate, 75) - np.percentile(rate, 25)),
                "resp_rate_min": float(np.min(rate)),
                "resp_rate_max": float(np.max(rate)),
                "resp_rate_range": float(np.max(rate) - np.min(rate)),
                "resp_rate_cv": float(np.std(rate) / (np.mean(rate) + 1e-6)),

                "resp_cycle_mean": float(np.mean(cycle)),
                "resp_cycle_std_enh": float(np.std(cycle)),
                "resp_cycle_cv": float(np.std(cycle) / (np.mean(cycle) + 1e-6)),
                "resp_cycle_rmssd": float(np.sqrt(np.mean(diff_cycle ** 2))) if len(diff_cycle) else np.nan,
                "resp_cycle_pnn20": float(np.mean(np.abs(diff_cycle) > 0.20)) if len(diff_cycle) else np.nan,
            })

    # ---------- amplitude stability features ----------
    # Split the 30s epoch into 10 small chunks; compute local amplitude.
    n_chunks = 10
    chunk_len = len(x) // n_chunks

    if chunk_len >= 5:
        amps = []
        for i in range(n_chunks):
            seg = x[i * chunk_len:(i + 1) * chunk_len]
            if len(seg) >= 5:
                amps.append(np.percentile(seg, 95) - np.percentile(seg, 5))

        amps = np.asarray(amps, dtype=np.float64)

        if len(amps) >= 2:
            amp_mean = np.mean(amps)
            amp_std = np.std(amps)

            out.update({
                "resp_amp_mean": float(amp_mean),
                "resp_amp_std": float(amp_std),
                "resp_amp_cv": float(amp_std / (amp_mean + 1e-6)),
                "resp_amp_iqr_enh": float(np.percentile(amps, 75) - np.percentile(amps, 25)),
                "resp_amp_range": float(np.max(amps) - np.min(amps)),
            })

            low_thr = max(np.percentile(amps, 20), 1e-12)
            out["resp_low_amp_ratio"] = float(np.mean(amps <= low_thr))

    # ---------- pause / flat proxy ----------
    dx = np.abs(np.diff(x))
    if len(dx):
        strict_thr = max(np.nanstd(x) * 0.005, 1e-8)
        flat_bool = dx < strict_thr
        out["resp_flat_ratio_strict"] = float(np.mean(flat_bool))

        # pause-like windows: local amplitude very low.
        # Use 2-second windows.
        win = max(1, int(resp_fs * 2.0))
        step = max(1, win // 2)

        pause_count = 0
        total_windows = 0

        global_amp = np.percentile(x, 95) - np.percentile(x, 5)
        pause_amp_thr = max(global_amp * 0.10, 1e-12)

        for start in range(0, len(x) - win + 1, step):
            seg = x[start:start + win]
            local_amp = np.percentile(seg, 95) - np.percentile(seg, 5)
            total_windows += 1
            if local_amp <= pause_amp_thr:
                pause_count += 1

        if total_windows > 0:
            out["resp_pause_count"] = float(pause_count)
            out["resp_pause_ratio"] = float(pause_count / total_windows)

    # ---------- HR / Resp ratio ----------
    try:
        ecg_peaks = detect_ecg_peaks(ecg, ecg_fs)
        if len(ecg_peaks) >= 2:
            rr = np.diff(ecg_peaks) / ecg_fs
            rr = rr[(rr >= 0.4) & (rr <= 1.5)]

            if len(rr) >= 2:
                hr_mean = float(np.mean(60.0 / np.clip(rr, 1e-6, None)))
            else:
                hr_mean = np.nan
        else:
            hr_mean = np.nan

        if np.isfinite(out["resp_rate_median"]) and np.isfinite(hr_mean):
            out["hr_resp_ratio"] = float(hr_mean / (out["resp_rate_median"] + 1e-6))
    except Exception:
        pass

    return out


# =========================
# Extract one epoch
# =========================

def extract_one(npz_path: Path) -> Dict[str, object]:
    obj = np.load(npz_path, allow_pickle=True)

    ecg = obj["ecg"].astype(np.float64)
    resp = obj["resp"].astype(np.float64)
    ecg_fs = float(obj["ecg_fs"])
    resp_fs = float(obj["resp_fs"])

    row = {
        "subject_id": str(obj["subject_id"]),
        "epoch_idx": int(obj["epoch_idx"]),
        "label": int(obj["label"]),
        "split": str(obj["split"]),

        "ecg_signal_mean": safe_mean(ecg),
        "ecg_signal_std": safe_std(ecg),
        "ecg_signal_range": robust_range(ecg),
        "ecg_flat_ratio": flat_ratio(ecg),
        "ecg_zero_crossing_rate": zero_crossing_rate(ecg),

        "resp_mean": safe_mean(resp),
        "resp_std": safe_std(resp),
        "resp_signal_range": robust_range(resp),
        "resp_flat_ratio": flat_ratio(resp),
        "resp_zero_crossing_rate": zero_crossing_rate(resp),
    }

    # v4.1 original features
    row.update(estimate_hr(ecg, ecg_fs))
    row.update(estimate_hrv(ecg, ecg_fs))
    row.update(estimate_resp_rate(resp, resp_fs))
    row.update(estimate_resp_extra(resp))

    # v5 enhanced resp features
    row.update(estimate_resp_enhanced(resp, resp_fs, ecg, ecg_fs))

    return row


# =========================
# Main
# =========================

def main(index_csv: Path, output_csv: Path) -> None:
    index_df = pd.read_csv(index_csv)

    if "file_path" not in index_df.columns:
        raise ValueError("index.csv must contain column: file_path")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    total = len(index_df)
    chunk_size = 10000  # 每10000个样本写入一次
    rows: List[Dict[str, object]] = []
    first_write = True

    for i, fp in enumerate(index_df["file_path"].tolist(), start=1):
        try:
            rows.append(extract_one(Path(fp)))
        except Exception as e:
            print(f"[WARN] failed {fp}: {e}")
            # 失败时添加空行
            rows.append({
                "subject_id": "",
                "epoch_idx": -1,
                "label": -1,
                "split": "",
            })

        # 每 chunk_size 个样本写入一次
        if i % chunk_size == 0 or i == total:
            df_chunk = pd.DataFrame(rows)
            df_chunk.to_csv(
                output_csv,
                index=False,
                encoding="utf-8-sig",
                mode="a",
                header=first_write
            )
            print(f"[INFO] processed {i}/{total}")
            rows = []
            first_write = False

    print(f"[OK] Saved: {output_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    
    # 默认路径
    default_index = Path(__file__).parent / "work" / "processed_epochs" / "index.csv"
    default_output = Path(__file__).parent / "work" / "features" / "features_v5_resp_enhanced.csv"

    p.add_argument(
        "--index-csv",
        type=Path,
        default=default_index,
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=default_output,
    )

    args = p.parse_args()
    main(args.index_csv, args.output_csv)