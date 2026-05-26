#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_epochs_v2_thor_only.py
从打包好的 dataset_packed 目录构建 SHHS 的 ECG + 胸呼吸带 epoch NPZ 文件。
目录结构：
    dataset_root/
    ├── train/edf/ + train/xml/
    ├── val/edf/ + val/xml/
    └── test/edf/ + test/xml/
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm

TARGET_ECG_FS = 100
TARGET_RESP_FS = 25
EPOCH_SEC = 30

ECG_ALIASES = [
    "ecg", "ekg", "lead ii", "lead2", "lead 2", "ecg1", "ecg2", "modified lead ii"
]

# 只保留胸腔呼吸运动通道，完全排除腹部及其他
RESP_PRIORITY_GROUPS = [
    ["thor res", "thor resp", "thoracic", "thorax", "chest"],
]


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def find_ecg_index(labels: List[str]) -> Optional[int]:
    norm = [normalize_name(x) for x in labels]
    for i, lab in enumerate(norm):
        if lab in ECG_ALIASES:
            return i
    for i, lab in enumerate(norm):
        if any(alias in lab for alias in ECG_ALIASES):
            return i
    return None


def find_resp_candidates(labels: List[str]) -> List[int]:
    norm = [normalize_name(x) for x in labels]
    found: List[int] = []

    for group in RESP_PRIORITY_GROUPS:
        for i, lab in enumerate(norm):
            if i in found:
                continue
            if any(alias == lab or alias in lab for alias in group):
                found.append(i)

    # 不进行宽泛匹配，避免引入无关通道
    return found


def butter_bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low = max(low, 0.001)
    high = min(high, nyq * 0.95)
    if not (0 < low < high < nyq):
        return x
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def butter_lowpass_filter(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    cutoff = min(cutoff, nyq * 0.95)
    if not (0 < cutoff < nyq):
        return x
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, x)


def resample_signal(x: np.ndarray, fs_old: float, fs_new: float) -> np.ndarray:
    n_new = int(round(len(x) * fs_new / fs_old))
    return resample(x, n_new)


def parse_shhs_xml(xml_path: Path) -> List[int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    stages = [int(x.text) for x in root.findall(".//SleepStage") if x.text is not None]
    return stages


def map_stage_to_binary(stage: int) -> int:
    if stage == 0:
        return 0  # Wake
    elif stage in [1, 2, 3, 4, 5]:
        return 1  # Sleep
    return -1


def flat_ratio(x: np.ndarray, eps_scale: float = 1e-3) -> float:
    if len(x) < 2:
        return 1.0
    scale = float(np.std(x))
    eps = max(scale * eps_scale, 1e-12)
    dx = np.abs(np.diff(x))
    return float(np.mean(dx < eps))


def resp_name_priority(name: str) -> float:
    """只给胸带高分，腹带不再参与评分"""
    n = normalize_name(name)
    if "thor" in n or "thorax" in n or "chest" in n:
        return 4.0
    # 腹带或其他通道不给分，不会被选中
    return 0.0


def score_resp_candidate(name: str, sig: np.ndarray) -> float:
    score = resp_name_priority(name)
    score += 2.0 * (1.0 - flat_ratio(sig))
    score += 0.5 * min(float(np.std(sig)), 10.0)
    score += 0.1 * min(float(np.max(sig) - np.min(sig)), 20.0)
    return float(score)


def load_split_lists(split_dir: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for split in ["train", "val", "test"]:
        txt = split_dir / f"{split}_list.txt"
        if not txt.exists():
            raise FileNotFoundError(f"Missing split file: {txt}")
        ids = [x.strip() for x in txt.read_text(encoding="utf-8").splitlines() if x.strip()]
        out[split] = ids
    return out


def process_one_record(edf_path: Path, xml_path: Path, out_dir: Path, split: str) -> List[List[object]]:
    with pyedflib.EdfReader(str(edf_path)) as f:
        labels = f.getSignalLabels()
        ecg_idx = find_ecg_index(labels)
        resp_candidates = find_resp_candidates(labels)

        if ecg_idx is None or not resp_candidates:
            raise ValueError(f"Missing ECG or Thor Resp in {edf_path.name}")

        ecg = f.readSignal(ecg_idx).astype(np.float64)
        ecg_fs = float(f.getSampleFrequency(ecg_idx))

        candidate_payloads = []
        for idx in resp_candidates:
            sig = f.readSignal(idx).astype(np.float64)
            fs = float(f.getSampleFrequency(idx))
            candidate_payloads.append((labels[idx], idx, fs, sig))

    resp_name, _, resp_fs, resp = max(
        candidate_payloads,
        key=lambda x: score_resp_candidate(x[0], x[3])
    )

    ecg_high = min(40.0, 0.45 * ecg_fs)
    ecg = butter_bandpass_filter(ecg, ecg_fs, 0.5, ecg_high)
    resp = butter_lowpass_filter(resp, resp_fs, 1.0)

    ecg = resample_signal(ecg, ecg_fs, TARGET_ECG_FS).astype(np.float32)
    resp = resample_signal(resp, resp_fs, TARGET_RESP_FS).astype(np.float32)

    stages = parse_shhs_xml(xml_path)
    labels_bin = [map_stage_to_binary(x) for x in stages]

    ecg_epoch_len = TARGET_ECG_FS * EPOCH_SEC
    resp_epoch_len = TARGET_RESP_FS * EPOCH_SEC
    n_epochs = min(len(labels_bin), len(ecg) // ecg_epoch_len, len(resp) // resp_epoch_len)

    subject_id = edf_path.stem
    rows: List[List[object]] = []

    for i in range(n_epochs):
        y = labels_bin[i]
        if y == -1:
            continue

        ecg_seg = ecg[i * ecg_epoch_len:(i + 1) * ecg_epoch_len]
        resp_seg = resp[i * resp_epoch_len:(i + 1) * resp_epoch_len]

        out_path = out_dir / split / f"{subject_id}_epoch_{i:04d}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            subject_id=subject_id,
            epoch_idx=i,
            split=split,
            label=y,
            ecg=ecg_seg,
            resp=resp_seg,
            ecg_fs=TARGET_ECG_FS,
            resp_fs=TARGET_RESP_FS,
            ecg_channel="ECG",
            resp_channel=resp_name,
        )

        rows.append([subject_id, i, y, split, str(out_path)])

    return rows


def build_all(dataset_root: Path, out_dir: Path, index_csv: Path) -> None:
    """从打包目录直接构建 epoch"""
    splits = ["train", "val", "test"]
    
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_all: List[List[object]] = []

    for split in splits:
        edf_dir = dataset_root / split / "edf"
        xml_dir = dataset_root / split / "xml"
        
        if not edf_dir.exists() or not xml_dir.exists():
            print(f"[WARN] Missing {split}/edf or {split}/xml, skipping")
            continue
        
        edf_files = sorted(edf_dir.glob("*.edf"))
        if not edf_files:
            print(f"[WARN] No EDF files in {split}/edf, skipping")
            continue
            
        print(f"\nProcessing {split}: {len(edf_files)} files")
        
        for edf_path in tqdm(edf_files, desc=f"Build {split}"):
            xml_path = xml_dir / f"{edf_path.stem}-profusion.xml"
            if not xml_path.exists():
                print(f"[WARN] Missing XML for {edf_path.name}")
                continue
            try:
                rows = process_one_record(edf_path, xml_path, out_dir, split)
                rows_all.extend(rows)
            except Exception as e:
                print(f"[WARN] Failed on {edf_path.name}: {type(e).__name__}: {e}")

    df = pd.DataFrame(rows_all, columns=["subject_id", "epoch_idx", "label", "split", "file_path"])
    df.to_csv(index_csv, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Saved index: {index_csv}")
    print(df["split"].value_counts())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).parent / "dataset_packed",
        help="打包数据集根目录 (包含 train/val/test 子目录)"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "work" / "processed_epochs",
        help="epoch 输出目录"
    )
    p.add_argument(
        "--index-csv",
        type=Path,
        default=None,
        help="index.csv 路径 (默认: out-dir/index.csv)"
    )
    args = p.parse_args()
    
    # 默认 index_csv 路径
    if args.index_csv is None:
        args.index_csv = args.out_dir / "index.csv"
    
    return args


if __name__ == "__main__":
    args = parse_args()
    build_all(args.dataset_root, args.out_dir, args.index_csv)