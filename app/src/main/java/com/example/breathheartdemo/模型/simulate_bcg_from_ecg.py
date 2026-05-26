#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_bcg_from_ecg.py

读取 processed_epochs_v2_thor 中的每个 npz 文件，
将 ECG 信号转换为模拟压力震动（BCG）信号，呼吸信号保持不变。
生成的新数据集保存在 processed_epochs_v2_simulated 中，
并生成新的 index.csv。

原理：
- 保留 ECG 的 R 峰位置（节律信息），替换尖峰为高斯宽脉冲
- 添加粉红噪声 + 基线漂移，模拟真实压力传感器采集环境
- 呼吸通道（THOR RES）与压力传感器的呼吸运动成分物理同源，无需改动
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from tqdm import tqdm


# ================== ECG 峰值检测（与特征提取脚本一致） ==================
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
    """与 extract_features_basic_v5_resp_enhanced 中的实现完全一致"""
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


# ================== 粉红噪声生成 ==================
def generate_pink_noise(n: int, alpha: float = 1.0) -> np.ndarray:
    """
    通过频域滤波生成近似粉红噪声 (1/f^alpha)。
    alpha=1.0 时是标准粉红噪声，这里略微调整以模拟传感器噪声。
    """
    f = np.fft.rfftfreq(n)
    f[0] = 1e-12  # 避免除零
    spectrum = 1.0 / (f ** (alpha / 2.0))
    spectrum = spectrum / np.sqrt(np.mean(spectrum ** 2))  # 归一化
    noise = np.random.randn(n)
    noise_fft = np.fft.rfft(noise)
    noise_fft = noise_fft * spectrum
    colored = np.fft.irfft(noise_fft, n=n)
    return colored / np.std(colored)


# ================== ECG 转 BCG 模拟 ==================
def ecg_to_simulated_bcg(
    ecg: np.ndarray,
    ecg_fs: float = 100.0,
    noise_level: float = 0.3,
    drift_amplitude: float = 0.05,
    drift_freq: float = 0.02,
    pulse_width_factor: float = 0.15,
    add_pink: bool = True,
) -> np.ndarray:
    """
    将 SHHS ECG 转换为模拟压力震动 BCG 信号。

    参数：
    ecg: 原始 ECG 信号（长度 = 30 s * ecg_fs）
    ecg_fs: 采样率（默认 100 Hz）
    noise_level: 白噪声标准差相对于 ECG 标准差的倍数
    drift_amplitude: 正弦基线漂移幅度（单位：ECG 标准差）
    drift_freq: 基线漂移频率（Hz）
    pulse_width_factor: BCG 脉冲宽度因子（乘以 fs 得到半窗宽度）
    add_pink: 是否添加粉红噪声（模拟低频震动）

    返回：
    simulated_bcg: 模拟 BCG 信号
    peak_count: 检测到的 R 峰数量（用于质量检查）
    """
    if len(ecg) < 3:
        return ecg.copy(), 0

    # 1. 检测 R 峰位置
    peaks = detect_ecg_peaks(ecg, ecg_fs)

    # 2. 生成高斯脉冲模板
    pulse_half_width = max(1, int(ecg_fs * pulse_width_factor))
    t = np.arange(-pulse_half_width, pulse_half_width + 1)
    sigma = pulse_half_width / 3.0
    pulse = np.exp(-0.5 * (t / sigma) ** 2)

    # 3. 构建模拟信号：在 R 峰位置叠加脉冲
    bcg = np.zeros_like(ecg, dtype=np.float64)
    ecg_std = float(np.std(ecg))

    for p in peaks:
        if p < pulse_half_width or p + pulse_half_width >= len(ecg):
            continue
        # 脉冲幅度：使用原始 ecg 在该点附近的能量作为权重
        local_amp = np.max(np.abs(ecg[max(0, p - 5):min(len(ecg), p + 5)]))
        amplitude = max(local_amp, ecg_std * 0.5)  # 确保最小幅度
        bcg[p - pulse_half_width:p + pulse_half_width + 1] += pulse * amplitude

    # 4. 添加基线漂移（低频正弦）
    t_arr = np.arange(len(ecg)) / ecg_fs
    drift = drift_amplitude * ecg_std * np.sin(2 * np.pi * drift_freq * t_arr)

    # 5. 添加噪声（白噪声 + 可选粉红噪声）
    white_noise = np.random.randn(len(ecg)) * ecg_std * noise_level
    if add_pink:
        pink_noise = generate_pink_noise(len(ecg), alpha=1.0) * ecg_std * noise_level * 0.5
        total_noise = white_noise + pink_noise
    else:
        total_noise = white_noise

    simulated = bcg + drift + total_noise

    # 确保数值稳定性
    simulated = np.nan_to_num(simulated, nan=0.0, posinf=0.0, neginf=0.0)
    simulated = np.clip(simulated, -10 * ecg_std, 10 * ecg_std)

    return simulated.astype(np.float32), len(peaks)


# ================== 批量处理 ==================
def process_directory(input_dir: Path, output_dir: Path, index_csv_out: Path) -> None:
    """
    遍历 input_dir/split 子目录下的所有 npz 文件，生成模拟 BCG 数据集。
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    rows = []
    total_processed = 0
    skipped_no_peaks = 0

    # 寻找所有 npz 文件（可能位于 train/val/test 子目录下）
    npz_files = list(input_dir.rglob("*.npz"))
    if not npz_files:
        raise ValueError(f"在 {input_dir} 中未找到任何 npz 文件")

    print(f"找到 {len(npz_files)} 个 npz 文件")

    for npz_path in tqdm(npz_files, desc="模拟 BCG 转换"):
        try:
            data = np.load(npz_path, allow_pickle=True)
            ecg = data["ecg"].astype(np.float64)
            resp = data["resp"].astype(np.float64)
            ecg_fs = float(data["ecg_fs"])
            resp_fs = float(data["resp_fs"])
            subject_id = str(data["subject_id"])
            epoch_idx = int(data["epoch_idx"])
            label = int(data["label"])
            split = str(data.get("split", "unknown"))
            ecg_channel = str(data.get("ecg_channel", "ECG"))
            resp_channel = str(data.get("resp_channel", "THOR RES"))

            # 生成模拟 BCG
            simulated_bcg, peak_count = ecg_to_simulated_bcg(ecg, ecg_fs)

            if peak_count < 2:
                skipped_no_peaks += 1
                continue  # 跳过质量太差的 epoch

            # 确定输出路径（保持原 split 目录结构）
            rel_path = npz_path.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存新的 npz（保持所有元信息不变）
            np.savez_compressed(
                out_path,
                subject_id=subject_id,
                epoch_idx=epoch_idx,
                split=split,
                label=label,
                ecg=simulated_bcg,
                resp=resp,
                ecg_fs=ecg_fs,
                resp_fs=resp_fs,
                ecg_channel=ecg_channel,
                resp_channel=resp_channel,
            )

            rows.append([subject_id, epoch_idx, label, split, str(out_path)])
            total_processed += 1

        except Exception as e:
            print(f"[WARN] 处理 {npz_path} 失败: {e}")

    print(f"\n处理完成: {total_processed} 个 epoch")
    print(f"因峰值过少跳过: {skipped_no_peaks} 个 epoch")

    # 保存 index.csv
    df = pd.DataFrame(rows, columns=["subject_id", "epoch_idx", "label", "split", "file_path"])
    df.to_csv(index_csv_out, index=False, encoding="utf-8-sig")
    print(f"已保存 index: {index_csv_out}")


# ================== 命令行参数 ==================
def main():
    parser = argparse.ArgumentParser(
        description="将 SHHS ECG 转换为模拟压力震动 BCG 信号"
    )
    # 默认路径：与 build_epochs_v2_thor_only.py 的输出保持一致
    default_input = Path(__file__).parent / "work" / "processed_epochs"
    default_output = Path(__file__).parent / "work" / "processed_epochs_simulated"
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="原始 epoch 目录（包含 train/val/test 子目录）"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="模拟 BCG 数据集输出目录"
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.3,
        help="白噪声强度（相对于 ECG 标准差）"
    )
    parser.add_argument(
        "--drift-amplitude",
        type=float,
        default=0.05,
        help="基线漂移幅度（相对于 ECG 标准差）"
    )
    args = parser.parse_args()

    index_csv_out = args.output_dir / "index.csv"

    process_directory(args.input_dir, args.output_dir, index_csv_out)


if __name__ == "__main__":
    main()