#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_app.py - 睡眠分期模型推理脚本
用于将训练好的模型部署到应用程序中

使用方法:
    python inference_app.py --model-dir work/feature_tcn_simulated_w11
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 模型结构定义（必须与训练时一致）
# ============================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class CausalTCN(nn.Module):
    """
    因果时序卷积网络
    输入: (batch_size, seq_len, feature_dim)
    输出: (batch_size, 2) - [清醒概率, 睡眠概率]
    """

    def __init__(
        self,
        input_dim: int = 57,
        channels: list = None,
        kernel_size: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 64, 128, 128]

        layers = []
        for i, out_ch in enumerate(channels):
            in_ch = input_dim if i == 0 else channels[i - 1]
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1], 2)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, L, D) -> (B, D, L)
        h = self.tcn(x)
        h_last = h[:, :, -1]  # 取最后一个时间步
        return self.classifier(h_last)


# ============================================================
# 推理类
# ============================================================

class SleepStagingPredictor:
    """
    睡眠分期预测器

    使用方法:
        predictor = SleepStagingPredictor(model_dir="work/feature_tcn_simulated_w11")
        state, prob = predictor.predict(features)  # features: (11, 57)
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = None,
        seq_len: int = 11,
        feature_dim: int = 57,
    ):
        self.model_dir = Path(model_dir)
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 加载完整模型
        model_path = self.model_dir / "complete_model.pt"
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 创建模型并加载权重
        config = checkpoint["config"]
        self.model = CausalTCN(
            input_dim=config["feature_dim"],
            channels=config["channels"],
            kernel_size=config["kernel_size"],
            dropout=config["dropout"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model.to(self.device)

        # 加载归一化参数
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]
        self.feature_cols = list(checkpoint["feature_cols"])

        # 加载阈值
        self.threshold = checkpoint["threshold"]

        print(f"[INFO] Model loaded from: {model_dir}")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Threshold: {self.threshold}")
        print(f"[INFO] Features: {len(self.feature_cols)}")

    def preprocess(self, raw_features: np.ndarray) -> torch.Tensor:
        """
        预处理特征

        Args:
            raw_features: 原始特征 (seq_len, feature_dim) 或 (feature_dim,)

        Returns:
            归一化后的张量 (1, seq_len, feature_dim)
        """
        # 确保是2D数组
        if raw_features.ndim == 1:
            raw_features = raw_features.reshape(1, -1)

        # 检查特征维度
        if raw_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"特征维度不匹配: 期望 {self.feature_dim}, 得到 {raw_features.shape[-1]}"
            )

        # 归一化
        x_norm = (raw_features - self.mean) / self.std
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # 转换为张量
        x = torch.tensor(x_norm, dtype=torch.float32)
        return x

    def predict_one(self, raw_features: np.ndarray) -> Tuple[int, float]:
        """
        预测单个样本

        Args:
            raw_features: 原始特征 (seq_len, feature_dim)

        Returns:
            (state, prob_sleep): 状态 (0=清醒, 1=睡眠) 和睡眠概率
        """
        x = self.preprocess(raw_features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            prob_sleep = torch.softmax(logits, dim=1)[0, 1].item()

        state = 1 if prob_sleep >= self.threshold else 0
        return state, prob_sleep

    def predict_batch(self, raw_features_batch: np.ndarray) -> Tuple[List[int], List[float]]:
        """
        批量预测

        Args:
            raw_features_batch: 原始特征 (batch, seq_len, feature_dim)

        Returns:
            (states, probs): 状态列表和概率列表
        """
        x = self.preprocess(raw_features_batch).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        states = (probs >= self.threshold).astype(int).tolist()
        return states, probs.tolist()

    def predict_streaming(self, feature_buffer: List | np.ndarray) -> Tuple[int, float]:
        """
        流式预测 - 持续接收特征并预测

        Args:
            feature_buffer: 特征缓冲区，需包含至少 seq_len 个特征

        Returns:
            (state, prob_sleep): 当前状态和概率
        """
        if isinstance(feature_buffer, List):
            feature_buffer = np.array(feature_buffer)

        if len(feature_buffer) < self.seq_len:
            raise ValueError(
                f"特征数量不足: 需要 {self.seq_len}, 当前 {len(feature_buffer)}"
            )

        # 取最后 seq_len 个特征
        features = feature_buffer[-self.seq_len:]
        return self.predict_one(features)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="睡眠分期模型推理")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="模型目录路径")
    parser.add_argument("--device", type=str, default=None,
                        help="设备: cuda 或 cpu")
    args = parser.parse_args()

    # 创建预测器
    predictor = SleepStagingPredictor(args.model_dir, device=args.device)

    # 测试推理
    print("\n" + "=" * 50)
    print("模型推理测试")
    print("=" * 50)

    # 生成随机测试数据（实际使用时替换为真实特征）
    test_features = np.random.randn(11, 57).astype(np.float32)
    state, prob = predictor.predict_one(test_features)

    print(f"测试输入形状: {test_features.shape}")
    print(f"预测状态: {'睡眠' if state == 1 else '清醒'}")
    print(f"睡眠概率: {prob:.4f}")
    print(f"阈值: {predictor.threshold}")
    print("=" * 50)


if __name__ == "__main__":
    main()
