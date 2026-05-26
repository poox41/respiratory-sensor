#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    confusion_matrix,
)


class SubjectSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        seq_len: int,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        self.df = df.sort_values(["subject_id", "epoch_idx"]).reset_index(drop=True)
        self.feature_cols = feature_cols
        self.seq_len = seq_len

        x_raw = self.df[feature_cols].values.astype(np.float32)
        x_raw = np.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)

        if mean is None:
            self.mean = np.mean(x_raw, axis=0).astype(np.float32)
            self.std = np.std(x_raw, axis=0).astype(np.float32)
            self.std[self.std < 1e-6] = 1.0
        else:
            self.mean = mean.astype(np.float32)
            self.std = std.astype(np.float32)

        x = (x_raw - self.mean) / self.std
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        self.X = x.astype(np.float32)
        self.y = self.df["label"].values.astype(np.int64)
        self.subject_id = self.df["subject_id"].astype(str).values
        self.epoch_idx = self.df["epoch_idx"].values.astype(np.int64)

        self.samples: List[Tuple[int, int]] = []
        self.skipped_discontinuous = 0

        for _, g in self.df.groupby("subject_id", sort=False):
            idxs = g.index.to_numpy()
            epochs = g["epoch_idx"].to_numpy()

            if len(idxs) < seq_len:
                continue

            for i in range(len(idxs) - seq_len + 1):
                win_idxs = idxs[i:i + seq_len]
                win_epochs = epochs[i:i + seq_len]

                if np.all(np.diff(win_epochs) == 1):
                    self.samples.append((int(win_idxs[0]), int(win_idxs[-1])))
                else:
                    self.skipped_discontinuous += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start, end = self.samples[idx]

        x = self.X[start:end + 1]
        y = self.y[end]
        sid = self.subject_id[end]
        epoch = self.epoch_idx[end]

        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.long),
            sid,
            torch.tensor(epoch, dtype=torch.long),
        )


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
    def __init__(
        self,
        input_dim: int,
        channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

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
        x = x.transpose(1, 2)
        h = self.tcn(x)
        h_last = h[:, :, -1]
        return self.classifier(h_last)


def compute_metrics(y_true, y_pred, threshold):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),

        "wake_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "wake_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "wake_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),

        "sleep_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "sleep_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "sleep_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),

        "tn_wake_correct": int(tn),
        "fp_wake_to_sleep": int(fp),
        "fn_sleep_to_wake": int(fn),
        "tp_sleep_correct": int(tp),

        "threshold": float(threshold),
    }


def predict_probs(model, loader, device):
    model.eval()

    y_all = []
    p_all = []
    sid_all = []
    epoch_all = []

    with torch.no_grad():
        for x, y, sid, epoch in loader:
            x = x.to(device, non_blocking=True)

            logits = model(x)
            prob_sleep = torch.softmax(logits, dim=1)[:, 1]

            y_all.append(y.numpy())
            p_all.append(prob_sleep.cpu().numpy())
            sid_all.extend(list(sid))
            epoch_all.append(epoch.numpy())

    return (
        np.concatenate(y_all),
        np.concatenate(p_all),
        np.array(sid_all),
        np.concatenate(epoch_all),
    )


def threshold_sweep(y_true, prob_sleep, objective):
    rows = []
    best = None

    for th in np.arange(0.05, 0.951, 0.005):
        y_pred = (prob_sleep >= th).astype(int)
        m = compute_metrics(y_true, y_pred, threshold=th)
        rows.append(m)

        if best is None or m[objective] > best[objective]:
            best = m

    return best, pd.DataFrame(rows)


def load_and_merge_split(features_path: Path, split_path: Path | None):
    df = pd.read_csv(features_path)

    if split_path is not None:
        split_df = pd.read_csv(split_path)

        if "split" in df.columns:
            df = df.drop(columns=["split"])

        if "subject_id" not in split_df.columns or "split" not in split_df.columns:
            raise ValueError("split_path must contain columns: subject_id, split")

        df["subject_id"] = df["subject_id"].astype(str)
        split_df["subject_id"] = split_df["subject_id"].astype(str)

        df = df.merge(split_df[["subject_id", "split"]], on="subject_id", how="left")

    if "split" not in df.columns:
        raise ValueError("No split column found. Provide --split-path or include split in features CSV.")

    missing_split = df["split"].isna().sum()
    if missing_split > 0:
        raise ValueError(f"{missing_split} rows have missing split after merge.")

    return df


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    features_path = Path(args.features_path)
    split_path = Path(args.split_path) if args.split_path else None

    df = load_and_merge_split(features_path, split_path)
    df = df.sort_values(["subject_id", "epoch_idx"]).reset_index(drop=True)

    exclude_cols = {"subject_id", "epoch_idx", "label", "split"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_ds = SubjectSequenceDataset(train_df, feature_cols, args.seq_len)
    val_ds = SubjectSequenceDataset(val_df, feature_cols, args.seq_len, train_ds.mean, train_ds.std)
    test_ds = SubjectSequenceDataset(test_df, feature_cols, args.seq_len, train_ds.mean, train_ds.std)

    np.savez(
        outdir / "norm_stats.npz",
        mean=train_ds.mean,
        std=train_ds.std,
        feature_cols=np.array(feature_cols),
    )

    print("===== CAUSAL TCN V4 =====")
    print("Features path:", features_path)
    print("Split path:", split_path)
    print("Feature dim:", len(feature_cols))
    print("Seq len:", args.seq_len)
    print("Train rows:", len(train_df), "samples:", len(train_ds), "skipped:", train_ds.skipped_discontinuous)
    print("Val rows:", len(val_df), "samples:", len(val_ds), "skipped:", val_ds.skipped_discontinuous)
    print("Test rows:", len(test_df), "samples:", len(test_ds), "skipped:", test_ds.skipped_discontinuous)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    channels = [int(x.strip()) for x in args.channels.split(",") if x.strip()]

    model = CausalTCN(
        input_dim=len(feature_cols),
        channels=channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)

    if args.class_weight_mode == "auto":
        counts = train_df["label"].value_counts().sort_index()
        n0 = int(counts.get(0, 1))
        n1 = int(counts.get(1, 1))
        total = n0 + n1
        w0 = total / (2.0 * n0)
        w1 = total / (2.0 * n1)
        weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        print("Class weights [Wake, Sleep]:", weights.detach().cpu().tolist())
    else:
        weights = None
        print("Class weights: None")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_score = -1e18
    best_epoch = -1
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_seen = 0

        for x, y, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_seen += bs

        train_loss = total_loss / max(total_seen, 1)

        y_val, p_val, _, _ = predict_probs(model, val_loader, device)
        val_best, _ = threshold_sweep(y_val, p_val, args.threshold_objective)

        score = val_best[args.model_select_metric]
        scheduler.step(score)

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "lr": float(current_lr),
            **{f"val_{k}": v for k, v in val_best.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} "
            f"val_acc={val_best['accuracy']:.4f} "
            f"val_kappa={val_best['kappa']:.4f} "
            f"val_macro_f1={val_best['macro_f1']:.4f} "
            f"val_wake_f1={val_best['wake_f1']:.4f} "
            f"th={val_best['threshold']:.3f} "
            f"lr={current_lr:.2e}"
        )

        if score > best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            bad_epochs = 0

            torch.save(model.state_dict(), outdir / "best_model.pt")

            with open(outdir / "best_val_metrics.json", "w", encoding="utf-8") as f:
                json.dump(val_best, f, indent=2, ensure_ascii=False)
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False, encoding="utf-8-sig")

    model.load_state_dict(torch.load(outdir / "best_model.pt", map_location=device))

    y_test, p_test, sid_test, epoch_test = predict_probs(model, test_loader, device)
    test_best, test_sweep = threshold_sweep(y_test, p_test, args.threshold_objective)

    test_best["best_epoch"] = int(best_epoch)
    test_best["best_score"] = float(best_score)

    pred = (p_test >= test_best["threshold"]).astype(int)

    pred_df = pd.DataFrame({
        "subject_id": sid_test,
        "epoch_idx": epoch_test.astype(int),
        "label": y_test.astype(int),
        "prob_sleep": p_test.astype(float),
        "pred": pred.astype(int),
    })

    pred_df = pred_df.sort_values(["subject_id", "epoch_idx"]).reset_index(drop=True)

    pred_df.to_csv(outdir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    test_sweep.to_csv(outdir / "threshold_sensitivity.csv", index=False, encoding="utf-8-sig")

    with open(outdir / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_best, f, indent=2, ensure_ascii=False)

    print("\n===== Final Test Metrics =====")
    for k, v in test_best.items():
        print(f"{k}: {v}")

    print("\n[OK] Saved:", outdir / "best_model.pt")
    print("[OK] Saved:", outdir / "best_val_metrics.json")
    print("[OK] Saved:", outdir / "history.csv")
    print("[OK] Saved:", outdir / "norm_stats.npz")
    print("[OK] Saved:", outdir / "threshold_sensitivity.csv")
    print("[OK] Saved:", outdir / "test_predictions.csv")
    print("[OK] Saved:", outdir / "final_test_metrics.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--features-path", required=True)
    p.add_argument("--split-path", default=None)
    p.add_argument("--outdir", required=True)

    p.add_argument("--seq-len", type=int, default=61)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--channels", type=str, default="64,64,128,128")
    p.add_argument("--kernel-size", type=int, default=3)

    p.add_argument("--class-weight-mode", choices=["none", "auto"], default="auto")
    p.add_argument(
        "--threshold-objective",
        choices=["accuracy", "kappa", "macro_f1", "wake_f1", "wake_recall"],
        default="accuracy",
    )
    p.add_argument(
        "--model-select-metric",
        choices=["accuracy", "kappa", "macro_f1", "wake_f1", "wake_recall"],
        default="accuracy",
    )

    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)

    args = p.parse_args()
    main(args)