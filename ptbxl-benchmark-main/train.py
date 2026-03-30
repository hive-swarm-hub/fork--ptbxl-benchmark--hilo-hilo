"""
PTB-XL 12-Lead ECG Classification

Compact 1D ResNet on raw ECG signals. CNN-only (no LightGBM) to fit <10 min.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + (self.downsample(x) if self.downsample else x))


class ECGResNet(nn.Module):
    """Compact 1D ResNet — ~480K params, ~5 min on CPU for 15 epochs."""
    def __init__(self, n_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 32, 15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlock1D(32, 64, stride=2)
        self.layer2 = ResBlock1D(64, 128, stride=2)
        self.layer3 = ResBlock1D(128, 128, stride=2)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.drop   = nn.Dropout(0.3)
        self.fc     = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.fc(self.drop(self.pool(x).squeeze(-1)))


def main():
    os.makedirs("predictions", exist_ok=True)

    with open("data/registry.json") as f:
        registry = json.load(f)
    classes = registry["classes"]

    print("Loading data...")
    X_train = np.load("data/X_train.npy")
    X_test  = np.load("data/X_test.npy")
    y_train = pd.read_csv("data/y_train.csv")
    test_ids = pd.read_csv("data/y_test_ids.csv")["ecg_id"].values
    y_train_np = y_train[classes].values.astype(np.float32)

    # Normalize per lead (train stats)
    X_tr = X_train.transpose(0, 2, 1).astype(np.float32)  # (N, 12, 1000)
    X_te = X_test.transpose(0, 2, 1).astype(np.float32)
    mean = X_tr.mean(axis=(0, 2), keepdims=True)
    std  = X_tr.std(axis=(0, 2), keepdims=True) + 1e-6
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    pos_weight = torch.tensor(
        [(y_train_np[:, i] == 0).sum() / max(y_train_np[:, i].sum(), 1)
         for i in range(y_train_np.shape[1])],
        dtype=torch.float32,
    )

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_train_np)),
        batch_size=128, shuffle=True, num_workers=0,
    )

    model = ECGResNet(n_classes=len(classes))
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("Training 1D ResNet (15 epochs)...")
    model.train()
    for epoch in range(15):
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()
        print(f"  epoch {epoch+1}/15  loss={total_loss/len(loader):.4f}")

    model.eval()
    X_te_t = torch.from_numpy(X_te)
    preds_arr = []
    with torch.no_grad():
        for i in range(0, len(X_te_t), 256):
            preds_arr.append(torch.sigmoid(model(X_te_t[i:i+256])).numpy())
    preds_arr = np.concatenate(preds_arr, axis=0)

    pred_df = pd.DataFrame({cls: preds_arr[:, i] for i, cls in enumerate(classes)})
    pred_df.insert(0, "ecg_id", test_ids)
    pred_df.to_csv("predictions/predictions.csv", index=False)
    print(f"Done. {len(pred_df)} rows saved.")


if __name__ == "__main__":
    main()
