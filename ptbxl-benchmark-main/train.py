"""
PTB-XL 12-Lead ECG Classification

1D ResNet on raw ECG signals (12 leads, 1000 timesteps) + LightGBM ensemble.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy import stats as sp_stats
from scipy.signal import butter, filtfilt, find_peaks
import pywt
import lightgbm as lgb


# ── 1D ResNet ───────────────────────────────────────────────────────────────

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
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ECGResNet(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 32, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlock1D(32, 32)
        self.layer2 = ResBlock1D(32, 64, stride=2)
        self.layer3 = ResBlock1D(64, 64)
        self.layer4 = ResBlock1D(64, 128, stride=2)
        self.layer5 = ResBlock1D(128, 128)
        self.layer6 = ResBlock1D(128, 256, stride=2)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.drop   = nn.Dropout(0.3)
        self.fc     = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.fc(x)


def train_cnn(X_train, y_train_np, X_test, n_epochs=20, batch_size=128, lr=1e-3):
    """Train 1D ResNet, return test probabilities."""
    # Normalize per sample: (sample - mean) / std
    X_tr = X_train.transpose(0, 2, 1).astype(np.float32)  # (N, 12, 1000)
    X_te = X_test.transpose(0, 2, 1).astype(np.float32)
    mean = X_tr.mean(axis=(0, 2), keepdims=True)
    std  = X_tr.std(axis=(0, 2), keepdims=True) + 1e-6
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t  = torch.from_numpy(y_train_np.astype(np.float32))
    X_te_t  = torch.from_numpy(X_te)

    # Class weights for pos_weight (imbalance handling)
    pos_weight = torch.tensor(
        [(y_train_np[:, i] == 0).sum() / max(y_train_np[:, i].sum(), 1)
         for i in range(y_train_np.shape[1])],
        dtype=torch.float32,
    )

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model   = ECGResNet(n_classes=y_train_np.shape[1])
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()
        print(f"  epoch {epoch+1}/{n_epochs}  loss={total_loss/len(loader):.4f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te_t), 256):
            batch = X_te_t[i:i+256]
            preds.append(torch.sigmoid(model(batch)).numpy())
    return np.concatenate(preds, axis=0)  # (N_test, 5)


# ── LightGBM features ───────────────────────────────────────────────────────

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=100.0, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal, axis=1)


def extract_features(X):
    n_samples, n_timesteps, n_leads = X.shape
    features = []

    X_filt = np.zeros_like(X)
    for i in range(n_samples):
        try:
            X_filt[i] = bandpass_filter(X[i])
        except Exception:
            X_filt[i] = X[i]

    for i in range(n_leads):
        lead = X_filt[:, :, i]
        features += [
            np.mean(lead, axis=1), np.std(lead, axis=1),
            np.min(lead, axis=1), np.max(lead, axis=1),
            sp_stats.skew(lead, axis=1), sp_stats.kurtosis(lead, axis=1),
            np.sum(np.diff(np.sign(lead), axis=1) != 0, axis=1).astype(np.float64),
            np.max(lead, axis=1) - np.min(lead, axis=1),
            np.percentile(lead, 25, axis=1), np.percentile(lead, 75, axis=1),
            np.sqrt(np.mean(lead ** 2, axis=1)),
            np.mean(np.abs(lead - np.mean(lead, axis=1, keepdims=True)), axis=1),
        ]
        fft_vals = np.abs(np.fft.rfft(lead, axis=1))
        freqs = np.fft.rfftfreq(n_timesteps, d=1.0 / 100.0)
        features.append(np.sum(fft_vals ** 2, axis=1))
        for flo, fhi in [(0.5, 4), (4, 8), (8, 15), (15, 30), (30, 40)]:
            mask = (freqs >= flo) & (freqs < fhi)
            features.append(np.sum(fft_vals[:, mask] ** 2, axis=1))
        features.append(freqs[np.argmax(fft_vals[:, 1:], axis=1) + 1])
        psd = fft_vals ** 2
        psd_norm = psd / (psd.sum(axis=1, keepdims=True) + 1e-10)
        features.append(-np.sum(psd_norm * np.log(psd_norm + 1e-10), axis=1))
        features.append(np.sum(freqs * fft_vals, axis=1) / (np.sum(fft_vals, axis=1) + 1e-10))
        cumsum = np.cumsum(fft_vals ** 2, axis=1)
        features.append(freqs[np.argmax(cumsum >= 0.85 * (cumsum[:, -1:] + 1e-10), axis=1)])

    # Wavelet features (db4 level 4)
    for i in range(n_leads):
        lead = X_filt[:, :, i]
        energies = np.zeros((n_samples, 5))
        for s in range(n_samples):
            coeffs = pywt.wavedec(lead[s], 'db4', level=4)
            e = np.array([np.sum(c ** 2) for c in coeffs])
            energies[s] = e
        total = energies.sum(axis=1, keepdims=True) + 1e-10
        for lvl in range(5):
            features.append(energies[:, lvl])
            features.append(energies[:, lvl] / total[:, 0])

    # Inter-lead correlations
    for i, j in [(0, 1), (0, 6), (1, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]:
        li, lj = X_filt[:, :, i], X_filt[:, :, j]
        mi = np.mean(li, axis=1, keepdims=True)
        mj = np.mean(lj, axis=1, keepdims=True)
        num = np.sum((li - mi) * (lj - mj), axis=1)
        den = np.sqrt(np.sum((li - mi)**2, axis=1) * np.sum((lj - mj)**2, axis=1)) + 1e-10
        features.append(num / den)

    # Temporal segments
    seg = n_timesteps // 4
    for s in range(4):
        for i in range(n_leads):
            sd = X_filt[:, s*seg:(s+1)*seg, i]
            features += [np.mean(sd, axis=1), np.std(sd, axis=1)]

    # R-peak/HRV on lead II
    lead_ii = X_filt[:, :, 1]
    hr_feats = []
    for s in range(n_samples):
        sig = lead_ii[s]
        thresh = np.percentile(sig, 75)
        peaks, _ = find_peaks(sig, height=thresh, distance=40)
        if len(peaks) >= 2:
            rr = np.diff(peaks) / 100.0
            hr_feats.append([len(peaks), 60.0/np.mean(rr), np.std(rr)*1000,
                             np.sqrt(np.mean(np.diff(rr)**2))*1000])
        else:
            hr_feats.append([0.0, 0.0, 0.0, 0.0])
    hr_arr = np.array(hr_feats)
    for c in range(4):
        features.append(hr_arr[:, c])

    return np.column_stack(features)


def train_lgbm(F_train, y_train, F_test, classes):
    preds = {}
    for cls in classes:
        labels = y_train[cls].values
        n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
        spw = n_neg / max(n_pos, 1)
        print(f"  LGB {cls} (pos={int(n_pos)}, w={spw:.1f})...")
        m = lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        m.fit(F_train, labels)
        preds[cls] = m.predict_proba(F_test)[:, 1]
    return preds


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
    y_train_np = y_train[classes].values  # (N, 5)

    # ── LightGBM branch ─────────────────────────────────────────────────────
    print("Extracting features for LightGBM...")
    F_train = extract_features(X_train)
    F_test  = extract_features(X_test)
    print(f"  {F_train.shape[1]} features")
    F_train = np.nan_to_num(F_train, nan=0.0, posinf=0.0, neginf=0.0)
    F_test  = np.nan_to_num(F_test,  nan=0.0, posinf=0.0, neginf=0.0)
    lgb_preds = train_lgbm(F_train, y_train, F_test, classes)

    # ── CNN branch ───────────────────────────────────────────────────────────
    print("Training 1D ResNet...")
    cnn_preds_arr = train_cnn(X_train, y_train_np, X_test, n_epochs=20)
    cnn_preds = {cls: cnn_preds_arr[:, i] for i, cls in enumerate(classes)}

    # ── Ensemble (equal weight average) ─────────────────────────────────────
    predictions = {cls: 0.5 * lgb_preds[cls] + 0.5 * cnn_preds[cls] for cls in classes}

    pred_df = pd.DataFrame(predictions)
    pred_df.insert(0, "ecg_id", test_ids)
    pred_df.to_csv("predictions/predictions.csv", index=False)
    print(f"Done. {len(pred_df)} rows saved.")


if __name__ == "__main__":
    main()
