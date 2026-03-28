#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find best available Python (3.10+)
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &> /dev/null; then
        version=$("$candidate" -c "import sys; print(sys.version_info.minor)")
        if [ "$version" -ge 10 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10+ required."
    exit 1
fi

echo "=== Using $PYTHON ($($PYTHON --version)) ==="

# Install libomp on macOS (needed for XGBoost/LightGBM)
if [[ "$(uname)" == "Darwin" ]] && command -v brew &> /dev/null; then
    if ! brew list libomp &> /dev/null 2>&1; then
        echo "=== Installing libomp (required for XGBoost on macOS) ==="
        brew install libomp
    fi
fi

# Create venv if needed
if [ ! -d ".venv" ]; then
    echo "=== Creating virtual environment ==="
    if command -v uv &> /dev/null; then
        uv venv --python "$PYTHON" .venv
    else
        "$PYTHON" -m venv .venv
    fi
fi

source .venv/bin/activate

echo "=== Installing dependencies ==="
if command -v uv &> /dev/null; then
    uv pip install -r requirements.txt
else
    pip install -r requirements.txt
fi

echo ""
echo "=== Downloading PTB-XL from PhysioNet ==="

mkdir -p data

# Download the dataset (~3 GB uncompressed, freely available, no credentials)
if [ ! -f "data/ptbxl_database.csv" ]; then
    if command -v aws &> /dev/null; then
        echo "  Downloading PTB-XL v1.0.3 via AWS S3 (fastest)..."
        aws s3 sync --no-sign-request \
            s3://physionet-open/ptb-xl/1.0.3/ data/ \
            --exclude "records500/*"
    else
        echo "  Downloading PTB-XL v1.0.3 via PhysioNet (~1.8 GB zip)..."
        ZIPFILE="data/ptbxl.zip"
        curl -L -o "$ZIPFILE" \
            "https://physionet.org/content/ptb-xl/get-zip/1.0.3/"

        echo "  Extracting..."
        unzip -q "$ZIPFILE" -d data/

        # Move files from the nested directory
        NESTED="data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
        if [ -d "$NESTED" ]; then
            cp -r "$NESTED"/* data/
            rm -rf "$NESTED"
        fi

        rm -f "$ZIPFILE"
    fi
else
    echo "  PTB-XL data already present"
fi

# Verify key files exist
for f in data/ptbxl_database.csv data/scp_statements.csv; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f after extraction"
        exit 1
    fi
done

if [ ! -d "data/records100" ]; then
    echo "ERROR: Missing data/records100/ directory"
    exit 1
fi

echo ""
echo "=== Converting WFDB signals to numpy arrays + creating splits ==="
python3 << 'PYEOF'
import ast
import json
import os

import numpy as np
import pandas as pd
import wfdb

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# ── Load metadata ──
print("  Loading metadata...")
df = pd.read_csv("data/ptbxl_database.csv", index_col="ecg_id")
df.scp_codes = df.scp_codes.apply(ast.literal_eval)

scp = pd.read_csv("data/scp_statements.csv", index_col=0)
scp = scp[scp.diagnostic == 1.0]

# ── Map SCP codes → 5 superclass labels ──
def aggregate_superclass(scp_codes):
    labels = np.zeros(5, dtype=np.float32)
    for code, confidence in scp_codes.items():
        if confidence >= 100.0 and code in scp.index:
            sc = scp.loc[code, "diagnostic_class"]
            if sc in SUPERCLASSES:
                labels[SUPERCLASSES.index(sc)] = 1.0
    return labels

print("  Computing multi-label targets...")
label_matrix = np.array([aggregate_superclass(codes) for codes in df.scp_codes])

# ── Load 100Hz WFDB signals ──
print("  Loading 100Hz ECG signals (this takes a few minutes)...")
signals = []
for i, (ecg_id, row) in enumerate(df.iterrows()):
    record = wfdb.rdrecord(os.path.join("data", row.filename_lr.strip()))
    signals.append(record.p_signal)  # shape: (1000, 12)
    if (i + 1) % 5000 == 0:
        print(f"    Loaded {i + 1}/{len(df)} records")

X = np.array(signals, dtype=np.float32)  # (N, 1000, 12)
print(f"  Signal array shape: {X.shape}")

# ── Split by official strat_fold ──
# Folds 1-8: training.  Fold 9: local evaluation (scored by eval.sh).
# Fold 10: NOT included — reserved as hidden holdout for final ranking.
folds = df.strat_fold.values
train_mask = folds <= 8
eval_mask = folds == 9

ecg_ids = df.index.values

print(f"  Train (folds 1-8): {train_mask.sum()}")
print(f"  Eval  (fold 9):    {eval_mask.sum()}")
print(f"  Hidden (fold 10):  {(folds == 10).sum()} (not included in repo)")

# ── Save numpy signal arrays ──
print("  Saving numpy arrays...")
np.save("data/X_train.npy", X[train_mask])
np.save("data/X_test.npy", X[eval_mask])

# ── Save label CSVs ──
# Training labels (agents can read)
train_label_df = pd.DataFrame(label_matrix[train_mask], columns=SUPERCLASSES)
train_label_df.insert(0, "ecg_id", ecg_ids[train_mask])
train_label_df.to_csv("data/y_train.csv", index=False)

# Eval: ECG IDs for agents to include in predictions
eval_ids_df = pd.DataFrame({"ecg_id": ecg_ids[eval_mask]})
eval_ids_df.to_csv("data/y_test_ids.csv", index=False)

# Eval: labels for evaluate.py only (agents must NOT read)
eval_label_df = pd.DataFrame(label_matrix[eval_mask], columns=SUPERCLASSES)
eval_label_df.insert(0, "ecg_id", ecg_ids[eval_mask])
eval_label_df.to_csv("data/y_test_labels.csv", index=False)

# ── Save registry ──
registry = {
    "classes": SUPERCLASSES,
    "n_train": int(train_mask.sum()),
    "n_test": int(eval_mask.sum()),
    "n_leads": 12,
    "sample_rate_hz": 100,
    "duration_sec": 10,
    "samples_per_ecg": 1000,
    "train_folds": "1-8",
    "eval_fold": 9,
    "hidden_fold": 10,
}
with open("data/registry.json", "w") as f:
    json.dump(registry, f, indent=2)

# ── Print class distribution ──
print("\n  Class distribution (train):")
train_labels = label_matrix[train_mask]
for i, cls in enumerate(SUPERCLASSES):
    pos = int(train_labels[:, i].sum())
    pct = 100 * pos / len(train_labels)
    print(f"    {cls:6s}  {pos:5d}/{len(train_labels)}  ({pct:.1f}%)")

print("\n=== Data preparation complete ===")
PYEOF

echo ""
echo "=== Verifying ==="
python3 -c "
import numpy as np
for name in ['X_train', 'X_test']:
    arr = np.load(f'data/{name}.npy')
    print(f'  {name}: shape={arr.shape}, size={arr.nbytes/1e6:.0f} MB')
import pandas as pd
for name in ['y_train', 'y_test_ids', 'y_test_labels']:
    df = pd.read_csv(f'data/{name}.csv')
    print(f'  {name}: {len(df)} rows, columns={list(df.columns)}')
"

echo ""
echo "=== Setup complete ==="
