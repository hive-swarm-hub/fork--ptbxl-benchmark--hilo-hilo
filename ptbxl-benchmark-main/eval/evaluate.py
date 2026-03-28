"""
Evaluation script for PTB-XL 12-Lead ECG Classification.

Computes per-class AUROC for 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP),
then reports the macro-averaged AUROC as the aggregate score.

Predictions are merged by ecg_id to prevent row-order corruption.
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def evaluate():
    with open("data/registry.json") as f:
        registry = json.load(f)

    classes = registry["classes"]

    # Load predictions and labels
    try:
        preds = pd.read_csv("predictions/predictions.csv")
        labels = pd.read_csv("data/y_test_labels.csv")
    except FileNotFoundError as e:
        print(f"  MISSING: {e}")
        print()
        print("---")
        print("score:            0.0000")
        return

    # Require ecg_id column for identity-based merging
    if "ecg_id" not in preds.columns:
        print("  ERROR: predictions.csv must contain an 'ecg_id' column")
        print()
        print("---")
        print("score:            0.0000")
        return

    # Merge predictions with labels by ecg_id
    merged = labels.merge(preds, on="ecg_id", how="left", suffixes=("_true", "_pred"))

    n_missing = merged[[f"{cls}_pred" for cls in classes]].isna().any(axis=1).sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing}/{len(labels)} test ECGs have no prediction")

    n_extra = len(preds) - len(preds[preds["ecg_id"].isin(labels["ecg_id"])])
    if n_extra > 0:
        print(f"  WARNING: {n_extra} predictions have unrecognized ecg_id (ignored)")

    class_scores = {}
    for cls in classes:
        true_col = f"{cls}_true"
        pred_col = f"{cls}_pred"

        if pred_col not in merged.columns:
            print(f"  MISSING column: {cls}")
            class_scores[cls] = 0.0
            continue

        # Drop rows where prediction is missing
        valid = merged[[true_col, pred_col]].dropna()
        y_true = valid[true_col].values
        y_pred = valid[pred_col].values

        if len(y_true) == 0:
            print(f"  WARNING: {cls} has no valid predictions")
            class_scores[cls] = 0.0
            continue

        # Need both positive and negative examples for AUROC
        if len(np.unique(y_true)) < 2:
            print(f"  WARNING: {cls} has only one class in test set")
            class_scores[cls] = 0.5
            continue

        try:
            score = roc_auc_score(y_true, y_pred)
        except ValueError:
            score = 0.5

        class_scores[cls] = score
        print(f"  {cls:8s}  AUROC  {score:.4f}")

    aggregate = np.mean(list(class_scores.values()))
    n_scored = sum(1 for s in class_scores.values() if s > 0)

    print()
    print("---")
    print(f"score:            {aggregate:.4f}")
    print(f"classes:          {n_scored}/{len(classes)}")
    for name, score in sorted(class_scores.items()):
        print(f"{name}:  {score:.4f}")


if __name__ == "__main__":
    evaluate()
