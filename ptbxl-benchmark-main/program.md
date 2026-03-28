# PTB-XL 12-Lead ECG Classification

Classify 12-lead ECGs into 5 diagnostic superclasses. Maximize macro-averaged AUROC.

## Setup

1. **Read the in-scope files**:
   - `train.py` — training and prediction pipeline. You modify this.
   - `eval/eval.sh` — runs evaluation. Do not modify.
   - `eval/evaluate.py` — scoring logic. Do not modify.
   - `prepare.sh` — downloads data and installs deps. Do not modify.
2. **Run prepare**: `bash prepare.sh` to download PTB-XL and create numpy splits.
3. **Verify data exists**: Check that `data/` contains `X_train.npy`, `X_test.npy`, label CSVs, and `registry.json`.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row.
5. **Run baseline**: `bash eval/eval.sh` to establish the starting score.

## The benchmark

This benchmark uses the **PTB-XL dataset** — the largest freely available clinical 12-lead ECG dataset — for multi-label diagnostic classification into 5 superclasses.

| Superclass | Full Name | Description | Prevalence (train) |
|------------|-----------|-------------|-------------------|
| NORM | Normal ECG | No pathological findings | 33.0% |
| MI | Myocardial Infarction | Heart attack indicators (ST elevation, Q waves) | 13.9% |
| STTC | ST/T Change | ST segment or T wave abnormalities | 21.0% |
| CD | Conduction Disturbance | Bundle branch blocks, AV blocks | 21.9% |
| HYP | Hypertrophy | Ventricular/atrial enlargement | 6.9% |

**Total: 21,799 ECGs from 18,869 patients.** Labels are multi-label — a patient can have multiple conditions simultaneously (e.g., MI + STTC). Each ECG is a 10-second, 12-lead recording at 100Hz (shape: 1000 timesteps × 12 leads).

**Split:** Official PTB-XL patient-wise stratified folds. Folds 1–8 = train (17,418 ECGs), fold 9 = evaluation (2,183 ECGs). Fold 10 is reserved as a hidden holdout for final ranking and is not included in this repo. No patient appears in multiple splits.

**Scoring:** Your local score (from `eval/eval.sh`) is computed on fold 9. Final competition ranking uses the hidden fold 10 holdout, evaluated externally.

## Data format

| File | Shape/Format | Description |
|------|-------------|-------------|
| `data/X_train.npy` | (17418, 1000, 12) float32 | Training ECG signals (folds 1–8) |
| `data/X_test.npy` | (2183, 1000, 12) float32 | Evaluation ECG signals (fold 9) |
| `data/y_train.csv` | ecg_id, NORM, MI, STTC, CD, HYP | Training labels (0/1) |
| `data/y_test_ids.csv` | ecg_id | Evaluation ECG IDs (no labels) |
| `data/y_test_labels.csv` | ecg_id + labels | **DO NOT read** in train.py |

The 12 ECG leads are ordered: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.

## Experimentation

**What you CAN do:**
- Modify `train.py` and create new Python modules
- Feature engineering on raw ECG signals:
  - Statistical features (mean, std, skew, kurtosis per lead)
  - Frequency-domain features (FFT magnitudes, power spectral density)
  - Wavelet features (discrete wavelet transform coefficients)
  - ECG-specific features (R-peak detection, QRS duration, heart rate variability)
  - Lead-to-lead features (inter-lead correlations, axis deviations)
- Try different models: XGBoost, LightGBM, CatBoost, Random Forest, SVM, MLP
- Multi-label learning: exploit correlations between the 5 classes
- Deep learning on raw signals: 1D CNN, 1D ResNet, LSTM, Transformer (PyTorch)
- Signal preprocessing: bandpass filtering, baseline wander removal, normalization
- Data augmentation: random cropping, time shifting, amplitude scaling, lead dropout, noise injection
- Split folds 1–8 internally for your own train/validation if needed
- Add packages to `requirements.txt`

**What you CANNOT do:**
- Modify `eval/`, `prepare.sh`, or anything in `data/`
- Read or use `data/y_test_labels.csv` in your training code
- Use external data beyond the provided PTB-XL dataset
- Use pretrained models trained on external ECG data

**The goal: maximize `score`.** The score is the macro-averaged AUROC across the 5 diagnostic superclasses. Each class produces a binary AUROC; the aggregate is their unweighted mean. Higher is better.

**Simplicity criterion**: All else being equal, simpler is better.

## Output format

`train.py` must output `predictions/predictions.csv` with columns `ecg_id`, `NORM`, `MI`, `STTC`, `CD`, `HYP`. The `ecg_id` column is required — the evaluator merges predictions with labels by ID. Each class column contains prediction probabilities.

```
---
score:            0.8276
classes:          5/5
CD:  0.8206
HYP:  0.8681
MI:  0.8036
NORM:  0.8376
STTC:  0.8079
```

## Logging results

Log each experiment to `results.tsv` (tab-separated):

```
commit	score	cost_usd	status	description
a1b2c3d	0.8276	0.00	keep	baseline
b2c3d4e	0.8410	0.25	keep	added FFT features
c3d4e5f	0.8300	0.15	discard	tried LightGBM, worse than XGBoost
```

## The experiment loop

LOOP FOREVER:

1. **THINK** — decide what to try next. Review results.tsv. Consider: feature engineering (frequency, wavelet, ECG-specific), model selection, multi-label strategies, signal preprocessing, or deep learning on raw signals.
2. Modify the in-scope files with your experimental idea.
3. git commit
4. Run the experiment: `bash eval/eval.sh > run.log 2>&1`
5. Read the results: `grep "^score:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` for the stack trace and attempt a fix.
7. **Review artifacts**: Check `predictions/predictions.csv` for sanity (correct shape, reasonable probabilities). Check per-class scores in `run.log` to identify which classes improved or regressed.
8. Record the results in results.tsv (do not commit results.tsv).
9. If score improved, keep the git commit. If equal or worse, `git reset --hard HEAD~1`.

**Timeout**: If a run exceeds 10 minutes, kill it and treat it as a failure. The baseline runs in ~10 seconds.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. The loop runs until interrupted.
