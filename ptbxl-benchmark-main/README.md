# PTB-XL 12-Lead ECG Classification Benchmark

A multi-label ECG diagnostic classification benchmark for AI agent competition. Agents compete to maximize macro-averaged AUROC across 5 clinically meaningful diagnostic superclasses using the PTB-XL dataset — the largest freely available clinical 12-lead ECG dataset.

## Why This Matters

Cardiovascular disease is the leading cause of death globally, killing ~17.9 million people annually. The 12-lead electrocardiogram (ECG) is the most widely used cardiac diagnostic tool — over 300 million ECGs are recorded per year worldwide. Yet ECG interpretation requires years of specialist training, and even experienced cardiologists disagree on ~20% of diagnoses.

Automated ECG interpretation has direct clinical value in:
- **Triage**: prioritizing critical ECGs in emergency departments
- **Screening**: population-level cardiac health screening
- **Low-resource settings**: enabling diagnosis where cardiologists are scarce
- **Edge deployment**: on-device interpretation in wearable/portable ECG devices

This benchmark tests whether AI agents can build models that accurately classify ECGs into 5 major diagnostic categories, using only the raw 12-lead signal and standard ML techniques.

## The Dataset: PTB-XL

**Source**: [PhysioNet PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
**Paper**: Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset" ([Nature Scientific Data, 2020](https://www.nature.com/articles/s41597-020-0495-6))
**License**: Creative Commons Attribution 4.0 International

| Property | Value |
|----------|-------|
| Total ECGs | 21,799 |
| Patients | 18,869 |
| Duration | 10 seconds per ECG |
| Leads | 12 standard (I, II, III, aVR, aVL, aVF, V1–V6) |
| Sample rate | 100 Hz (used here) and 500 Hz (available) |
| Labels | Multi-label, cardiologist-annotated |
| Split | Patient-wise stratified 10-fold |

PTB-XL was collected at the PTB (Physikalisch-Technische Bundesanstalt) from clinical routine ECGs. Each ECG was annotated by up to two cardiologists using SCP-ECG diagnostic statements, then validated by a senior cardiologist.

## The 5 Diagnostic Superclasses

| Superclass | Full Name | Clinical Meaning | Prevalence (train) |
|------------|-----------|-------------------|-------------------|
| **NORM** | Normal ECG | No pathological findings — normal sinus rhythm, normal axis, no ST/T abnormalities | 33.0% |
| **MI** | Myocardial Infarction | Evidence of heart attack — pathological Q waves, ST elevation/depression, T wave inversions indicating myocardial damage | 13.9% |
| **STTC** | ST/T Change | Non-specific ST segment depression/elevation or T wave changes — can indicate ischemia, electrolyte imbalance, or drug effects | 21.0% |
| **CD** | Conduction Disturbance | Abnormal electrical conduction — bundle branch blocks (RBBB, LBBB), AV blocks, fascicular blocks | 21.9% |
| **HYP** | Hypertrophy | Chamber enlargement — left/right ventricular hypertrophy, atrial enlargement, indicated by high voltages and axis deviation | 6.9% |

**Multi-label**: A single ECG can have multiple diagnoses (e.g., MI + STTC is common, as myocardial infarction causes ST/T changes). Approximately 22% of ECGs have more than one superclass label.

## Evaluation

**Metric**: Macro-averaged AUROC across the 5 superclasses.

Each class is scored independently as a binary AUROC (area under the receiver operating characteristic curve). The aggregate score is the unweighted mean of the 5 per-class AUROCs. This metric is:
- **Threshold-independent**: no need to choose a classification threshold
- **Imbalance-robust**: gives equal weight to each class regardless of prevalence
- **Standard**: matches the evaluation protocol in the original PTB-XL benchmark paper

**Split protocol**: The official PTB-XL patient-wise stratified split:
- **Folds 1–8**: Training (17,418 ECGs)
- **Fold 9**: Local evaluation (2,183 ECGs) — scored by `eval/eval.sh`
- **Fold 10**: Hidden holdout (2,198 ECGs) — **not included in repo**, reserved for final ranking

No patient appears in multiple splits, preventing data leakage. Agents may split folds 1–8 internally for their own train/validation needs.

**Two-tier scoring**: Local scores (from `eval/eval.sh`) reflect fold 9 performance. Final competition ranking uses the hidden fold 10 holdout, evaluated externally. This prevents leaderboard gaming via label peeking.

## Baseline Results

The baseline uses per-lead statistical features (mean, std, min, max, skew, kurtosis, zero crossings, peak-to-peak = 96 features) with XGBoost classifiers trained independently per class.

| Superclass | AUROC | Notes |
|------------|-------|-------|
| NORM | 0.8376 | Reasonably well-detected by basic amplitude/variance features |
| MI | 0.8036 | Weakest — MI requires detecting subtle Q waves and ST morphology |
| STTC | 0.8079 | Weak — ST/T changes need waveform shape, not just statistics |
| CD | 0.8206 | Conduction blocks have distinctive duration/timing features |
| HYP | 0.8681 | Best — hypertrophy produces high-amplitude signals easily captured by max/peak-to-peak |
| **Aggregate** | **0.8276** | |

**Runtime**: ~8.5 seconds (CPU), making rapid iteration feasible.

**Published state-of-the-art**: ~0.93 macro-AUROC with deep learning approaches (1D CNN/ResNet on raw signals), leaving significant room for improvement (~10+ percentage points).

## Where the Baseline Is Weak

The baseline only uses 8 statistical features per lead. It completely misses:

1. **Waveform morphology**: The shape of P waves, QRS complexes, and T waves carries critical diagnostic information that statistics can't capture
2. **Frequency content**: Different pathologies have characteristic frequency signatures (e.g., high-frequency notching in bundle branch blocks)
3. **Temporal patterns**: R-R interval variability, QRS duration, QT interval — these durations are core to ECG interpretation
4. **Inter-lead relationships**: Axis deviation (comparing limb leads), reciprocal ST changes (anterior vs inferior leads), and lead-specific patterns
5. **Multi-label correlations**: MI and STTC often co-occur; the baseline treats each class independently

## Improvement Strategies

### Feature Engineering (fast iteration, ~10–30s)
- Add frequency-domain features (FFT, power spectral density per band)
- Wavelet coefficients (DWT with Daubechies wavelets — standard for ECG analysis)
- R-peak detection and heart rate variability features
- QRS duration and morphology features
- Inter-lead correlation matrix features
- RMS voltage per lead, signal energy, spectral entropy

### Model Improvements (moderate, ~30–60s)
- LightGBM/CatBoost (often better than XGBoost on tabular data)
- Multi-label classifiers that capture label correlations (classifier chains, label powerset)
- Stacking ensembles (multiple feature sets → meta-learner)
- Target-aware feature selection per class

### Deep Learning on Raw Signals (slower iteration, ~2–10 min on CPU)
- 1D ResNet (the standard PTB-XL baseline architecture)
- 1D InceptionTime
- SE-ResNet with squeeze-excitation blocks
- Multi-scale CNN (parallel convolutions at different kernel sizes)
- Transformer or attention mechanisms on temporal features

### Signal Processing
- Bandpass filtering (0.5–40 Hz) to remove baseline wander and high-frequency noise
- Notch filter at 50/60 Hz for powerline interference
- R-peak alignment and beat-level feature extraction
- Continuous wavelet transform for time-frequency representation

## Repository Structure

```
ptbxl-benchmark/
├── train.py              # Training pipeline (agents modify this)
├── eval/
│   ├── eval.sh           # Evaluation entry point (read-only)
│   └── evaluate.py       # Scoring logic (read-only)
├── prepare.sh            # Downloads PTB-XL, preprocesses to numpy (read-only)
├── program.md            # Agent instructions
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── .gitignore
```

After running `bash prepare.sh`:
```
data/
├── X_train.npy           # (17418, 1000, 12) float32 — training signals (folds 1–8)
├── X_test.npy            # (2183, 1000, 12) float32 — evaluation signals (fold 9)
├── y_train.csv           # Training labels (ecg_id + 5 binary columns)
├── y_test_ids.csv        # Evaluation ECG IDs (for predictions — include ecg_id in output)
├── y_test_labels.csv     # Evaluation labels (for eval only — DO NOT use in training)
├── registry.json         # Metadata (classes, split sizes, signal properties)
├── ptbxl_database.csv    # Original PTB-XL metadata
├── scp_statements.csv    # SCP code definitions
└── records100/           # Raw WFDB files at 100Hz (for advanced agents)
```

## Quickstart

```bash
# 1. Clone and enter directory
git clone <repo-url> && cd ptbxl-benchmark

# 2. Download data and install dependencies (~10 min first time)
bash prepare.sh

# 3. Run baseline evaluation
bash eval/eval.sh
# Expected output: score: 0.8276

# 4. Modify train.py to improve the score
# 5. Re-run eval/eval.sh to check your improvements
```

## Requirements

- Python 3.10+
- ~2 GB disk space for data
- ~2 GB RAM for loading signal arrays
- CPU sufficient (GPU optional for deep learning approaches)

## Data Source

All data is from PhysioNet's PTB-XL dataset v1.0.3. Downloaded via AWS S3 mirror (`s3://physionet-open/ptb-xl/1.0.3/`) when `aws` CLI is available, otherwise via direct PhysioNet download.

**Citation**:
> Wagner, P., Strodthoff, N., Bousseljot, R.D., Kreiseler, D., Lunze, F.I., Samek, W., Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. Scientific Data, 7(1), 154.

## Rules

- Modify only `train.py` (and create new Python modules as needed)
- Do not read `data/y_test_labels.csv` in training code
- Do not modify `eval/`, `prepare.sh`, or files in `data/`
- No external data or pretrained models from external ECG datasets
- Output `predictions/predictions.csv` with columns `ecg_id`, NORM, MI, STTC, CD, HYP (ecg_id required)
