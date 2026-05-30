# Official CAMUS Dataset Split Implementation

## Overview

This project now implements **faithful reproduction of the original CAMUS experimental setup** using the official dataset splits provided by the dataset authors. All training/validation/testing uses the predetermined patient lists without custom random splitting by default.

## Official Split Composition

- **Training**: 400 patients (database_split/subgroup_training.txt)
- **Validation**: 50 patients (database_split/subgroup_validation.txt)
- **Testing**: 50 patients (database_split/subgroup_testing.txt)

**Total**: 500 patients with no overlap between splits.

## Critical: Patient-Level Grouping

### Why Patient-Level Splitting is Essential

Each CAMUS patient contributes **exactly 4 related frames** from the same cardiac acquisition:
- 2CH_ED (2-chamber, end-diastole)
- 2CH_ES (2-chamber, end-systole)
- 4CH_ED (4-chamber, end-diastole)
- 4CH_ES (4-chamber, end-systole)

### The Data Leakage Problem

**Frame-level splitting** (splitting by individual images) would:
1. Put spatially and temporally correlated frames from the same cardiac cycle into different splits
2. Create near-duplicate anatomy in train/validation/test sets
3. Cause feature leakage from training set to evaluation sets
4. Artificially inflate performance metrics (e.g., Dice scores appear higher than realistic)

**Patient-level splitting** ensures:
- Each patient's **entire cardiac data** stays together
- Train/validation/test sets contain truly independent cardiac cycles
- Metrics reflect realistic generalization performance
- Results are reproducible and comparable to published CAMUS studies

## Usage

### Default: Official Split Only (No Cross-Validation)

```bash
python train.py
```

Trains on official 400 training patients, validates on official 50 validation patients, and tests on official 50 testing patients.

**Output**: `runs/official_split/` with results and overlay visualizations.

### Optional: 10-Fold Patient-Level Cross-Validation

```bash
python train.py --cross-validate
```

- Performs 10-fold CV **only over official 400 training patients**
- Official 50 validation patients become the **held-out test set**
- Official 50 testing patients become the **external evaluation set**
- Each fold: train on ~360 patients, validate on ~40 patients, test on official 50 test patients

**Output**: `runs/cross_validation/` with per-fold results.

### Run Single Cross-Validation Fold

```bash
python train.py --cross-validate --fold 0
```

Runs only fold 0 (useful for distributed training).

### Full Cross-Validation Help

```bash
python train.py --help
```

Key arguments:
- `--cross-validate`: Enable optional 10-fold CV (default: disabled)
- `--folds N`: Number of CV folds (default: 10)
- `--fold N`: Run single fold N (0-based, default: -1 for all)
- `--split-root`: Path to database_split folder (default: database_split)
- `--nifti-root`: Path to database_nifti folder (default: database_nifti)
- `--output`: Results directory (default: runs/official_split)
- `--epochs`: Training epochs (default: 30)

## Implementation Details

### New Functions in `utils.py`

#### `load_patient_list(path: Path) -> List[str]`
Loads a single split file, strips whitespace, validates non-empty.

#### `load_official_camus_splits(split_root: Path) -> tuple[List[str], List[str], List[str]]`
- Loads all three official split files
- **Validates** 400/50/50 patient counts
- **Checks** for no overlap between splits
- Returns: (training_patients, validation_patients, testing_patients)
- Raises descriptive errors if split files are invalid or missing

### Updated `train.py`

#### Main Changes
1. **Imports** `load_official_camus_splits` from utils
2. **New arguments**:
   - `--split-root`: Path to official split files
   - `--cross-validate`: Enable optional 10-fold CV
3. **Default behavior**: Uses official splits without CV
4. **Optional CV mode**: Respects 10-fold CV when `--cross-validate` enabled

#### Training Workflow
1. **Load official splits** from text files
2. **Validate patient existence** in NIfTI directory
3. **Standard mode** (default):
   - Train: official 400 patients
   - Validate: official 50 validation patients
   - Test: official 50 testing patients
4. **CV mode** (with `--cross-validate`):
   - Train: ~360 patients per fold
   - Validate: ~40 patients per fold (from official training)
   - Test: official 50 testing patients
   - Results: `fold_00/`, `fold_01/`, ..., `fold_09/`

### Updated `dataset.py`

Enhanced docstring explaining patient-level grouping and data leakage risks in echocardiography segmentation.

## Example: Reproducing Official CAMUS Results

```bash
# Train using official splits (standard setup)
python train.py --epochs 50 --batch-size 4 --lr 1e-3

# Optional: 10-fold CV for robust evaluation
python train.py --cross-validate --epochs 50 --batch-size 4

# Check specific fold
python train.py --cross-validate --fold 2 --epochs 50 --batch-size 4
```

## Output Structure

### Standard Mode: `runs/official_split/`
```
fold_00/
  best_model.pt          # Best validation checkpoint
  loss_curve.png         # Training/validation loss curve
  dice_curve.png         # Training/validation Dice curve
  test_metrics.json      # Test set metrics (Dice, IoU, Hausdorff, MSD)
  overlays/
    sample_000.png       # Prediction overlay visualizations
    ...
run_summary.json         # Aggregated metrics summary
```

### Cross-Validation Mode: `runs/cross_validation/`
```
fold_00/ ... fold_09/    # Identical structure per fold
run_summary.json         # CV mean ± std statistics
```

## Metrics Reported

- **Dice**: Intersection-over-union type metric for segmentation quality
- **IoU**: Intersection-over-union (Jaccard index)
- **Hausdorff**: Maximum distance between boundary contours
- **MSD**: Mean surface distance between predictions and ground truth

## Notes

- Patient grouping is enforced at the **Dataset level** (see `CamusPatientDataset`)
- Each patient's four frames are always loaded together as separate samples within a training run
- The dataset does **not** split frames across train/val/test
- All augmentations respect the patient-level invariant
- Results are deterministic given a seed (`--seed`, default 42)

## References

Official CAMUS dataset: https://www.creatis.insa-lyon.fr/Challenge/camus/

CAMUS Study: Leclerc et al., "Deep Learning for Cardiac Segmentation", 2019
