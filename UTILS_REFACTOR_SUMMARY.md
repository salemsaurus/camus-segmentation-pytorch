# `utils.py` Refactor Summary

## Overview
Comprehensive refactoring of `utils.py` for improved evaluation reliability, visualization quality, and post-processing safety for multiclass cardiac segmentation.

---

## 1. ✅ Removed Myocardium-LV Dilation Constraint

**What Changed:**
- Removed the line: `myocardium_mask = myocardium_mask & lv_dilation`
- This constraint artificially limited myocardium predictions to be adjacent to the LV

**Why:**
- Not part of the original network output
- Could reduce myocardium Dice scores
- Could hide segmentation errors
- Introduced incorrect anatomical assumptions during evaluation

**Current Behavior:**
Only overlaps are removed:
```python
myocardium_mask = myocardium_mask & ~lv_mask
atrium_mask = atrium_mask & ~(lv_mask | myocardium_mask)
```

**Impact:** Evaluation now faithfully reflects model predictions without artificial constraints.

---

## 2. ✅ Improved Multiclass Post-processing Configuration

**New Feature: `POSTPROCESS_CLASSES`**

```python
POSTPROCESS_CLASSES = {
    1: True,   # LV: apply largest_connected_component
    2: False,  # Myocardium: skip (ring-shaped structure)
    3: True,   # LA: apply largest_connected_component
}
```

**Rationale:**
- **LV (class 1):** Often has spurious components from noise → benefits from largest component filtering
- **Myocardium (class 2):** Ring-shaped structure → aggressive component filtering damages topology
- **LA (class 3):** Usually compact → benefits from largest component filtering

**Usage:**
Modify `POSTPROCESS_CLASSES` dictionary to experiment with different post-processing strategies without code changes.

---

## 3. ✅ Contour Overlay Visualization

**New Function: `save_contour_overlay(image, gt, pred, out_path)`**

Features:
- Grayscale ultrasound background preserved
- **Ground truth contours:** solid lines
- **Prediction contours:** dashed lines
- **Class colors:**
  - LV: Red
  - Myocardium: Green
  - LA: Blue

**Usage:**
```python
from utils import save_contour_overlay
save_contour_overlay(image, gt_mask, pred_mask, Path("contour_overlay.png"))
```

**Benefits:**
- Easier visual comparison of GT vs prediction
- Easier error inspection
- Preserves ultrasound texture for clinical context

---

## 4. ✅ Three-Panel Comparison Figure

**New Function: `save_comparison_figure(image, gt, pred, out_path)`**

Generates a PNG with three panels:
1. **Panel 1:** Ultrasound image (grayscale)
2. **Panel 2:** Ground truth segmentation (colored)
3. **Panel 3:** Prediction segmentation (colored)

Includes color legend:
- LV Cavity (Red)
- Myocardium (Green)
- Left Atrium (Blue)

**Usage:**
```python
from utils import save_comparison_figure
save_comparison_figure(image, gt_mask, pred_mask, Path("comparison.png"))
```

**Benefits:**
- Single figure for publication or reports
- Side-by-side evaluation
- Automated batch generation

---

## 5. ✅ Error Map Visualization

**New Function: `save_error_map(image, gt, pred, out_path)`**

Color-coded error highlighting:
- **Red:** False Negatives (GT has class, pred doesn't)
- **Blue:** False Positives (Pred has class, GT doesn't)
- **Green:** Correct (GT and pred agree)
- **Gray background:** Ultrasound image

**Usage:**
```python
from utils import save_error_map
save_error_map(image, gt_mask, pred_mask, Path("error_map.png"))
```

**Benefits:**
- Immediately identify failure regions
- Distinguish FN from FP at a glance
- Useful for error analysis and debugging

---

## 6. ✅ Batch Visualization Export

**New Function: `save_batch_visualizations(images, gt_masks, pred_masks, output_dir)`**

For each sample in a batch, saves 4 PNG files:
1. `overlay.png` – Direct mask overlay
2. `contour_overlay.png` – Contour comparison (GT solid, Pred dashed)
3. `comparison.png` – Three-panel figure
4. `error_map.png` – FN/FP/correct highlighting

**Directory Structure:**
```
output_dir/
  sample_0000/
    overlay.png
    contour_overlay.png
    comparison.png
    error_map.png
  sample_0001/
    ...
```

**Usage:**
```python
from utils import save_batch_visualizations
from pathlib import Path

save_batch_visualizations(
    images=test_images,          # (N, H, W)
    gt_masks=test_gt_masks,      # (N, H, W)
    pred_masks=test_pred_masks,  # (N, H, W)
    output_dir=Path("test_visualizations")
)
```

**Useful for:**
- Test-set review after training
- Error analysis
- Publication figures
- Model debugging

---

## 7. ✅ Improved Reproducibility

**Enhanced `seed_everything(seed: int)`**

Now sets:
- ✅ `os.environ["PYTHONHASHSEED"]` – hash randomization control
- ✅ `random.seed(seed)` – Python standard library
- ✅ `np.random.seed(seed)` – NumPy
- ✅ `torch.manual_seed(seed)` – PyTorch CPU
- ✅ `torch.cuda.manual_seed_all(seed)` – PyTorch GPU
- ✅ `torch.backends.cudnn.deterministic = True`
- ✅ `torch.backends.cudnn.benchmark = False`
- ✅ `torch.use_deterministic_algorithms(True)` (PyTorch 1.12+)

**Usage:**
```python
from utils import seed_everything

seed_everything(42)  # Set all RNG seeds for reproducibility
```

**Benefits:**
- Deterministic training/validation
- Reproducible across runs
- Useful for debugging and publication

---

## 8. ✅ Metric History Export

**New Function: `save_training_history(history, out_json)`**

Exports training metrics to JSON for offline analysis.

**Usage:**
```python
from utils import save_training_history

history = {
    'train_loss': [0.5, 0.4, 0.3, ...],
    'val_loss': [0.6, 0.5, 0.4, ...],
    'train_dice': [0.7, 0.75, 0.8, ...],
    'val_dice': [0.65, 0.7, 0.75, ...],
}
save_training_history(history, Path("training_history.json"))
```

**Benefits:**
- Analyze metrics without re-training
- Create custom plots
- Compare across training runs
- Share reproducible results

---

## 9. ✅ Code Quality Improvements

### Type Hints
All functions now include type hints:
```python
def save_error_map(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
) -> None:
```

### Docstrings
All functions have comprehensive docstrings with:
- Description
- Args section
- Returns section
- Raises section (where applicable)
- Usage examples

### Backward Compatibility
✅ **No breaking changes** – all existing functions preserved:
- ✅ `list_patients()`
- ✅ `load_patient_list()`
- ✅ `load_official_camus_splits()`
- ✅ `largest_connected_component()`
- ✅ `remove_holes()`
- ✅ `postprocess_prediction()` – same API, improved implementation
- ✅ `save_loss_curves()`
- ✅ `save_dice_curves()`
- ✅ `save_overlay()` – same API, now with better documentation
- ✅ `save_json()`
- ✅ `split_patients_validation()`
- ✅ `seed_everything()` – enhanced, same API

### Imports
All necessary imports added:
```python
import os
import random
from typing import Dict, Optional, Tuple
import matplotlib.patches as mpatches
```

---

## Summary of New Functions

| Function | Purpose |
|----------|---------|
| `save_contour_overlay()` | Contour-based comparison visualization |
| `save_comparison_figure()` | Three-panel side-by-side figure |
| `save_error_map()` | FN/FP/correct highlighting |
| `save_batch_visualizations()` | Batch export all 4 visualization types |
| `save_training_history()` | Export metrics to JSON |

---

## Integration with Existing Code

### `train.py`
No changes required. Existing `train.py` will work unchanged.

Optional enhancements:
```python
# Export visualizations for test set
from utils import save_batch_visualizations
save_batch_visualizations(
    test_images, test_gt_masks, test_pred_masks,
    output_dir=Path(output_fold) / "visualizations"
)

# Export training history
from utils import save_training_history
save_training_history(history, Path(output_fold) / "training_history.json")
```

### `inference.py`
No changes required. Existing post-processing will use new `POSTPROCESS_CLASSES` configuration automatically.

### `dataset.py`
No changes required.

### `metrics.py`
No changes required. Benefits from improved post-processing reliability.

---

## Testing the Refactor

Quick validation:
```bash
python -c "
import utils
print('✓ Module imported')
print('✓ POSTPROCESS_CLASSES:', utils.POSTPROCESS_CLASSES)
print('✓ New functions available:')
for fn in ['save_contour_overlay', 'save_comparison_figure', 'save_error_map', 'save_batch_visualizations', 'save_training_history']:
    print(f'  - {fn}: {hasattr(utils, fn)}')
"
```

Expected output:
```
✓ Module imported
✓ POSTPROCESS_CLASSES: {1: True, 2: False, 3: True}
✓ New functions available:
  - save_contour_overlay: True
  - save_comparison_figure: True
  - save_error_map: True
  - save_batch_visualizations: True
  - save_training_history: True
```

---

## Next Steps

1. **Use `save_batch_visualizations()` for test-set review**
   ```python
   # After training, generate visualizations for all test samples
   save_batch_visualizations(test_images, test_gt, test_pred, Path("runs/fold_0/test_visualizations"))
   ```

2. **Monitor myocardium Dice changes**
   - The dilation constraint removal may affect myocardium metrics
   - Track whether this reveals previously hidden errors

3. **Experiment with `POSTPROCESS_CLASSES`**
   - Modify the dictionary to test different post-processing strategies
   - No code changes needed – just update the values

4. **Use error maps for debugging**
   - Identify systematic FN/FP patterns
   - Inform data augmentation and loss function adjustments

5. **Export training history for analysis**
   - Create custom plots across multiple folds
   - Identify training dynamics and overfitting patterns

---

## Summary

**Improvements delivered:**
- ✅ Removed anatomical constraints that hid model errors
- ✅ Configurable per-class post-processing
- ✅ 4 new visualization functions (contour, comparison, error map, batch export)
- ✅ Enhanced reproducibility with PYTHONHASHSEED
- ✅ Metric history export for offline analysis
- ✅ Full type hints and docstrings
- ✅ Zero breaking changes – backward compatible

**Result:** More reliable evaluation, better error visualization, and improved reproducibility for your CAMUS multiclass segmentation pipeline.
