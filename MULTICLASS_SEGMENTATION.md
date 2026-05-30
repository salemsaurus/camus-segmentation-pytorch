# CAMUS Multiclass Segmentation Implementation Guide

## Overview

The CAMUS U-Net1 training pipeline now supports **true multiclass segmentation** with faithful preservation of all 4 cardiac structure classes.

## CAMUS Multiclass Structure

The segmentation task has **4 classes**:
- **Class 0**: Background (no cardiac tissue)
- **Class 1**: LV cavity (left ventricle blood pool)
- **Class 2**: Myocardium (LV wall)
- **Class 3**: Left atrium (LA)

## Architecture & Loss Function

### Model Output
- **Shape**: `[B, 4, H, W]` (batch size, 4 classes, height, width)
- **Type**: Raw logits (no activation applied)
- **Purpose**: Input to CrossEntropyLoss

### Loss Function: CrossEntropyLoss
Located in [losses.py](losses.py)

**Multiclass training**:
```python
criterion = nn.CrossEntropyLoss(reduction="mean")
loss = criterion(logits, masks.long())
```

**Requirements**:
- Logits: `[B, 4, H, W]` (raw, no softmax)
- Masks: `[B, H, W]` with dtype `torch.long`
- Mask values: {0, 1, 2, 3}

## Metrics: Per-Class Evaluation

Updated in [metrics.py](metrics.py). All metrics compute **per-class** scores for classes 1-3, then report the mean:

### Dice Score
- Computed independently for each class (1, 2, 3)
- Binary Dice for each: `(2*TP) / (2*TP + FP + FN)`
- Result: Mean across all class scores

### IoU (Jaccard Index)
- Computed independently for each class
- Formula: `TP / (TP + FP + FN)`
- Result: Mean across all classes

### Hausdorff Distance
- Surface-based metric computed per-class
- Measures maximum distance between boundary points
- Result: Mean Hausdorff across classes

### Mean Surface Distance (MSD)
- Average distance between prediction and ground-truth surfaces
- Computed per-class
- Result: Mean MSD across classes

## Dataset Loading

[dataset.py](dataset.py): Multiclass mask loading

**Mask loading pipeline**:
1. Load NIfTI as float32
2. Convert to `torch.long` (class indices)
3. Interpolate with `mode="nearest"` (preserves class labels)
4. Return as `[H, W]` long tensor with values {0, 1, 2, 3}

**Why `nearest` mode**: Bilinear interpolation would create fractional class labels; nearest-neighbor preserves exact class boundaries.

## Inference & Predictions

### Prediction Pipeline

**In training/validation** ([train.py](train.py), [metrics.py](metrics.py)):
```python
# Raw logits from model
logits: [B, 4, H, W]

# Convert to class predictions
pred = torch.argmax(logits, dim=1)  # [B, H, W] with values {0,1,2,3}
```

### Key: No Binary Collapsing
**REMOVED**:
```python
pred = (pred > 0).astype(np.uint8)  # DON'T DO THIS
```

**Preserved**:
```python
pred = torch.argmax(logits[i : i + 1], dim=1).squeeze().cpu().numpy().astype(np.uint8)
# Keeps class labels: {0, 1, 2, 3} intact
```

## Post-Processing

Updated [utils.py](utils.py): `postprocess_prediction()`

**For multiclass masks**, each class is processed **independently**:

```python
final_mask = np.zeros_like(pred)

for cls in [1, 2, 3]:
    cls_mask = (pred == cls).astype(np.uint8)
    
    # Apply morphology to this class only
    if keep_largest:
        cls_mask = largest_connected_component(cls_mask)
    if fill_holes:
        cls_mask = remove_holes(cls_mask)
    
    # Assign back without overwriting other classes
    final_mask[cls_mask == 1] = cls

return final_mask
```

**Why independent processing**:
- Prevents morphology operations from merging class boundaries
- Preserves anatomical separation between LV, myocardium, and LA
- Maintains class label integrity

## Visualization: Multiclass Overlays

Updated [utils.py](utils.py): `save_overlay()`

**Multiclass overlay scheme**:
- **Red channel** (max 1.0): Class 1 (LV cavity)
- **Green channel** (max 1.0): Class 2 (Myocardium)
- **Blue channel** (max 1.0): Class 3 (Left atrium)
- **Grayscale background**: Original ultrasound image

**Intensity levels**:
- Ground truth: Full intensity (1.0)
- Predictions: Half intensity (0.5)
- Allows visual comparison of where prediction differs from GT

**Example colors**:
- Prediction-only LV: Red (0.5, 0, 0)
- GT+Pred myocardium: Bright green (1.0, 1.0, 0) if both present
- Atrium mismatch: Magenta-ish from overlapping red/blue

## Training Configuration

### Default Settings
```bash
python train.py
```

Uses:
- Official 400 training patients
- Official 50 validation patients
- Official 50 test patients
- 4 classes (num_classes=4 by default)
- CrossEntropyLoss for multiclass

### Command-line Arguments
```bash
python train.py \
  --epochs 50 \
  --batch-size 4 \
  --lr 1e-3 \
  --num-classes 4 \          # Always 4 for CAMUS multiclass
  --postprocess-eval \        # Enable post-processing
  --seed 42
```

## Output Structure

```
runs/training/
├── best_model.pt              # Best validation checkpoint
├── loss_curve.png             # Training/validation loss
├── dice_curve.png             # Training/validation Dice
├── test_metrics.json          # Test metrics (per-class Dice, IoU, etc.)
└── overlays/
    ├── sample_000.png         # Multiclass color overlay
    ├── sample_001.png
    └── ...
```

### Metrics File Format
```json
{
  "dice": 0.75,               # Mean Dice over classes 1-3
  "iou": 0.65,                # Mean IoU over classes 1-3
  "hausdorff": 15.2,          # Mean Hausdorff distance
  "msd": 2.1,                 # Mean surface distance
  "loss": 0.48
}
```

## Model Architecture

[model.py](model.py): CamusUnet1 with multiclass support

**Unchanged from original**:
- Bilinear upsampling (per CAMUS paper)
- Conv2d + ReLU only (no BatchNorm)
- Compact channel progression: 1 → 32 → 64 → 128
- Skip connections

**Multiclass adaptation**:
- Output layer: `nn.Conv2d(16, 4, kernel_size=1)` (4 classes)
- No post-hoc activation needed (CrossEntropyLoss expects raw logits)
- `forward()` returns raw logits for training
- Optional `predict()` applies softmax for inference

## Validation Checklist

✅ Loss function: CrossEntropyLoss (multiclass)
✅ Model output: [B, 4, H, W] logits
✅ Mask input: [B, H, W] long tensors with {0,1,2,3}
✅ Binary collapsing: Removed from predictions
✅ Metrics: Per-class Dice, IoU, Hausdorff, MSD
✅ Post-processing: Class-independent morphology
✅ Overlays: Multiclass RGB visualization
✅ Dataset: Patient-level grouping maintained
✅ Official split: 400/50/50 respected

## Backward Compatibility: Binary Segmentation

For experiments requiring **binary segmentation** (num_classes=1):

```bash
python train.py --num-classes 1
```

**Binary behavior**:
- Logits: `[B, 1, H, W]`
- Loss: BCEWithLogitsLoss
- Predictions: Sigmoid threshold at 0.5
- Metrics: Binary foreground/background

All binary code paths remain functional.

## Reproducing Original CAMUS Results

To train on official CAMUS splits with faithful multiclass evaluation:

```bash
python train.py \
  --epochs 50 \
  --batch-size 4 \
  --num-classes 4 \
  --postprocess-eval \
  --seed 42 \
  --output runs/camus_multiclass
```

Results will be saved with:
- Per-class metrics for each cardiac structure
- Color-coded visualization overlays
- Faithful reproduction of CAMUS multiclass experimental setup
