# Training Runtime Tracking

## Overview

`train.py` now tracks and reports comprehensive training runtime statistics.

---

## Features Added

### 1. **Per-Epoch Timing**
- Each epoch execution time is recorded
- Displayed in console output: `time=XX.XXs`

### 2. **Training Summary Report**
After training completes, a timing summary is displayed:
```
======================================================================
TRAINING TIME SUMMARY
======================================================================
Total training time:  1234.5s (20.6m / 0.34h)
Average epoch time:   41.15s
Min epoch time:       38.92s
Max epoch time:       47.23s
Test evaluation time: 12.35s
Total time (incl. test): 1246.9s (20.8m)
======================================================================
```

### 3. **Timing Statistics Saved to JSON**
File: `runs/training/fold_0/training_time.json`

Contents:
```json
{
  "total_training_time_seconds": 1234.5,
  "total_training_time_minutes": 20.57,
  "total_training_time_hours": 0.343,
  "average_epoch_time_seconds": 41.15,
  "min_epoch_time_seconds": 38.92,
  "max_epoch_time_seconds": 47.23,
  "test_evaluation_time_seconds": 12.35,
  "num_epochs": 30
}
```

---

## What Changed in `train.py`

### Import Added
```python
import time
```

### Epoch Loop Modified
```python
# Track training time
epoch_times = []
training_start_time = time.time()

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    
    # ... training code ...
    
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)
    
    # ... print epoch_time ...
```

### Test Evaluation Timing
```python
test_start_time = time.time()
test_metrics = run_epoch(...)
test_time = time.time() - test_start_time
```

### Statistics Computed and Saved
```python
timing_info = {
    "total_training_time_seconds": training_total_time,
    "total_training_time_minutes": training_total_time / 60.0,
    "total_training_time_hours": training_total_time / 3600.0,
    "average_epoch_time_seconds": avg_epoch_time,
    "min_epoch_time_seconds": min_epoch_time,
    "max_epoch_time_seconds": max_epoch_time,
    "test_evaluation_time_seconds": test_time,
    "num_epochs": args.epochs,
}
save_json(fold_dir / "training_time.json", timing_info)
```

---

## Console Output Example

When running training:
```
epoch 001/030 | train loss=0.5234 dice=0.6789 | val loss=0.4856 dice=0.7145 | time=41.23s
epoch 002/030 | train loss=0.4892 dice=0.7012 | val loss=0.4523 dice=0.7389 | time=40.89s
epoch 003/030 | train loss=0.4561 dice=0.7234 | val loss=0.4289 dice=0.7598 | time=42.15s
...
epoch 030/030 | train loss=0.2134 dice=0.8923 | val loss=0.2456 dice=0.8834 | time=41.05s

======================================================================
TRAINING TIME SUMMARY
======================================================================
Total training time:  1234.5s (20.6m / 0.34h)
Average epoch time:   41.15s
Min epoch time:       38.92s
Max epoch time:       47.23s
Test evaluation time: 12.35s
Total time (incl. test): 1246.9s (20.8m)
======================================================================
```

---

## Output Files

### `training_time.json`
Saved to: `runs/training/fold_0/training_time.json`

Contains timing statistics in JSON format for:
- Programmatic analysis
- Comparison across runs
- Graphing and reporting

### Console Output
Printed to terminal at the end of training with a formatted summary table.

---

## Use Cases

### 1. **Estimate Future Training Times**
```python
import json

with open("runs/training/fold_0/training_time.json") as f:
    timing = json.load(f)

avg_time = timing["average_epoch_time_seconds"]
num_epochs_new = 50
estimated_time_hours = (avg_time * num_epochs_new) / 3600
print(f"Estimated time for 50 epochs: {estimated_time_hours:.2f}h")
```

### 2. **Compare Training Across Environments**
```bash
# Local GPU training
python train.py --epochs 30  # saves training_time.json

# Colab GPU training
# Compare the training_time.json files to see GPU speedup
```

### 3. **Monitor for Performance Regression**
If epochs take significantly longer than average, it may indicate:
- System load increase
- Memory pressure
- GPU throttling
- Dataset loading bottleneck

### 4. **Benchmark Different Configurations**
```bash
# Test 1: --amp enabled
python train.py --amp --epochs 5

# Test 2: --no-amp
python train.py --no-amp --epochs 5

# Compare training_time.json from both runs
```

---

## Backward Compatibility

✅ **No breaking changes**
- All existing functionality preserved
- Timing is added transparently
- Code runs identically, just with timing data collected

---

## Summary

**Added:**
- ✅ Per-epoch timing in console output
- ✅ Training summary report printed at end
- ✅ `training_time.json` file saved for future analysis
- ✅ Test evaluation timing tracked
- ✅ Statistics: total, average, min, max epoch times

**Result:** You now have complete visibility into training runtime for:
- Planning training sessions
- Detecting performance issues
- Comparing different configurations
- Reproducing timing across runs
