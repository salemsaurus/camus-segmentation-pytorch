"""
Practical Examples: Using Refactored utils.py Functions

This file demonstrates how to integrate new utils.py functions into your
CAMUS training and evaluation pipeline.
"""

from pathlib import Path
import numpy as np
from utils import (
    save_batch_visualizations,
    save_training_history,
    save_error_map,
    save_comparison_figure,
    seed_everything,
)


# ============================================================================
# EXAMPLE 1: Export Test-Set Visualizations After Training
# ============================================================================

def visualize_test_set(
    test_loader,
    model,
    output_dir: Path,
    device: str = "cuda",
) -> None:
    """
    Generate comprehensive visualizations for entire test set.
    
    Usage:
        visualize_test_set(test_loader, model, Path("runs/fold_0/visualizations"))
    
    Output structure:
        visualizations/
            sample_0000/
                overlay.png
                contour_overlay.png
                comparison.png
                error_map.png
            sample_0001/
                ...
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    images_list = []
    gts_list = []
    preds_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            gt_masks = batch["mask"].cpu().numpy()
            
            logits = model(images)  # (B, 4, H, W)
            pred_masks = torch.argmax(logits, dim=1).cpu().numpy()  # (B, H, W)
            
            # Store for batch visualization
            images_list.append(images.cpu().numpy())
            gts_list.append(gt_masks)
            preds_list.append(pred_masks)
    
    # Concatenate all batches
    images = np.concatenate(images_list, axis=0)  # (N, H, W)
    gts = np.concatenate(gts_list, axis=0)  # (N, H, W)
    preds = np.concatenate(preds_list, axis=0)  # (N, H, W)
    
    # Generate all visualizations
    save_batch_visualizations(images, gts, preds, output_dir)
    print(f"✓ Test-set visualizations saved to {output_dir}")


# ============================================================================
# EXAMPLE 2: Save Training Metrics History
# ============================================================================

def save_fold_history(
    fold_output_dir: Path,
    history: dict,
) -> None:
    """
    Export training metrics for later analysis.
    
    Usage in train.py:
        # After training loop
        save_fold_history(
            Path("runs/training/fold_0"),
            history = {
                'train_loss': train_losses,
                'val_loss': val_losses,
                'train_dice': train_dices,
                'val_dice': val_dices,
            }
        )
    
    Then analyze offline:
        >>> import json
        >>> with open("runs/training/fold_0/training_history.json") as f:
        ...     history = json.load(f)
        >>> # Custom analysis, plotting, comparison
    """
    history_path = fold_output_dir / "training_history.json"
    save_training_history(history, history_path)
    print(f"✓ Training history saved to {history_path}")


# ============================================================================
# EXAMPLE 3: Error Analysis for Specific Samples
# ============================================================================

def analyze_sample_errors(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    output_dir: Path,
    sample_id: str = "sample_001",
) -> None:
    """
    Generate detailed error maps for specific samples.
    
    Usage:
        analyze_sample_errors(
            image=test_image,
            gt_mask=test_gt,
            pred_mask=test_pred,
            output_dir=Path("error_analysis"),
            sample_id="patient0042"
        )
    
    Output:
        error_analysis/
            error_map.png        # Red: FN, Blue: FP, Green: Correct
            comparison.png       # Three-panel comparison
            contour_overlay.png  # Contour-based comparison
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all visualization types
    save_error_map(image, gt_mask, pred_mask, output_dir / f"{sample_id}_error_map.png")
    save_comparison_figure(image, gt_mask, pred_mask, output_dir / f"{sample_id}_comparison.png")
    
    print(f"✓ Error analysis for {sample_id} saved to {output_dir}")
    print(f"  → {sample_id}_error_map.png")
    print(f"  → {sample_id}_comparison.png")


# ============================================================================
# EXAMPLE 4: Deterministic Training with seed_everything()
# ============================================================================

def setup_deterministic_training(seed: int = 42) -> None:
    """
    Initialize all random seeds before training.
    
    Usage at start of train.py:
        from utils import seed_everything
        seed_everything(42)  # Do this BEFORE loading data/model
        
        # Now all operations are deterministic
        model = UNet1(...)
        train_loader = DataLoader(...)
        # Training will be reproducible
    
    Sets:
        - Python's random module
        - NumPy
        - PyTorch (CPU and GPU)
        - CuDNN (deterministic algorithms)
        - PYTHONHASHSEED environment variable
    """
    seed_everything(seed)
    print(f"✓ All random seeds set to {seed}")
    print(f"✓ PYTHONHASHSEED={seed}")
    print(f"✓ CuDNN deterministic mode enabled")
    print(f"✓ Training will be reproducible")


# ============================================================================
# EXAMPLE 5: Complete Training Loop Integration
# ============================================================================

def train_with_visualization(
    model,
    train_loader,
    val_loader,
    test_loader,
    num_epochs: int = 100,
    output_fold: Path = Path("runs/training/fold_0"),
    seed: int = 42,
) -> None:
    """
    Complete training loop with visualization integration.
    
    Shows how to incorporate all new utils.py features into train.py.
    """
    # Setup reproducibility
    seed_everything(seed)
    
    output_fold.mkdir(parents=True, exist_ok=True)
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}
    
    model = model.to("cuda")
    
    # Training loop (simplified)
    for epoch in range(num_epochs):
        # ... training code ...
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_dice={val_dice:.4f}")
    
    # After training: export history
    save_fold_history(output_fold, history)
    
    # After training: generate test visualizations
    test_viz_dir = output_fold / "test_visualizations"
    visualize_test_set(test_loader, model, test_viz_dir, device="cuda")
    
    # Summary
    print(f"\n✓ Training complete")
    print(f"  → History: {output_fold / 'training_history.json'}")
    print(f"  → Visualizations: {test_viz_dir}")


# ============================================================================
# EXAMPLE 6: Analyzing Multiple Folds
# ============================================================================

def compare_fold_results(
    runs_dir: Path = Path("runs/training"),
    num_folds: int = 5,
) -> None:
    """
    Compare training histories across multiple folds.
    
    Usage:
        compare_fold_results(Path("runs/training"), num_folds=5)
    
    Reads:
        runs/training/fold_0/training_history.json
        runs/training/fold_1/training_history.json
        ...
        runs/training/fold_4/training_history.json
    
    Then you can analyze aggregated metrics across folds.
    """
    import json
    import matplotlib.pyplot as plt
    
    histories = {}
    for fold in range(num_folds):
        history_path = runs_dir / f"fold_{fold}" / "training_history.json"
        with open(history_path) as f:
            histories[fold] = json.load(f)
    
    # Example: plot val_dice across all folds
    fig, ax = plt.subplots(figsize=(10, 6))
    for fold, history in histories.items():
        ax.plot(history['val_dice'], label=f'Fold {fold}', alpha=0.7)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Dice')
    ax.set_title('Validation Dice Across Folds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tight_layout()
    ax.savefig(runs_dir / "fold_comparison.png", dpi=100)
    
    print(f"✓ Fold comparison plot saved to {runs_dir / 'fold_comparison.png'}")


# ============================================================================
# EXAMPLE 7: Quality Assurance - Check Post-processing Configuration
# ============================================================================

def verify_postprocessing_config() -> None:
    """
    Verify that post-processing configuration matches your expectations.
    
    Run this to confirm POSTPROCESS_CLASSES is set correctly.
    """
    from utils import POSTPROCESS_CLASSES
    
    print("Post-processing Configuration:")
    print("-" * 40)
    for cls_id, use_largest in POSTPROCESS_CLASSES.items():
        class_names = {1: "LV", 2: "Myocardium", 3: "Left Atrium"}
        status = "✓ Enabled" if use_largest else "✗ Disabled"
        print(f"  Class {cls_id} ({class_names[cls_id]}): {status}")
    
    print("\nNote: Myocardium (class 2) skips largest_connected_component")
    print("      to preserve ring-shaped structure.")
    print("\nTo modify, edit POSTPROCESS_CLASSES in utils.py:")
    print("  POSTPROCESS_CLASSES = {1: True, 2: False, 3: True}")


# ============================================================================
# Main: Quick Test
# ============================================================================

if __name__ == "__main__":
    print("utils.py Integration Examples")
    print("=" * 50)
    print()
    
    # Example 7: Verify configuration
    verify_postprocessing_config()
    print()
    
    # Example 4: Setup deterministic training
    print("\nExample 4: Deterministic Training")
    print("-" * 50)
    setup_deterministic_training(seed=42)
    print()
    
    # Example: Create dummy visualizations
    print("\nExample: Creating Sample Visualizations")
    print("-" * 50)
    
    # Create dummy data
    dummy_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    dummy_gt = np.random.randint(0, 4, (256, 256), dtype=np.uint8)
    dummy_pred = np.random.randint(0, 4, (256, 256), dtype=np.uint8)
    
    output_dir = Path("/tmp/camus_viz_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving sample visualizations to {output_dir}...")
    analyze_sample_errors(
        dummy_image, dummy_gt, dummy_pred,
        output_dir=output_dir,
        sample_id="dummy_sample"
    )
    print("✓ Done!\n")
