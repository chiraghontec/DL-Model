#!/usr/bin/env python3
"""
Visualize and compare training metrics and test results for two models.
"""
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_training_log(log_file):
    """Parse training log to extract epoch metrics."""
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    with open(log_file, 'r') as f:
        text = f.read()
    
    # Find all epoch summaries
    pattern = r'Epoch\s+(\d+)/\d+.*?Train loss:\s+([\d.]+)\s+acc:\s+([\d.]+)\s+\|\s+Val loss:\s+([\d.]+)\s+acc:\s+([\d.]+)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        epoch, tloss, tacc, vloss, vacc = match
        epochs.append(int(epoch))
        train_loss.append(float(tloss))
        train_acc.append(float(tacc))
        val_loss.append(float(vloss))
        val_acc.append(float(vacc))
    
    return epochs, train_loss, train_acc, val_loss, val_acc


def parse_eval_log(log_file):
    """Parse evaluation log to extract test accuracy."""
    with open(log_file, 'r') as f:
        text = f.read()
    
    # Find accuracy line
    acc_match = re.search(r'accuracy\s+([\d.]+)\s+\d+', text)
    if acc_match:
        return float(acc_match.group(1))
    return None


def create_comparison_plots(log_file, eval28_file, eval30_file, output_dir='./'):
    """Create comparison plots for training and evaluation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Parse training log
    epochs, train_loss, train_acc, val_loss, val_acc = parse_training_log(log_file)
    
    # Parse evaluation results
    test_acc_28 = parse_eval_log(eval28_file)
    test_acc_30 = parse_eval_log(eval30_file)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Training and Validation Loss
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    ax1.axvline(x=28, color='g', linestyle='--', alpha=0.5, label='Epoch 28 (Best)')
    ax1.axvline(x=30, color='purple', linestyle='--', alpha=0.5, label='Epoch 30 (Last)')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Training and Validation Accuracy
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, [a*100 for a in train_acc], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, [a*100 for a in val_acc], 'r-', label='Val Acc', linewidth=2)
    ax2.axvline(x=28, color='g', linestyle='--', alpha=0.5, label='Epoch 28 (Best)')
    ax2.axvline(x=30, color='purple', linestyle='--', alpha=0.5, label='Epoch 30 (Last)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Test Accuracy Comparison
    ax3 = plt.subplot(2, 3, 3)
    models = ['Epoch 28\n(Best Val)', 'Epoch 30\n(Last)']
    test_accs = [test_acc_28*100 if test_acc_28 else 0, test_acc_30*100 if test_acc_30 else 0]
    colors = ['green', 'purple']
    bars = ax3.bar(models, test_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax3.set_title('Test Set Accuracy Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylim([90, 100])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 4. Epoch 28 Metrics Summary
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    epoch28_idx = epochs.index(28) if 28 in epochs else -1
    if epoch28_idx >= 0:
        summary28 = f"""
EPOCH 28 (BEST VALIDATION MODEL)

Training Metrics:
  • Train Loss:  {train_loss[epoch28_idx]:.4f}
  • Train Acc:   {train_acc[epoch28_idx]*100:.2f}%
  • Val Loss:    {val_loss[epoch28_idx]:.4f}
  • Val Acc:     {val_acc[epoch28_idx]*100:.2f}%

Test Metrics:
  • Test Acc:    {test_acc_28*100:.2f}%

Generalization Gap:
  • Train-Test:  {(train_acc[epoch28_idx]-test_acc_28)*100:.2f}%
  • Val-Test:    {(val_acc[epoch28_idx]-test_acc_28)*100:.2f}%
        """
        ax4.text(0.1, 0.5, summary28, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # 5. Epoch 30 Metrics Summary
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    epoch30_idx = epochs.index(30) if 30 in epochs else -1
    if epoch30_idx >= 0:
        summary30 = f"""
EPOCH 30 (FINAL MODEL)

Training Metrics:
  • Train Loss:  {train_loss[epoch30_idx]:.4f}
  • Train Acc:   {train_acc[epoch30_idx]*100:.2f}%
  • Val Loss:    {val_loss[epoch30_idx]:.4f}
  • Val Acc:     {val_acc[epoch30_idx]*100:.2f}%

Test Metrics:
  • Test Acc:    {test_acc_30*100:.2f}%

Generalization Gap:
  • Train-Test:  {(train_acc[epoch30_idx]-test_acc_30)*100:.2f}%
  • Val-Test:    {(val_acc[epoch30_idx]-test_acc_30)*100:.2f}%
        """
        ax5.text(0.1, 0.5, summary30, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='plum', alpha=0.3))
    
    # 6. Overall Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    best_val_epoch = epochs[np.argmax(val_acc)]
    best_val_acc = max(val_acc) * 100
    
    overall_summary = f"""
TRAINING SUMMARY

Dataset:
  • Train: 69,103 images (18 classes)
  • Val:   10,050 images (18 classes)
  • Test:  20,042 images (18 classes)

Best Results:
  • Best Val Epoch: {best_val_epoch}
  • Best Val Acc:   {best_val_acc:.2f}%
  • Test Acc (E28): {test_acc_28*100:.2f}%
  • Test Acc (E30): {test_acc_30*100:.2f}%

Improvement vs Old Model:
  • Old Test Acc: 13.87%
  • New Test Acc: {test_acc_28*100:.2f}%
  • Improvement:  +{test_acc_28*100-13.87:.2f}%
    """
    ax6.text(0.1, 0.5, overall_summary, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Model Comparison: Epoch 28 vs Epoch 30', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_file = output_dir / 'model_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_file}")
    
    # Show plot
    plt.show()
    
    return output_file


if __name__ == '__main__':
    log_file = 'train_clean_dataset.log'
    eval28_file = 'eval_epoch28.log'
    eval30_file = 'eval_epoch30.log'
    
    create_comparison_plots(log_file, eval28_file, eval30_file)
