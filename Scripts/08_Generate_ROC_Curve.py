# -*- coding: utf-8 -*-
"""
Generate REAL ROC Curves from actual predictions and true labels
PAPER 3: DYNASP - Dynamic Sparsity Neural Network
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import glob
import re

# =============================================
# CONFIGURATION
# =============================================

# Paths to your DYNASP experiment results
base_dirs = [
    r"D:\zebfish2\Dynamic_Sparsity_Results_0.01",
    r"D:\zebfish2\Dynamic_Sparsity_Results_0.001",
    r"D:\zebfish2\Dynamic_Sparsity_Results_0.0001",
]

# Original method names as they appear in your folders
methods = [
    'Logistic_Regression',
    'Standard_MLP',
    'VARDON_DynamicSparsity',
    'VARDON_Sparsity_RealVD',
    'VARDON_Sparsity_AdaptiveVD',
    'VARDON_Sparsity_Gate',
    'VARDON_Sparsity_Full'
]

# Map original names to DYNASP display names
method_display_names = {
    'Logistic_Regression': 'Logistic Regression',
    'Standard_MLP': 'Standard MLP',
    'VARDON_DynamicSparsity': 'DYNASP Base',
    'VARDON_Sparsity_RealVD': 'DYNASP + RealVD',
    'VARDON_Sparsity_AdaptiveVD': 'DYNASP + AdaptiveVD',
    'VARDON_Sparsity_Gate': 'DYNASP + Gate',
    'VARDON_Sparsity_Full': 'DYNASP Full'
}

# Best configurations from your Paper 3 results (based on Test MCC)
best_configs = {
    'VARDON_Sparsity_Gate': {'lr': 0.0001, 'bs': 64},
    'VARDON_DynamicSparsity': {'lr': 0.0001, 'bs': 128},
    'Standard_MLP': {'lr': 0.0001, 'bs': 32},
    'VARDON_Sparsity_AdaptiveVD': {'lr': 0.01, 'bs': 64},
    'VARDON_Sparsity_RealVD': {'lr': 0.001, 'bs': 128},
    'VARDON_Sparsity_Full': {'lr': 0.01, 'bs': 128},
    'Logistic_Regression': {'lr': 0.001, 'bs': 64}
}

# Colors for methods (based on DYNASP display names)
method_colors = {
    'Logistic Regression': '#808080',      # Gray
    'Standard MLP': '#1f77b4',             # Blue
    'DYNASP Base': '#2ca02c',              # Green
    'DYNASP + RealVD': '#d62728',          # Red
    'DYNASP + AdaptiveVD': '#ff7f0e',      # Orange
    'DYNASP + Gate': '#9467bd',            # Purple
    'DYNASP Full': '#8c564b'               # Brown
}

def load_roc_data(method, lr, bs):
    """
    Load predictions and true labels from all CV folds to compute ROC
    For Paper 3, we use CV fold predictions (same as Paper 1 approach)
    """
    
    all_probs = []
    all_labels = []
    
    for base_dir in base_dirs:
        if str(lr) in base_dir:
            # Find the config folder
            config_pattern = f"lr_*_bs_{bs}"
            config_dirs = glob.glob(os.path.join(base_dir, "cv_runs", config_pattern, method))
            
            for config_dir in config_dirs:
                npy_dir = os.path.join(config_dir, "npy_files")
                if os.path.exists(npy_dir):
                    for fold in range(1, 11):
                        # Load predictions (probability for class 1 - Enzyme)
                        pred_file = os.path.join(npy_dir, f"fold{fold}_predictions.npy")
                        labels_file = os.path.join(npy_dir, f"fold{fold}_true_labels.npy")
                        
                        if os.path.exists(pred_file) and os.path.exists(labels_file):
                            pred = np.load(pred_file)
                            labels = np.load(labels_file)
                            
                            # Get probability for class 1 (Enzyme)
                            if pred.shape[1] == 2:
                                probs = pred[:, 1]
                            else:
                                probs = pred
                            
                            all_probs.extend(probs)
                            all_labels.extend(labels)
                            print(f"      Loaded fold {fold}: {len(probs)} samples")
    
    if len(all_probs) > 0:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
    return None, None, None

# =============================================
# FIRST, CHECK WHAT FILES ARE AVAILABLE
# =============================================

print("\n" + "="*60)
print("PAPER 3: Checking available prediction files")
print("="*60)

for base_dir in base_dirs:
    print(f"\n📁 Checking: {os.path.basename(base_dir)}")
    if os.path.exists(base_dir):
        cv_runs_dir = os.path.join(base_dir, "cv_runs")
        if os.path.exists(cv_runs_dir):
            configs = glob.glob(os.path.join(cv_runs_dir, "*"))
            print(f"   Found {len(configs)} configuration folders")
            for config in configs[:2]:
                print(f"     - {os.path.basename(config)}")
                for method in methods[:2]:
                    method_dir = os.path.join(config, method)
                    if os.path.exists(method_dir):
                        npy_dir = os.path.join(method_dir, "npy_files")
                        if os.path.exists(npy_dir):
                            pred_files = glob.glob(os.path.join(npy_dir, "fold*_predictions.npy"))
                            if pred_files:
                                print(f"         ✅ {method}: {len(pred_files)} prediction files found")

# =============================================
# GENERATE REAL ROC CURVES
# =============================================

print("\n" + "="*60)
print("PAPER 3: Generating REAL ROC Curves from Actual Data")
print("DYNASP - Dynamic Sparsity Neural Network")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 8))

loaded_methods = []

for method in methods:
    config = best_configs.get(method, {'lr': 0.001, 'bs': 64})
    print(f"\n  Loading {method} (LR={config['lr']}, BS={config['bs']})...")
    
    fpr, tpr, roc_auc = load_roc_data(method, config['lr'], config['bs'])
    
    display_name = method_display_names.get(method, method)
    color = method_colors.get(display_name, '#1f77b4')
    
    # Determine line style (dashed for baselines)
    if method in ['Logistic_Regression', 'Standard_MLP']:
        linestyle = '--'
        linewidth = 2
    else:
        linestyle = '-'
        linewidth = 2.5
    
    if fpr is not None:
        ax.plot(fpr, tpr, linewidth=linewidth, 
                color=color, linestyle=linestyle,
                label=f'{display_name} (AUC = {roc_auc:.4f})')
        print(f"  ✅ {display_name}: AUC = {roc_auc:.4f}")
        loaded_methods.append(method)
    else:
        print(f"  ⚠️ {display_name}: No ROC data found")

# Diagonal line (random classifier)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Random (AUC = 0.5)')

ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
ax.set_title('Figure 2. Receiver Operating Characteristic (ROC) Curves\nDYNASP (Dynamic Sparsity Neural Network)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

# Save figures
output_dir = r"D:\zebfish2\DYNASP_Paper_Figures_ROC_Real"
png_dir = os.path.join(output_dir, "PNG")
tiff_dir = os.path.join(output_dir, "TIFF")
os.makedirs(png_dir, exist_ok=True)
os.makedirs(tiff_dir, exist_ok=True)

if loaded_methods:
    png_path = os.path.join(png_dir, "Figure2_ROC_Curves_DYNASP_REAL.png")
    tiff_path = os.path.join(tiff_dir, "Figure2_ROC_Curves_DYNASP_REAL.tiff")
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(tiff_path, dpi=300, bbox_inches='tight', facecolor='white', 
                format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    print(f"\n✅ Saved: {png_path}")
    print(f"✅ Saved: {tiff_path}")
    
    # Also save as PDF
    pdf_path = os.path.join(output_dir, "Figure2_ROC_Curves_DYNASP_REAL.pdf")
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {pdf_path}")
else:
    print("\n❌ No ROC data found for any method!")
    print("\nPossible reasons:")
    print("  1. The .npy files don't exist in the expected locations")
    print("  2. The method names don't match folder names")
    print("  3. The learning rate/batch size combinations are incorrect")
    
    # Show what folders exist
    print("\n📁 Available folders in your results:")
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            cv_runs = os.path.join(base_dir, "cv_runs")
            if os.path.exists(cv_runs):
                configs = glob.glob(os.path.join(cv_runs, "*"))
                print(f"\n  {os.path.basename(base_dir)}:")
                for config in configs[:5]:
                    print(f"    - {os.path.basename(config)}")

plt.close(fig)

# =============================================
# OPTIONAL: Create a second figure with only DYNASP variants
# =============================================

if loaded_methods:
    print("\n" + "="*60)
    print("Generating ROC Curves for DYNASP Variants Only")
    print("="*60)
    
    dynasp_methods = ['VARDON_DynamicSparsity', 'VARDON_Sparsity_RealVD', 
                      'VARDON_Sparsity_AdaptiveVD', 'VARDON_Sparsity_Gate', 
                      'VARDON_Sparsity_Full']
    
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    
    for method in dynasp_methods:
        config = best_configs.get(method, {'lr': 0.001, 'bs': 64})
        fpr, tpr, roc_auc = load_roc_data(method, config['lr'], config['bs'])
        
        display_name = method_display_names.get(method, method)
        color = method_colors.get(display_name, '#1f77b4')
        
        if fpr is not None:
            ax2.plot(fpr, tpr, linewidth=2.5, 
                    color=color, linestyle='-',
                    label=f'{display_name} (AUC = {roc_auc:.4f})')
            print(f"  ✅ {display_name}: AUC = {roc_auc:.4f}")
    
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Random (AUC = 0.5)')
    ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    ax2.set_title('Figure 2. ROC Curves - DYNASP Variants Only', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    
    png_path2 = os.path.join(png_dir, "Figure2_ROC_Curves_DYNASP_Only_REAL.png")
    tiff_path2 = os.path.join(tiff_dir, "Figure2_ROC_Curves_DYNASP_Only_REAL.tiff")
    fig2.savefig(png_path2, dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig(tiff_path2, dpi=300, bbox_inches='tight', facecolor='white', 
                format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    print(f"\n✅ Saved: {png_path2}")
    print(f"✅ Saved: {tiff_path2}")
    
    plt.close(fig2)

print("\n" + "="*60)
print("✅ PAPER 3 ROC CURVE GENERATION COMPLETE!")
print(f"📁 Output directory: {output_dir}")
print("="*60)