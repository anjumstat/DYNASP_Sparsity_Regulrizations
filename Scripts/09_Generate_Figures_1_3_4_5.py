# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:00:00 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Generate Publication-Ready Figures for PAPER 3: DYNASP
Based on REAL experimental results from .npy files and CV data
Figures: 1, 3, 4, 5
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re

# =============================================
# CONFIGURATION
# =============================================

# Path to your combined CV results for Paper 3
cv_results_path = r"D:\zebfish2\Dynamic_Sparsity_Combined_Results\CV_all_models\ALL_METHODS_Fold_Metrics_Combined.csv"

# Output directories
output_dir = r"D:\zebfish2\DYNASP_Paper_Figures_REAL1_updated"
png_dir = os.path.join(output_dir, "PNG")
tiff_dir = os.path.join(output_dir, "TIFF")
os.makedirs(png_dir, exist_ok=True)
os.makedirs(tiff_dir, exist_ok=True)

# =============================================
# METHOD MAPPING
# =============================================

method_mapping = {
    'VARDON_DynamicSparsity': 'DYNASP Base',
    'VARDON_Sparsity_RealVD': 'DYNASP + RealVD',
    'VARDON_Sparsity_AdaptiveVD': 'DYNASP + AdaptiveVD',
    'VARDON_Sparsity_Gate': 'DYNASP + Gate',
    'VARDON_Sparsity_Full': 'DYNASP Full',
    'Standard_MLP': 'Standard MLP',
    'Logistic_Regression': 'Logistic Regression'
}

original_methods = [
    'Logistic_Regression',
    'Standard_MLP',
    'VARDON_DynamicSparsity',
    'VARDON_Sparsity_RealVD',
    'VARDON_Sparsity_AdaptiveVD',
    'VARDON_Sparsity_Gate',
    'VARDON_Sparsity_Full'
]

dynasp_original_names = [
    'VARDON_DynamicSparsity',
    'VARDON_Sparsity_RealVD',
    'VARDON_Sparsity_AdaptiveVD',
    'VARDON_Sparsity_Gate',
    'VARDON_Sparsity_Full'
]

method_colors = {
    'Logistic Regression': '#808080',
    'Standard MLP': '#1f77b4',
    'DYNASP Base': '#2ca02c',
    'DYNASP + RealVD': '#d62728',
    'DYNASP + AdaptiveVD': '#ff7f0e',
    'DYNASP + Gate': '#9467bd',
    'DYNASP Full': '#8c564b'
}

def get_line_style(original_name):
    if original_name in dynasp_original_names:
        return '-'
    else:
        return '--'

# =============================================
# GLOBAL FONT SETTINGS
# Increased all font sizes by 3 points and made bold
# =============================================

BASE_FONT_SIZE = 13
AXIS_LABEL_FONT_SIZE = 15
TITLE_FONT_SIZE = 15
SUPTITLE_FONT_SIZE = 17
LEGEND_FONT_SIZE = 12
TICK_FONT_SIZE = 12
ANNOTATION_FONT_SIZE = 12

plt.rcParams.update({
    'font.size': BASE_FONT_SIZE,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],

    'axes.labelsize': AXIS_LABEL_FONT_SIZE,
    'axes.titlesize': TITLE_FONT_SIZE,
    'legend.fontsize': LEGEND_FONT_SIZE,
    'xtick.labelsize': TICK_FONT_SIZE,
    'ytick.labelsize': TICK_FONT_SIZE,

    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.weight': 'bold',

    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def save_figure(fig, filename):
    """Save figure as both PNG and TIFF"""
    png_path = os.path.join(png_dir, f"{filename}.png")
    tiff_path = os.path.join(tiff_dir, f"{filename}.tiff")

    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(
        tiff_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        format='tiff',
        pil_kwargs={"compression": "tiff_lzw"}
    )

    print(f"  ✅ Saved: {filename}")

def make_axis_text_bold(ax):
    """Make all x-axis and y-axis tick labels bold and larger."""
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    ax.xaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)

    ax.title.set_fontweight('bold')
    ax.title.set_fontsize(TITLE_FONT_SIZE)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(TICK_FONT_SIZE)

    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(TICK_FONT_SIZE)

def make_colorbar_text_bold(cbar):
    """Make colorbar label and tick labels bold."""
    cbar.ax.yaxis.label.set_fontweight('bold')
    cbar.ax.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)

    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(TICK_FONT_SIZE)

# =============================================
# FIGURE 1: HYPERPARAMETER EFFECTS (LR × BS)
# =============================================

print("\n" + "=" * 60)
print("PAPER 3 - DYNASP: Generating Figure 1: Hyperparameter Effects (LR × BS)")
print("Using REAL CV data")
print("=" * 60)

if os.path.exists(cv_results_path):
    df = pd.read_csv(cv_results_path)
    print(f"✅ Loaded CV results: {df.shape[0]} rows")

    lr_values = [0.0001, 0.001, 0.01]
    bs_values = [32, 64, 128]

    heatmap_data = np.zeros((len(bs_values), len(lr_values)))

    for i, bs in enumerate(bs_values):
        for j, lr in enumerate(lr_values):
            subset = df[(df['Learning_Rate'] == lr) & (df['Batch_Size'] == bs)]
            if len(subset) > 0:
                heatmap_data[i, j] = subset['F1'].mean()

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Figure 1A: Heatmap
    im = ax1.imshow(
        heatmap_data,
        cmap='RdYlGn',
        aspect='auto',
        vmin=0.86,
        vmax=0.875
    )

    ax1.set_xticks(np.arange(len(lr_values)))
    ax1.set_yticks(np.arange(len(bs_values)))

    ax1.set_xticklabels(
        [f'{lr:.4f}' for lr in lr_values],
        fontweight='bold',
        fontsize=TICK_FONT_SIZE
    )
    ax1.set_yticklabels(
        bs_values,
        fontweight='bold',
        fontsize=TICK_FONT_SIZE
    )

    ax1.set_xlabel('Learning Rate', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax1.set_ylabel('Batch Size', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax1.set_title('Figure 1A. Mean F1 Score (LR × BS)', fontsize=TITLE_FONT_SIZE, fontweight='bold')

    for i in range(len(bs_values)):
        for j in range(len(lr_values)):
            ax1.text(
                j,
                i,
                f'{heatmap_data[i, j]:.4f}',
                ha="center",
                va="center",
                color="black",
                fontsize=ANNOTATION_FONT_SIZE,
                fontweight='bold'
            )

    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Mean F1 Score', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    make_colorbar_text_bold(cbar)

    # Figure 1B: Bar plot of mean F1 by Learning Rate
    best_by_lr = []
    for lr in lr_values:
        subset = df[df['Learning_Rate'] == lr]
        best_f1 = subset['F1'].mean() if len(subset) > 0 else 0
        best_by_lr.append(best_f1)

    bars = ax2.bar(
        [f'LR={lr}' for lr in lr_values],
        best_by_lr,
        color=['#1f77b4', '#ff7f0e', '#2ca02c'],
        edgecolor='black'
    )

    ax2.set_ylabel('Mean F1 Score', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax2.set_xlabel('Learning Rate', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax2.set_title('Figure 1B. Mean F1 by Learning Rate', fontsize=TITLE_FONT_SIZE, fontweight='bold')

    # Slightly expanded upper limit to keep labels inside the plot area
    ax2.set_ylim([0.86, 0.882])
    ax2.grid(True, axis='y', alpha=0.3)

    # Fixed label placement: labels are now inside bars/plot area
    for bar, val in zip(bars, best_by_lr):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val - 0.0010,
            f'{val:.4f}',
            ha='center',
            va='top',
            fontsize=ANNOTATION_FONT_SIZE,
            fontweight='bold',
            clip_on=True
        )

    make_axis_text_bold(ax1)
    make_axis_text_bold(ax2)

    plt.tight_layout()
    save_figure(fig1, "Figure1_Hyperparameter_Effects_DYNASP")
    plt.close(fig1)

# =============================================
# FIGURE 3: TRAINING AND VALIDATION CURVES
# Previous Figure 2 is now Figure 3
# =============================================

print("\n" + "=" * 60)
print("PAPER 3 - DYNASP: Generating Figure 3: Training and Validation Curves")
print("Using REAL .npy files from experiment")
print("=" * 60)

top_methods_original = [
    'VARDON_Sparsity_Gate',
    'VARDON_DynamicSparsity',
    'Standard_MLP'
]

best_configs = {
    'VARDON_Sparsity_Gate': {'lr': 0.0001, 'bs': 64},
    'VARDON_DynamicSparsity': {'lr': 0.0001, 'bs': 128},
    'Standard_MLP': {'lr': 0.0001, 'bs': 32}
}

base_dirs_dynasp = [
    r"D:\zebfish2\Dynamic_Sparsity_Results_0.01",
    r"D:\zebfish2\Dynamic_Sparsity_Results_0.001",
    r"D:\zebfish2\Dynamic_Sparsity_Results_0.0001"
]

def load_training_and_validation_data_dynasp(method, lr, bs):
    """Load REAL training and validation history from .npy files"""

    for base_dir in base_dirs_dynasp:
        if str(lr) in base_dir:
            config_pattern = f"lr_*_bs_{bs}"
            config_dirs = glob.glob(os.path.join(base_dir, "cv_runs", config_pattern, method))

            for config_dir in config_dirs:
                npy_dir = os.path.join(config_dir, "npy_files")

                if os.path.exists(npy_dir):
                    all_train_acc = []
                    all_val_acc = []
                    all_train_loss = []
                    all_val_loss = []

                    for fold in range(1, 11):
                        train_acc_file = os.path.join(npy_dir, f"fold{fold}_accuracy.npy")
                        if os.path.exists(train_acc_file):
                            all_train_acc.append(np.load(train_acc_file))

                        val_acc_file = os.path.join(npy_dir, f"fold{fold}_val_accuracy.npy")
                        if os.path.exists(val_acc_file):
                            all_val_acc.append(np.load(val_acc_file))
                        elif os.path.exists(train_acc_file):
                            all_val_acc.append(np.load(train_acc_file))

                        train_loss_file = os.path.join(npy_dir, f"fold{fold}_loss.npy")
                        if os.path.exists(train_loss_file):
                            all_train_loss.append(np.load(train_loss_file))

                        val_loss_file = os.path.join(npy_dir, f"fold{fold}_val_loss.npy")
                        if os.path.exists(val_loss_file):
                            all_val_loss.append(np.load(val_loss_file))
                        elif os.path.exists(train_loss_file):
                            all_val_loss.append(np.load(train_loss_file))

                    if all_train_acc:
                        max_len = max(len(acc) for acc in all_train_acc)

                        padded_train_acc = np.array([
                            np.pad(acc, (0, max_len - len(acc)), constant_values=acc[-1])
                            for acc in all_train_acc
                        ])

                        if all_val_acc:
                            padded_val_acc = np.array([
                                np.pad(acc, (0, max_len - len(acc)), constant_values=acc[-1])
                                for acc in all_val_acc
                            ])
                        else:
                            padded_val_acc = padded_train_acc

                        if all_train_loss:
                            padded_train_loss = np.array([
                                np.pad(loss, (0, max_len - len(loss)), constant_values=loss[-1])
                                for loss in all_train_loss
                            ])
                            train_loss_mean = padded_train_loss.mean(axis=0)
                        else:
                            train_loss_mean = None

                        if all_val_loss:
                            padded_val_loss = np.array([
                                np.pad(loss, (0, max_len - len(loss)), constant_values=loss[-1])
                                for loss in all_val_loss
                            ])
                            val_loss_mean = padded_val_loss.mean(axis=0)
                        else:
                            val_loss_mean = None

                        return {
                            'train_acc_mean': padded_train_acc.mean(axis=0),
                            'train_acc_std': padded_train_acc.std(axis=0),
                            'val_acc_mean': padded_val_acc.mean(axis=0),
                            'val_acc_std': padded_val_acc.std(axis=0),
                            'train_loss_mean': train_loss_mean,
                            'val_loss_mean': val_loss_mean,
                            'epochs': max_len
                        }

    return None

fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle(
    'Figure 3. Training and Validation Curves - DYNASP (REAL Data)',
    fontsize=SUPTITLE_FONT_SIZE,
    fontweight='bold'
)

for method in top_methods_original:
    config = best_configs.get(method, {'lr': 0.001, 'bs': 64})
    data = load_training_and_validation_data_dynasp(method, config['lr'], config['bs'])

    if data:
        epochs = np.arange(1, data['epochs'] + 1)
        display_name = method_mapping.get(method, method)
        color = method_colors.get(display_name, '#1f77b4')
        line_style = get_line_style(method)

        # Training Accuracy
        axes[0, 0].plot(
            epochs,
            data['train_acc_mean'],
            linewidth=2.5,
            color=color,
            linestyle=line_style,
            label=f'{display_name}'
        )
        axes[0, 0].fill_between(
            epochs,
            data['train_acc_mean'] - data['train_acc_std'],
            data['train_acc_mean'] + data['train_acc_std'],
            alpha=0.15,
            color=color
        )

        # Validation Accuracy
        axes[0, 1].plot(
            epochs,
            data['val_acc_mean'],
            linewidth=2.5,
            color=color,
            linestyle=line_style,
            label=f'{display_name}'
        )
        axes[0, 1].fill_between(
            epochs,
            data['val_acc_mean'] - data['val_acc_std'],
            data['val_acc_mean'] + data['val_acc_std'],
            alpha=0.15,
            color=color
        )

        # Training Loss
        if data['train_loss_mean'] is not None:
            axes[1, 0].plot(
                epochs,
                data['train_loss_mean'],
                linewidth=2.5,
                color=color,
                linestyle=line_style,
                label=f'{display_name}'
            )

        # Validation Loss
        if data['val_loss_mean'] is not None:
            axes[1, 1].plot(
                epochs,
                data['val_loss_mean'],
                linewidth=2.5,
                color=color,
                linestyle=line_style,
                label=f'{display_name}'
            )

# Configure Figure 3 axes
axes[0, 0].set_xlabel('Epoch', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
axes[0, 0].set_ylabel('Training Accuracy', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
axes[0, 0].set_title('Figure 3A. Training Accuracy (REAL)', fontsize=TITLE_FONT_SIZE, fontweight='bold')
axes[0, 0].legend(loc='lower right', fontsize=LEGEND_FONT_SIZE)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0.7, 1.0])

axes[0, 1].set_xlabel('Epoch', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
axes[0, 1].set_ylabel('Validation Accuracy', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
axes[0, 1].set_title('Figure 3B. Validation Accuracy (REAL)', fontsize=TITLE_FONT_SIZE, fontweight='bold')
axes[0, 1].legend(loc='lower right', fontsize=LEGEND_FONT_SIZE)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0.7, 1.0])

axes[1, 0].set_xlabel('Epoch', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
axes[1, 0].set_ylabel('Training Loss', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
axes[1, 0].set_title('Figure 3C. Training Loss (REAL)', fontsize=TITLE_FONT_SIZE, fontweight='bold')
axes[1, 0].legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 0.8])

axes[1, 1].set_xlabel('Epoch', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
axes[1, 1].set_ylabel('Validation Loss', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
axes[1, 1].set_title('Figure 3D. Validation Loss (REAL)', fontsize=TITLE_FONT_SIZE, fontweight='bold')
axes[1, 1].legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0, 0.8])

for ax in axes.flatten():
    make_axis_text_bold(ax)

plt.tight_layout()
save_figure(fig2, "Figure3_Training_Validation_Curves_DYNASP_REAL")
plt.close(fig2)

# =============================================
# FIGURE 4: DYNASP VARIANTS RANKING
# Previous Figure 3 is now Figure 4
# =============================================

print("\n" + "=" * 60)
print("PAPER 3 - DYNASP: Generating Figure 4: DYNASP Variants Ranking")
print("Using REAL CV data")
print("=" * 60)

if os.path.exists(cv_results_path):
    df = pd.read_csv(cv_results_path)

    dynasp_df = df[df['Method_Name'].isin(dynasp_original_names)]

    if not dynasp_df.empty:
        dynasp_stats = dynasp_df.groupby('Method_Name').agg({
            'F1': 'mean',
            'AUC': 'mean',
            'Accuracy': 'mean'
        }).round(4)

        dynasp_stats = dynasp_stats.sort_values('F1', ascending=False)
        display_indices = [method_mapping.get(idx, idx) for idx in dynasp_stats.index]

        fig3, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(display_indices))
        width = 0.25

        bars1 = ax.bar(
            x - width,
            dynasp_stats['F1'],
            width,
            label='F1 Score',
            color='#2ca02c',
            edgecolor='black',
            linewidth=0.5
        )

        bars2 = ax.bar(
            x,
            dynasp_stats['AUC'],
            width,
            label='AUC',
            color='#1f77b4',
            edgecolor='black',
            linewidth=0.5
        )

        bars3 = ax.bar(
            x + width,
            dynasp_stats['Accuracy'],
            width,
            label='Accuracy',
            color='#ff7f0e',
            edgecolor='black',
            linewidth=0.5
        )

        ax.set_xlabel('DYNASP Variant', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
        ax.set_ylabel('Score', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
        ax.set_title(
            'Figure 4. Performance Ranking of DYNASP Variants (REAL)',
            fontsize=TITLE_FONT_SIZE,
            fontweight='bold'
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            display_indices,
            rotation=45,
            ha='right',
            fontsize=TICK_FONT_SIZE,
            fontweight='bold'
        )

        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=LEGEND_FONT_SIZE)
        ax.set_ylim([0.75, 0.99])
        ax.grid(True, axis='y', alpha=0.3)

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.002,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )

        if len(bars1) > 0:
            bars1[0].set_edgecolor('gold')
            bars1[0].set_linewidth(2.5)
            bars2[0].set_edgecolor('gold')
            bars2[0].set_linewidth(2.5)
            bars3[0].set_edgecolor('gold')
            bars3[0].set_linewidth(2.5)

        make_axis_text_bold(ax)

        plt.tight_layout()
        save_figure(fig3, "Figure4_DYNASP_Ranking_REAL")
        plt.close(fig3)

# =============================================
# FIGURE 5: CROSS-VALIDATION BOXPLOTS
# Previous Figure 4 is now Figure 5
# =============================================

print("\n" + "=" * 60)
print("PAPER 3 - DYNASP: Generating Figure 5: Cross-Validation Boxplots")
print("Using REAL CV data")
print("=" * 60)

if os.path.exists(cv_results_path):
    df = pd.read_csv(cv_results_path)

    fig4, ax = plt.subplots(figsize=(12, 7))

    boxplot_data = []
    boxplot_labels = []
    boxplot_colors = []

    for method in original_methods:
        method_data = df[df['Method_Name'] == method]['F1'].values

        if len(method_data) > 0:
            boxplot_data.append(method_data)

            display_name = method_mapping.get(method, method)
            boxplot_labels.append(display_name)
            boxplot_colors.append(method_colors.get(display_name, '#1f77b4'))

    if boxplot_data:
        bp = ax.boxplot(
            boxplot_data,
            labels=boxplot_labels,
            patch_artist=True,
            medianprops=dict(linewidth=2, color='black'),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1)
        )

        for patch, color in zip(bp['boxes'], boxplot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        dynasp_display_names = [method_mapping.get(m, m) for m in dynasp_original_names]

        for i, label in enumerate(boxplot_labels):
            if label in dynasp_display_names:
                bp['boxes'][i].set_edgecolor('gold')
                bp['boxes'][i].set_linewidth(2.5)

        ax.set_ylabel('F1 Score', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
        ax.set_xlabel('Method', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
        ax.set_title(
            'Figure 5. Cross-Validation F1 Score Distribution (10-Fold)\n'
            '(Gold borders = DYNASP variants) - REAL',
            fontsize=TITLE_FONT_SIZE,
            fontweight='bold'
        )

        ax.tick_params(axis='x', rotation=45, labelsize=TICK_FONT_SIZE)
        ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim([0.75, 0.95])

        best_dynasp_original = 'VARDON_Sparsity_Gate'

        if best_dynasp_original in df['Method_Name'].values:
            best_mean = df[df['Method_Name'] == best_dynasp_original]['F1'].mean()

            ax.axhline(
                y=best_mean,
                color='green',
                linestyle='--',
                linewidth=1.5,
                alpha=0.7
            )

            best_display = method_mapping.get(best_dynasp_original, best_dynasp_original)

            ax.text(
                len(boxplot_labels) - 0.5,
                best_mean + 0.003,
                f'Best DYNASP ({best_display}) Mean: {best_mean:.4f}',
                fontsize=11,
                color='green',
                ha='right',
                fontweight='bold'
            )

        make_axis_text_bold(ax)

        plt.tight_layout()
        save_figure(fig4, "Figure5_CV_Boxplots_DYNASP_REAL")
        plt.close(fig4)

# =============================================
# SUMMARY
# =============================================

print("\n" + "=" * 60)
print("PAPER 3 - DYNASP: FIGURE GENERATION COMPLETE")
print("ALL FIGURES BASED ON REAL EXPERIMENTAL DATA")
print("=" * 60)
print(f"\n📁 PNG files saved to: {png_dir}")
print(f"📁 TIFF files saved to: {tiff_dir}")

print("\nGenerated figures with updated numbering:")
print("  - Figure 1: Hyperparameter Effects")
print("  - Figure 3: Training and Validation Curves")
print("  - Figure 4: DYNASP Variants Ranking")
print("  - Figure 5: Cross-Validation Boxplots")

print("\nUpdates applied:")
print("  - Figure 1B bar labels moved inside the plot area")
print("  - Figure 2 renamed to Figure 3")
print("  - Figure 3 renamed to Figure 4")
print("  - Figure 4 renamed to Figure 5")
print("  - Figure 1 remains Figure 1")
print("  - X-axis and Y-axis labels increased by 3 points and made bold")
print("  - X-ticks and Y-ticks increased by 3 points and made bold")
print("  - Figure titles and legends increased by 3 points")
print("\n" + "=" * 60)