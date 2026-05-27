# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:09:26 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Combine Results from Multiple Learning Rate Experiments for PAPER 3
Dynamic Sparsity Regularization for Efficient Neural Networks
"""

import os
import pandas as pd
import glob

# =============================================
# Configuration
# =============================================
base_dirs = [
    r"D:\zebfish2\Dynamic_Sparsity_Results_0.01",
    r"D:\zebfish2\Dynamic_Sparsity_Results_0.001",
    r"D:\zebfish2\Dynamic_Sparsity_Results_0.0001",
]

output_dir = r"D:\zebfish2\Dynamic_Sparsity_Combined_Results"
os.makedirs(output_dir, exist_ok=True)

# =============================================
# 1. Combine CV Summary Results
# =============================================
print("=" * 70)
print("PAPER 3: COMBINING DYNAMIC SPARSITY CV RESULTS")
print("=" * 70)

all_cv_results = []

for base_dir in base_dirs:
    # Look for Paper3 CV summary file
    cv_file = os.path.join(base_dir, "Paper3_CV_Summary_Results.csv")
    
    # Extract learning rate from folder name
    lr_value = base_dir.split("_")[-1]  # Gets "0.01", "0.001", or "0.0001"
    
    if os.path.exists(cv_file):
        df = pd.read_csv(cv_file)
        df['Learning_Rate'] = float(lr_value)
        all_cv_results.append(df)
        print(f"✅ Loaded: {cv_file} ({len(df)} rows, LR={lr_value})")
    else:
        print(f"⚠️ Not found: {cv_file}")

if all_cv_results:
    combined_cv = pd.concat(all_cv_results, ignore_index=True)
    
    # Sort by Learning Rate and MCC
    combined_cv = combined_cv.sort_values(['Learning_Rate', 'mean_mcc'], 
                                          ascending=[True, False])
    
    # Save combined CV results
    cv_output_path = os.path.join(output_dir, "Combined_CV_Results.csv")
    combined_cv.to_csv(cv_output_path, index=False)
    print(f"\n✅ Combined CV results saved to: {cv_output_path}")
    print(f"   Total rows: {len(combined_cv)}")
    print(f"   Learning rates: {combined_cv['Learning_Rate'].unique()}")
    print(f"   Methods: {combined_cv['method'].unique()}")
else:
    print("❌ No CV results found!")

# =============================================
# 2. Combine Test Results for All Models
# =============================================
print("\n" + "=" * 70)
print("COMBINING TEST RESULTS FOR ALL MODELS")
print("=" * 70)

all_test_results = []

for base_dir in base_dirs:
    test_file = os.path.join(base_dir, "Paper3_Test_Results_All_Models.csv")
    lr_value = base_dir.split("_")[-1]
    
    if os.path.exists(test_file):
        df = pd.read_csv(test_file)
        df['Learning_Rate'] = float(lr_value)
        all_test_results.append(df)
        print(f"✅ Loaded: {test_file} ({len(df)} rows, LR={lr_value})")
    else:
        print(f"⚠️ Not found: {test_file}")

if all_test_results:
    combined_test = pd.concat(all_test_results, ignore_index=True)
    
    # Sort by Learning Rate and Test MCC
    combined_test = combined_test.sort_values(['Learning_Rate', 'Test_MCC'], 
                                              ascending=[True, False])
    
    # Save combined test results
    test_output_path = os.path.join(output_dir, "Combined_Test_Results_All_Models.csv")
    combined_test.to_csv(test_output_path, index=False)
    print(f"\n✅ Combined test results saved to: {test_output_path}")
    print(f"   Total rows: {len(combined_test)}")
    print(f"   Learning rates: {combined_test['Learning_Rate'].unique()}")
    print(f"   Methods: {combined_test['Method'].unique()}")
else:
    print("❌ No test results found!")

# =============================================
# 3. Combine Selected Model Results (CV + Test)
# =============================================
print("\n" + "=" * 70)
print("COMBINING SELECTED MODEL RESULTS")
print("=" * 70)

all_selected_results = []

for base_dir in base_dirs:
    selected_file = os.path.join(base_dir, "Paper3_Selected_Model_CV_and_Test_Result.csv")
    lr_value = base_dir.split("_")[-1]
    
    if os.path.exists(selected_file):
        df = pd.read_csv(selected_file)
        df['Learning_Rate'] = float(lr_value)
        all_selected_results.append(df)
        print(f"✅ Loaded: {selected_file} (LR={lr_value})")
    else:
        print(f"⚠️ Not found: {selected_file}")

if all_selected_results:
    combined_selected = pd.concat(all_selected_results, ignore_index=True)
    
    # Sort by Test MCC
    combined_selected = combined_selected.sort_values('Test_MCC', ascending=False)
    
    # Save combined selected results
    selected_output_path = os.path.join(output_dir, "Combined_Selected_Model_Results.csv")
    combined_selected.to_csv(selected_output_path, index=False)
    print(f"\n✅ Combined selected model results saved to: {selected_output_path}")
    print(f"   Total rows: {len(combined_selected)}")
else:
    print("❌ No selected model results found!")

# =============================================
# 4. Create Best Model Summary Table for Paper
# =============================================
print("\n" + "=" * 70)
print("CREATING PAPER TABLE: BEST MODEL PER CONFIGURATION")
print("=" * 70)

if 'combined_test' in locals() and not combined_test.empty:
    # For each method and learning rate, get best batch size
    paper_table = combined_test.groupby(['Method', 'Learning_Rate']).agg({
        'Batch_Size': lambda x: x.iloc[0],
        'Test_Accuracy': 'max',
        'Test_Precision': 'max', 
        'Test_Recall': 'max',
        'Test_F1': 'max',
        'Test_MCC': 'max',
        'Test_AUC': 'max',
        'Test_Sparsity': 'max',
        'Test_Compression_Ratio': 'max',
        'CV_Mean_Accuracy': 'first',
        'CV_Mean_MCC': 'first',
        'CV_Mean_Sparsity': 'first',
        'CV_Mean_Compression_Ratio': 'first',
        'Feature_Stability_Jaccard': 'first'
    }).reset_index()
    
    # Sort by Test MCC
    paper_table = paper_table.sort_values('Test_MCC', ascending=False)
    
    # Save paper table
    paper_table_path = os.path.join(output_dir, "Paper_Best_Results_Table.csv")
    paper_table.to_csv(paper_table_path, index=False)
    
    print("\n📊 BEST RESULTS FOR PAPER (Dynamic Sparsity):")
    print("=" * 80)
    print(paper_table[['Method', 'Learning_Rate', 'Batch_Size', 
                       'Test_Accuracy', 'Test_F1', 'Test_MCC', 'Test_AUC',
                       'Test_Sparsity', 'Test_Compression_Ratio']].to_string(index=False))
    
    print(f"\n✅ Paper table saved to: {paper_table_path}")

# =============================================
# 5. Summary Statistics
# =============================================
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

if 'combined_cv' in locals() and not combined_cv.empty:
    print("\n📊 Best CV Performance by Method (across all LRs and batch sizes):")
    print("-" * 60)
    best_cv = combined_cv.loc[combined_cv.groupby('method')['mean_mcc'].idxmax()]
    best_cv = best_cv.sort_values('mean_mcc', ascending=False)
    for _, row in best_cv.iterrows():
        print(f"  {row['method']:<35}: MCC={row['mean_mcc']:.4f} | "
              f"Sparsity={row.get('mean_sparsity', 0):.4f} | "
              f"Compression={row.get('mean_compression_ratio', 0):.2f}x | "
              f"(LR={row['Learning_Rate']}, BS={row['batch_size']})")

if 'combined_test' in locals() and not combined_test.empty:
    print("\n📊 Best Test Performance by Method (across all LRs and batch sizes):")
    print("-" * 60)
    best_test = combined_test.loc[combined_test.groupby('Method')['Test_MCC'].idxmax()]
    best_test = best_test.sort_values('Test_MCC', ascending=False)
    for _, row in best_test.iterrows():
        print(f"  {row['Method']:<35}: Test MCC={row['Test_MCC']:.4f} | "
              f"Test Sparsity={row.get('Test_Sparsity', 0):.4f} | "
              f"Compression={row.get('Test_Compression_Ratio', 0):.2f}x | "
              f"(LR={row['Learning_Rate']}, BS={row['Batch_Size']})")

# =============================================
# 6. Overall Best Model (Trade-off: MCC vs Compression)
# =============================================
print("\n" + "=" * 70)
print("🏆 OVERALL BEST MODEL")
print("=" * 70)

if 'combined_selected' in locals() and not combined_selected.empty:
    best_overall = combined_selected.iloc[0]
    print(f"\nMethod:              {best_overall['Selected_Method']}")
    print(f"Description:         {best_overall['Description']}")
    print(f"Learning Rate:       {best_overall['Learning_Rate']}")
    print(f"Batch Size:          {best_overall['Batch_Size']}")
    print(f"\n📊 Cross-Validation Performance:")
    print(f"   CV Mean MCC:         {best_overall['CV_Mean_MCC']:.4f} ± {best_overall['CV_Std_MCC']:.4f}")
    print(f"   CV Mean Sparsity:    {best_overall['CV_Mean_Sparsity']:.4f} ± {best_overall['CV_Std_Sparsity']:.4f}")
    print(f"   CV Mean Compression: {best_overall['CV_Mean_Compression_Ratio']:.2f}x")
    print(f"\n📊 Test Performance:")
    print(f"   Test Accuracy:       {best_overall['Test_Accuracy']:.4f}")
    print(f"   Test F1:             {best_overall['Test_F1']:.4f}")
    print(f"   Test MCC:            {best_overall['Test_MCC']:.4f}")
    print(f"   Test AUC:            {best_overall['Test_AUC']:.4f}")
    print(f"   Test Sparsity:       {best_overall['Test_Sparsity']:.4f}")
    print(f"   Test Compression:    {best_overall['Test_Compression_Ratio']:.2f}x")
    print(f"\n📊 Feature Stability:   {best_overall['Feature_Stability_Jaccard']:.4f}")

# =============================================
# 7. Create Sparsity vs Performance Trade-off Table
# =============================================
print("\n" + "=" * 70)
print("SPARSITY vs PERFORMANCE TRADE-OFF")
print("=" * 70)

if 'combined_test' in locals() and not combined_test.empty:
    tradeoff_table = combined_test[['Method', 'Learning_Rate', 'Batch_Size', 
                                     'Test_MCC', 'Test_Sparsity', 'Test_Compression_Ratio']].copy()
    tradeoff_table = tradeoff_table.sort_values('Test_MCC', ascending=False)
    tradeoff_table = tradeoff_table.head(10)
    
    tradeoff_path = os.path.join(output_dir, "Sparsity_Performance_Tradeoff.csv")
    tradeoff_table.to_csv(tradeoff_path, index=False)
    
    print("\n📊 Top 10 Models by Test MCC with Sparsity Metrics:")
    print("-" * 70)
    print(tradeoff_table.to_string(index=False))

print("\n" + "=" * 70)
print("✅ PAPER 3 RESULTS COMBINED!")
print(f"📁 Combined results saved to: {output_dir}")
print("=" * 70)