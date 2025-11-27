import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

"""
QUICK DIAGNOSTIC: Is your accuracy real or inflated by data leakage?

This script checks if overlapping segments cause memorization instead of learning.
"""

def check_segment_similarity_in_splits(X, fold_indices, num_checks=50):
    """
    Check if very similar segments appear in both train and test sets
    High similarity = DATA LEAKAGE = Inflated accuracy
    """
    
    print("="*70)
    print("🔍 DATA LEAKAGE DIAGNOSTIC TEST")
    print("="*70)
    print("\nChecking if train and test sets have overlapping segments...")
    print("(This would explain artificially high accuracy)\n")
    
    all_fold_results = []
    
    for fold_no, (train_val_idx, test_idx) in enumerate(fold_indices, start=1):
        print(f"\n{'─'*70}")
        print(f"📊 FOLD {fold_no}")
        print(f"{'─'*70}")
        
        X_train = X[train_val_idx]
        X_test = X[test_idx]
        
        print(f"Train size: {len(X_train)} segments")
        print(f"Test size: {len(X_test)} segments")
        
        # Check correlation between test and train segments
        high_correlation_count = 0
        perfect_match_count = 0
        correlations = []
        
        # Sample random test segments to check
        test_sample_indices = np.random.choice(len(X_test), 
                                              min(num_checks, len(X_test)), 
                                              replace=False)
        
        print(f"\nChecking {len(test_sample_indices)} random test segments...")
        
        for test_idx_sample in test_sample_indices:
            test_seg = X_test[test_idx_sample].flatten()
            
            # Find most similar segment in training set
            max_corr = -1
            
            # Sample training segments (checking all would be too slow)
            train_sample_indices = np.random.choice(len(X_train), 
                                                   min(200, len(X_train)), 
                                                   replace=False)
            
            for train_idx_sample in train_sample_indices:
                train_seg = X_train[train_idx_sample].flatten()
                
                # Calculate Pearson correlation
                corr = np.corrcoef(test_seg, train_seg)[0, 1]
                
                if corr > max_corr:
                    max_corr = corr
            
            correlations.append(max_corr)
            
            # Thresholds for leakage detection
            if max_corr > 0.99:  # Almost identical
                perfect_match_count += 1
            elif max_corr > 0.90:  # Very similar
                high_correlation_count += 1
        
        # Results for this fold
        correlations = np.array(correlations)
        
        print(f"\n📈 RESULTS:")
        print(f"  Mean max correlation: {correlations.mean():.4f}")
        print(f"  Median max correlation: {np.median(correlations):.4f}")
        print(f"  Max correlation found: {correlations.max():.4f}")
        print(f"  Min correlation found: {correlations.min():.4f}")
        
        print(f"\n⚠️ LEAKAGE INDICATORS:")
        print(f"  Perfect matches (>0.99): {perfect_match_count}/{len(test_sample_indices)}")
        print(f"  High similarity (>0.90): {high_correlation_count}/{len(test_sample_indices)}")
        
        # Interpretation
        leakage_score = (perfect_match_count + high_correlation_count) / len(test_sample_indices)
        
        if leakage_score > 0.5:
            print(f"\n  🚨 SEVERE LEAKAGE: {leakage_score*100:.1f}% of test segments highly similar to train")
            verdict = "SEVERE LEAKAGE"
        elif leakage_score > 0.2:
            print(f"\n  ⚠️ MODERATE LEAKAGE: {leakage_score*100:.1f}% of test segments similar to train")
            verdict = "MODERATE LEAKAGE"
        elif leakage_score > 0.05:
            print(f"\n  ⚡ MINOR LEAKAGE: {leakage_score*100:.1f}% of test segments similar to train")
            verdict = "MINOR LEAKAGE"
        else:
            print(f"\n  ✅ MINIMAL LEAKAGE: Only {leakage_score*100:.1f}% similarity")
            verdict = "GOOD"
        
        all_fold_results.append({
            'fold': fold_no,
            'mean_corr': correlations.mean(),
            'median_corr': np.median(correlations),
            'max_corr': correlations.max(),
            'leakage_score': leakage_score,
            'verdict': verdict,
            'correlations': correlations
        })
    
    return all_fold_results


def visualize_leakage_analysis(fold_results):
    """
    Create visualizations to show leakage severity
    """
    print("\n" + "="*70)
    print("📊 CREATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Mean correlation per fold
    ax = axes[0, 0]
    folds = [r['fold'] for r in fold_results]
    mean_corrs = [r['mean_corr'] for r in fold_results]
    
    colors = ['red' if c > 0.9 else 'orange' if c > 0.7 else 'yellow' if c > 0.5 else 'green' 
              for c in mean_corrs]
    
    ax.bar(folds, mean_corrs, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0.9, color='red', linestyle='--', label='Severe Leakage (>0.9)', linewidth=2)
    ax.axhline(y=0.7, color='orange', linestyle='--', label='Moderate Leakage (>0.7)', linewidth=2)
    ax.axhline(y=0.5, color='yellow', linestyle='--', label='Minor Leakage (>0.5)', linewidth=2)
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Correlation', fontsize=12, fontweight='bold')
    ax.set_title('Mean Train-Test Similarity per Fold', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 2. Distribution of correlations (all folds combined)
    ax = axes[0, 1]
    all_corrs = np.concatenate([r['correlations'] for r in fold_results])
    
    ax.hist(all_corrs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0.9, color='red', linestyle='--', linewidth=2, label='Severe (>0.9)')
    ax.axvline(x=0.7, color='orange', linestyle='--', linewidth=2, label='Moderate (>0.7)')
    ax.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Max Correlations', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Leakage score per fold
    ax = axes[1, 0]
    leakage_scores = [r['leakage_score'] * 100 for r in fold_results]
    verdicts = [r['verdict'] for r in fold_results]
    
    colors_leak = ['darkred' if v == 'SEVERE LEAKAGE' else 
                   'orange' if v == 'MODERATE LEAKAGE' else 
                   'yellow' if v == 'MINOR LEAKAGE' else 'green' 
                   for v in verdicts]
    
    bars = ax.bar(folds, leakage_scores, color=colors_leak, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Leakage Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Data Leakage Score per Fold', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    overall_mean = np.mean(mean_corrs)
    overall_leakage = np.mean(leakage_scores)
    
    if overall_mean > 0.9:
        overall_verdict = "🚨 SEVERE LEAKAGE DETECTED"
        color = 'darkred'
    elif overall_mean > 0.7:
        overall_verdict = "⚠️ MODERATE LEAKAGE DETECTED"
        color = 'orange'
    elif overall_mean > 0.5:
        overall_verdict = "⚡ MINOR LEAKAGE DETECTED"
        color = 'gold'
    else:
        overall_verdict = "✅ MINIMAL LEAKAGE"
        color = 'green'
    
    summary_text = f"""
    OVERALL ASSESSMENT
    {'='*40}
    
    {overall_verdict}
    
    Average Correlation: {overall_mean:.4f}
    Average Leakage Score: {overall_leakage:.1f}%
    
    Folds with severe leakage: {sum(1 for v in verdicts if v == 'SEVERE LEAKAGE')}/5
    Folds with moderate leakage: {sum(1 for v in verdicts if v == 'MODERATE LEAKAGE')}/5
    Folds with minor leakage: {sum(1 for v in verdicts if v == 'MINOR LEAKAGE')}/5
    Folds without leakage: {sum(1 for v in verdicts if v == 'GOOD')}/5
    
    {'─'*40}
    INTERPRETATION:
    
    • Correlation > 0.9: Segments are nearly identical
      → Model is memorizing, not learning
      → Accuracy is INFLATED
    
    • Correlation 0.7-0.9: Segments are very similar
      → Some overfitting likely
      → Accuracy is somewhat inflated
    
    • Correlation < 0.5: Segments are different
      → Genuine learning
      → Accuracy is TRUSTWORTHY
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor=color, alpha=0.3, pad=1))
    
    plt.tight_layout()
    plt.savefig('results/leakage_diagnostic.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved: results/leakage_diagnostic.png")
    plt.show()


def final_verdict(fold_results, model_accuracy):
    """
    Give final verdict on whether accuracy is trustworthy
    """
    print("\n" + "="*70)
    print("🎯 FINAL VERDICT")
    print("="*70)
    
    mean_corrs = [r['mean_corr'] for r in fold_results]
    overall_mean = np.mean(mean_corrs)
    overall_leakage = np.mean([r['leakage_score'] for r in fold_results])
    
    print(f"\nYour Model Accuracy: {model_accuracy:.2%}")
    print(f"Average Train-Test Similarity: {overall_mean:.4f}")
    print(f"Average Leakage Score: {overall_leakage:.2%}\n")
    
    if overall_mean > 0.9:
        print("❌ VERDICT: Your accuracy is INFLATED by data leakage")
        print("\n📋 What's happening:")
        print("  • Overlapping segments from same files are in train AND test")
        print("  • Model is memorizing patterns, not learning generalizable features")
        print("  • Test set is NOT truly unseen data")
        print("\n💡 Expected real accuracy: 5-15% lower")
        print("\n✅ Action needed: Implement file-level split (use GroupKFold)")
        
    elif overall_mean > 0.7:
        print("⚠️ VERDICT: Your accuracy is MODERATELY inflated")
        print("\n📋 What's happening:")
        print("  • Some overlapping segments leak between train and test")
        print("  • Model has some advantage from seeing similar data")
        print("  • Results are overly optimistic")
        print("\n💡 Expected real accuracy: 3-8% lower")
        print("\n✅ Action needed: Implement file-level split for publication")
        
    elif overall_mean > 0.5:
        print("⚡ VERDICT: Your accuracy is SLIGHTLY inflated")
        print("\n📋 What's happening:")
        print("  • Minor overlap exists but not severe")
        print("  • Model is mostly learning real patterns")
        print("  • Results are mostly trustworthy but could be better")
        print("\n💡 Expected real accuracy: 1-3% lower")
        print("\n✅ Recommendation: Use file-level split for best practice")
        
    else:
        print("✅ VERDICT: Your accuracy is TRUSTWORTHY")
        print("\n📋 What's happening:")
        print("  • Train and test sets are properly separated")
        print("  • Model is learning generalizable patterns")
        print("  • No significant data leakage detected")
        print("\n💡 Your results are publishable!")
    
    print("\n" + "="*70)


# ===================== MAIN EXECUTION =====================
if __name__ == "__main__":
    import os
    
    print("\n" + "="*70)
    print("🔬 DATA LEAKAGE DIAGNOSTIC TOOL")
    print("="*70)
    print("\nThis tool checks if your high accuracy is real or due to data leakage")
    print("from overlapping segments.\n")
    
    # Check if files exist
    if not os.path.exists("preprocessed/ALL_X.npy"):
        print("❌ Error: preprocessed/ALL_X.npy not found!")
        print("Please run preprocessing first.")
        exit()
    
    if not os.path.exists("results/fold_indices.npy"):
        print("❌ Error: results/fold_indices.npy not found!")
        print("Please run training first.")
        exit()
    
    # Load data
    print("📂 Loading data...")
    X = np.load("preprocessed/ALL_X.npy")
    y = np.load("preprocessed/ALL_y.npy")
    fold_indices = np.load("results/fold_indices.npy", allow_pickle=True)
    
    print(f"✅ Loaded {len(X)} segments")
    
    # Run leakage check
    print("\n🔍 Starting leakage analysis...")
    print("(This may take 1-2 minutes...)\n")
    
    fold_results = check_segment_similarity_in_splits(X, fold_indices, num_checks=50)
    
    # Visualize
    visualize_leakage_analysis(fold_results)
    
    # Load your actual model accuracy from training
    # You need to update this with your actual accuracy
    try:
        # Try to extract accuracy from your results
        # Update this line with your actual model accuracy
        model_accuracy = 0.95  # REPLACE WITH YOUR ACTUAL ACCURACY
        print(f"\n⚠️ Using placeholder accuracy: {model_accuracy:.2%}")
        print("Update line 372 in the script with your actual accuracy!")
    except:
        model_accuracy = 0.95
    
    # Final verdict
    final_verdict(fold_results, model_accuracy)
    
    print("\n" + "="*70)
    print("✅ DIAGNOSTIC COMPLETE")
    print("="*70)
    print("\nCheck 'results/leakage_diagnostic.png' for visualizations")