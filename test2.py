import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import seaborn as sns

# -------------------- Load Your Data --------------------
X = np.load(r"preprocessed\ALL_X.npy")
y = np.load(r"preprocessed\ALL_y.npy")

# Convert to binary
y_encoded = np.where(y == 'ICTAL', 0, 1)

# Load fold indices
fold_indices = np.load("results(Normal_VS_All)/fold_indices.npy", allow_pickle=True)

print("="*80)
print("🔍 DATA LEAKAGE DETECTION - COMPREHENSIVE ANALYSIS")
print("="*80)
print(f"Total samples: {len(X)}")
print(f"Sample shape: {X.shape}")
print(f"Label distribution: {np.unique(y_encoded, return_counts=True)}")
print()

# -------------------- CHECK 1: Exact Duplicates --------------------
def check_exact_duplicates(X):
    """Check for identical samples in dataset"""
    print("\n" + "="*80)
    print("CHECK 1: EXACT DUPLICATES")
    print("="*80)
    
    unique_samples, unique_indices, inverse_indices = np.unique(
        X.reshape(X.shape[0], -1), 
        axis=0, 
        return_index=True, 
        return_inverse=True
    )
    
    n_duplicates = len(X) - len(unique_samples)
    duplicate_percentage = (n_duplicates / len(X)) * 100
    
    print(f"Total samples: {len(X)}")
    print(f"Unique samples: {len(unique_samples)}")
    print(f"Duplicate samples: {n_duplicates}")
    print(f"Duplicate percentage: {duplicate_percentage:.2f}%")
    
    if n_duplicates > 0:
        print("⚠️  WARNING: Found exact duplicates in dataset!")
        # Find which samples are duplicated
        from collections import Counter
        counter = Counter(inverse_indices)
        duplicated = [idx for idx, count in counter.items() if count > 1]
        print(f"Number of duplicated patterns: {len(duplicated)}")
    else:
        print("✅ No exact duplicates found")
    
    return n_duplicates

# -------------------- CHECK 2: High Similarity Between Splits --------------------
def check_cross_split_similarity(X_train, X_val, X_test, threshold=0.95):
    """Check if samples in different splits are too similar (potential overlap issue)"""
    print("\n" + "="*80)
    print("CHECK 2: CROSS-SPLIT SIMILARITY (Detecting Overlapping Segments)")
    print("="*80)
    print(f"Similarity threshold: {threshold}")
    print(f"(Checking if segments from different splits have >{threshold*100}% similarity)")
    
    # Randomly sample to speed up computation
    n_samples = min(500, len(X_train), len(X_val), len(X_test))
    
    train_sample = X_train[np.random.choice(len(X_train), n_samples, replace=False)]
    val_sample = X_val[np.random.choice(len(X_val), n_samples, replace=False)]
    test_sample = X_test[np.random.choice(len(X_test), n_samples, replace=False)]
    
    # Flatten for cosine similarity
    train_flat = train_sample.reshape(n_samples, -1)
    val_flat = val_sample.reshape(n_samples, -1)
    test_flat = test_sample.reshape(n_samples, -1)
    
    # Calculate similarities
    print("\nCalculating Train-Val similarity...")
    sim_train_val = cosine_similarity(train_flat, val_flat)
    max_sim_train_val = np.max(sim_train_val)
    mean_sim_train_val = np.mean(sim_train_val)
    high_sim_train_val = np.sum(sim_train_val > threshold)
    
    print("\nCalculating Train-Test similarity...")
    sim_train_test = cosine_similarity(train_flat, test_flat)
    max_sim_train_test = np.max(sim_train_test)
    mean_sim_train_test = np.mean(sim_train_test)
    high_sim_train_test = np.sum(sim_train_test > threshold)
    
    print("\nCalculating Val-Test similarity...")
    sim_val_test = cosine_similarity(val_flat, test_flat)
    max_sim_val_test = np.max(sim_val_test)
    mean_sim_val_test = np.mean(sim_val_test)
    high_sim_val_test = np.sum(sim_val_test > threshold)
    
    # Results
    print("\n📊 SIMILARITY RESULTS:")
    print(f"\nTrain-Val:")
    print(f"  Max similarity: {max_sim_train_val:.4f}")
    print(f"  Mean similarity: {mean_sim_train_val:.4f}")
    print(f"  Pairs above {threshold}: {high_sim_train_val} / {n_samples*n_samples}")
    
    print(f"\nTrain-Test:")
    print(f"  Max similarity: {max_sim_train_test:.4f}")
    print(f"  Mean similarity: {mean_sim_train_test:.4f}")
    print(f"  Pairs above {threshold}: {high_sim_train_test} / {n_samples*n_samples}")
    
    print(f"\nVal-Test:")
    print(f"  Max similarity: {max_sim_val_test:.4f}")
    print(f"  Mean similarity: {mean_sim_val_test:.4f}")
    print(f"  Pairs above {threshold}: {high_sim_val_test} / {n_samples*n_samples}")
    
    # Warning
    if high_sim_train_val > 10 or high_sim_train_test > 10 or high_sim_val_test > 10:
        print(f"\n⚠️  WARNING: Found {high_sim_train_val + high_sim_train_test + high_sim_val_test} highly similar pairs!")
        print("This suggests potential data leakage from overlapping segments!")
    else:
        print("\n✅ No significant cross-split similarity detected")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot similarity distributions
    axes[0].hist(sim_train_val.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
    axes[0].set_title(f'Train-Val Similarity\nMax={max_sim_train_val:.3f}, Mean={mean_sim_train_val:.3f}')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    axes[1].hist(sim_train_test.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
    axes[1].set_title(f'Train-Test Similarity\nMax={max_sim_train_test:.3f}, Mean={mean_sim_train_test:.3f}')
    axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    axes[2].hist(sim_val_test.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[2].axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
    axes[2].set_title(f'Val-Test Similarity\nMax={max_sim_val_test:.3f}, Mean={mean_sim_val_test:.3f}')
    axes[2].set_xlabel('Cosine Similarity')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("results(Normal_VS_All)/similarity_analysis.png", dpi=150)
    print("\n📁 Similarity plots saved to: results(Normal_VS_All)/similarity_analysis.png")
    plt.show()
    
    return {
        'train_val': (max_sim_train_val, mean_sim_train_val, high_sim_train_val),
        'train_test': (max_sim_train_test, mean_sim_train_test, high_sim_train_test),
        'val_test': (max_sim_val_test, mean_sim_val_test, high_sim_val_test)
    }

# -------------------- CHECK 3: Overlap Detection (Segment-Level) --------------------
def check_temporal_overlap(fold_indices, X, segment_length=347):
    """
    Check if consecutive segments from same original signal appear in different splits
    This detects the 50% overlap issue
    """
    print("\n" + "="*80)
    print("CHECK 3: TEMPORAL OVERLAP DETECTION")
    print("="*80)
    print("(Checking if overlapping segments from same recording appear in different splits)")
    
    # This is a heuristic check
    # If we have 50% overlap, consecutive segments will be very similar
    
    for fold_no, (train_val_idx, test_idx) in enumerate(fold_indices[:1], start=1):  # Check first fold
        print(f"\n🔹 Analyzing Fold {fold_no}:")
        
        from sklearn.model_selection import train_test_split
        X_train_val = X[train_val_idx]
        y_train_val = y_encoded[train_val_idx]
        
        train_idx, val_idx = train_test_split(
            range(len(X_train_val)), 
            test_size=0.1765, 
            stratify=y_train_val, 
            random_state=42
        )
        
        X_train = X_train_val[train_idx]
        X_val = X_train_val[val_idx]
        X_test = X[test_idx]
        
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Check similarity within each split (should be low if no overlap)
        print("\n  Checking intra-split similarity...")
        
        # Sample random pairs within training set
        n_pairs = min(1000, len(X_train) - 1)
        train_flat = X_train.reshape(len(X_train), -1)
        
        similarities = []
        for _ in range(n_pairs):
            idx1, idx2 = np.random.choice(len(X_train), 2, replace=False)
            sim = cosine_similarity([train_flat[idx1]], [train_flat[idx2]])[0, 0]
            similarities.append(sim)
        
        mean_intra_sim = np.mean(similarities)
        max_intra_sim = np.max(similarities)
        
        print(f"  Mean intra-train similarity: {mean_intra_sim:.4f}")
        print(f"  Max intra-train similarity: {max_intra_sim:.4f}")
        
        if mean_intra_sim > 0.7:
            print("  ⚠️  High intra-split similarity suggests overlapping segments!")
        else:
            print("  ✅ Normal intra-split similarity")

# -------------------- CHECK 4: Index Overlap --------------------
def check_index_overlap(fold_indices):
    """Verify that train/val/test indices don't overlap"""
    print("\n" + "="*80)
    print("CHECK 4: INDEX OVERLAP VERIFICATION")
    print("="*80)
    
    all_clear = True
    
    for fold_no, (train_val_idx, test_idx) in enumerate(fold_indices, start=1):
        # Check train_val vs test
        overlap = set(train_val_idx) & set(test_idx)
        
        if len(overlap) > 0:
            print(f"❌ Fold {fold_no}: Found {len(overlap)} overlapping indices between train_val and test!")
            all_clear = False
        else:
            print(f"✅ Fold {fold_no}: No index overlap between train_val and test")
    
    if all_clear:
        print("\n✅ All folds have proper index separation")
    else:
        print("\n❌ WARNING: Index overlap detected!")
    
    return all_clear

# -------------------- CHECK 5: Visualize Sample Comparison --------------------
def visualize_sample_comparison(X_train, X_val, X_test):
    """Visualize samples from each split to check if they look different"""
    print("\n" + "="*80)
    print("CHECK 5: VISUAL SAMPLE COMPARISON")
    print("="*80)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    
    # Plot 3 random samples from each split
    for i in range(3):
        # Train
        idx_train = np.random.randint(0, len(X_train))
        axes[0, i].plot(X_train[idx_train].flatten(), linewidth=0.8)
        axes[0, i].set_title(f'Train Sample {idx_train}')
        axes[0, i].set_ylabel('Amplitude')
        axes[0, i].grid(True, alpha=0.3)
        
        # Val
        idx_val = np.random.randint(0, len(X_val))
        axes[1, i].plot(X_val[idx_val].flatten(), linewidth=0.8, color='orange')
        axes[1, i].set_title(f'Val Sample {idx_val}')
        axes[1, i].set_ylabel('Amplitude')
        axes[1, i].grid(True, alpha=0.3)
        
        # Test
        idx_test = np.random.randint(0, len(X_test))
        axes[2, i].plot(X_test[idx_test].flatten(), linewidth=0.8, color='green')
        axes[2, i].set_title(f'Test Sample {idx_test}')
        axes[2, i].set_xlabel('Sample Index')
        axes[2, i].set_ylabel('Amplitude')
        axes[2, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results(Normal_VS_All)/sample_comparison.png", dpi=150)
    print("📁 Sample comparison saved to: results(Normal_VS_All)/sample_comparison.png")
    plt.show()

# -------------------- RUN ALL CHECKS --------------------
def run_all_checks():
    """Run all leakage detection checks"""
    
    # Check 1: Exact duplicates
    n_duplicates = check_exact_duplicates(X)
    
    # Check 4: Index overlap (run before split)
    check_index_overlap(fold_indices)
    
    # Get first fold for detailed analysis
    train_val_idx, test_idx = fold_indices[0]
    X_train_val, X_test = X[train_val_idx], X[test_idx]
    y_train_val, y_test = y_encoded[train_val_idx], y_encoded[test_idx]
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42
    )
    
    print(f"\n📊 Split sizes for Fold 1:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    # Check 2: Cross-split similarity
    similarity_results = check_cross_split_similarity(X_train, X_val, X_test, threshold=0.95)
    
    # Check 3: Temporal overlap
    check_temporal_overlap(fold_indices, X)
    
    # Check 5: Visual comparison
    visualize_sample_comparison(X_train, X_val, X_test)
    
    # -------------------- FINAL REPORT --------------------
    print("\n" + "="*80)
    print("📋 FINAL LEAKAGE DETECTION REPORT")
    print("="*80)
    
    issues_found = []
    
    if n_duplicates > 0:
        issues_found.append(f"❌ Found {n_duplicates} exact duplicates")
    
    for split_name, (max_sim, mean_sim, high_count) in similarity_results.items():
        if high_count > 10:
            issues_found.append(f"❌ High similarity in {split_name}: {high_count} pairs > 0.95")
    
    if len(issues_found) == 0:
        print("✅ ✅ ✅ NO MAJOR DATA LEAKAGE DETECTED!")
        print("\nYour train/val/test splits appear to be properly separated.")
        print("The high accuracy (99.2%) is likely due to:")
        print("  1. Good model architecture")
        print("  2. Clean, well-separated data")
        print("  3. Effective preprocessing")
    else:
        print("⚠️  POTENTIAL DATA LEAKAGE ISSUES FOUND:")
        for issue in issues_found:
            print(f"  {issue}")
        print("\n🔧 RECOMMENDED FIXES:")
        print("  1. Remove the 50% overlap in segmentation (use step_size = segment_length)")
        print("  2. Remove exact duplicates from dataset")
        print("  3. Re-run this check after fixes")
    
    print("="*80)

# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":
    run_all_checks()