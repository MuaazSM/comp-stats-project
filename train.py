"""
training script for mumbai ward clustering model
includes train/test/validation split and model evaluation
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

from src.features import load_data, aggregate_by_ward, prepare_features
from src.modeling import (
    fit_kmeans,
    evaluate_clustering,
    compute_elbow_curve,
    compute_silhouette_scores,
    save_model
)


def split_data(ward_df, test_size=0.2, val_size=0.1, random_state=42):
    """
    split ward data into train/validation/test sets
    
    args:
        ward_df: ward-level dataframe
        test_size: proportion for test set
        val_size: proportion for validation set
        random_state: random seed
    
    returns:
        train, validation, and test dataframes
    """
    # first split: separate test set
    train_val_df, test_df = train_test_split(
        ward_df, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # second split: separate validation from training
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def train_model(n_clusters=4, test_size=0.2, val_size=0.1, random_state=42, max_k_eval=10):
    """
    complete training pipeline with evaluation
    
    args:
        n_clusters: number of clusters for final model
        test_size: proportion for test set
        val_size: proportion for validation set
        random_state: random seed
        max_k_eval: maximum k for evaluation metrics
    """
    print("=" * 60)
    print("MUMBAI WARD CLUSTERING - TRAINING PIPELINE")
    print("=" * 60)
    
    # load and aggregate data
    print("\n[1/6] loading raw data...")
    df = load_data('data/mumbai-suburban.csv')
    print(f"  ✓ loaded {len(df):,} records")
    
    print("\n[2/6] aggregating by ward...")
    ward_df = aggregate_by_ward(df)
    print(f"  ✓ aggregated to {len(ward_df)} wards")
    
    # split data
    print(f"\n[3/6] splitting data (train: {1-test_size-val_size:.0%}, val: {val_size:.0%}, test: {test_size:.0%})...")
    train_df, val_df, test_df = split_data(ward_df, test_size, val_size, random_state)
    print(f"  ✓ train: {len(train_df)} wards")
    print(f"  ✓ validation: {len(val_df)} wards")
    print(f"  ✓ test: {len(test_df)} wards")
    
    # prepare features for each set
    print("\n[4/6] preparing features...")
    X_train_scaled, feature_cols, scaler = prepare_features(train_df)
    print(f"  ✓ {len(feature_cols)} features extracted and scaled")
    
    # transform validation and test sets using the same scaler
    X_val = val_df[feature_cols].values
    X_val_scaled = scaler.transform(X_val)
    
    X_test = test_df[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    
    # evaluate different k values on validation set
    print(f"\n[5/6] evaluating optimal k (range: 2-{max_k_eval}) on validation set...")
    k_values, sse_values = compute_elbow_curve(X_val_scaled, max_k=max_k_eval)
    k_values_sil, sil_values = compute_silhouette_scores(X_val_scaled, max_k=max_k_eval)
    
    # find best k based on silhouette score
    best_k_idx = np.argmax(sil_values)
    best_k = k_values_sil[best_k_idx]
    best_sil = sil_values[best_k_idx]
    
    print(f"  ✓ elbow curve computed")
    print(f"  ✓ silhouette scores computed")
    print(f"  ✓ best k based on silhouette: {best_k} (score: {best_sil:.3f})")
    
    # train final model with specified n_clusters
    print(f"\n[6/6] training final model with k={n_clusters}...")
    kmeans, train_labels = fit_kmeans(X_train_scaled, n_clusters=n_clusters)
    print(f"  ✓ model trained on {len(train_df)} wards")
    
    # evaluate on all three sets
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # training set evaluation
    train_metrics = evaluate_clustering(X_train_scaled, train_labels)
    print(f"\n📊 training set metrics:")
    print(f"  • silhouette score: {train_metrics['silhouette_score']:.4f}")
    print(f"  • davies-bouldin index: {train_metrics['davies_bouldin_score']:.4f}")
    
    # validation set evaluation
    val_labels = kmeans.predict(X_val_scaled)
    val_metrics = evaluate_clustering(X_val_scaled, val_labels)
    print(f"\n📊 validation set metrics:")
    print(f"  • silhouette score: {val_metrics['silhouette_score']:.4f}")
    print(f"  • davies-bouldin index: {val_metrics['davies_bouldin_score']:.4f}")
    
    # test set evaluation
    test_labels = kmeans.predict(X_test_scaled)
    test_metrics = evaluate_clustering(X_test_scaled, test_labels)
    print(f"\n📊 test set metrics:")
    print(f"  • silhouette score: {test_metrics['silhouette_score']:.4f}")
    print(f"  • davies-bouldin index: {test_metrics['davies_bouldin_score']:.4f}")
    
    # cluster distribution across sets
    print("\n" + "=" * 60)
    print("CLUSTER DISTRIBUTION")
    print("=" * 60)
    
    print("\n📈 training set:")
    train_dist = pd.Series(train_labels).value_counts().sort_index()
    for cluster_id, count in train_dist.items():
        print(f"  • cluster {cluster_id}: {count} wards ({count/len(train_labels)*100:.1f}%)")
    
    print("\n📈 validation set:")
    val_dist = pd.Series(val_labels).value_counts().sort_index()
    for cluster_id, count in val_dist.items():
        print(f"  • cluster {cluster_id}: {count} wards ({count/len(val_labels)*100:.1f}%)")
    
    print("\n📈 test set:")
    test_dist = pd.Series(test_labels).value_counts().sort_index()
    for cluster_id, count in test_dist.items():
        print(f"  • cluster {cluster_id}: {count} wards ({count/len(test_labels)*100:.1f}%)")
    
    # save model and scaler
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    model_path, scaler_path = save_model(kmeans, scaler, n_clusters)
    print(f"\n✅ model saved: {model_path}")
    print(f"✅ scaler saved: {scaler_path}")
    
    # save training metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_clusters': n_clusters,
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'data_split': {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'test_ratio': test_size,
            'val_ratio': val_size
        },
        'metrics': {
            'train': {
                'silhouette_score': float(train_metrics['silhouette_score']),
                'davies_bouldin_score': float(train_metrics['davies_bouldin_score'])
            },
            'validation': {
                'silhouette_score': float(val_metrics['silhouette_score']),
                'davies_bouldin_score': float(val_metrics['davies_bouldin_score'])
            },
            'test': {
                'silhouette_score': float(test_metrics['silhouette_score']),
                'davies_bouldin_score': float(test_metrics['davies_bouldin_score'])
            }
        },
        'evaluation': {
            'best_k_by_silhouette': int(best_k),
            'best_silhouette_score': float(best_sil),
            'k_range_evaluated': f'2-{max_k_eval}'
        },
        'model_path': model_path,
        'scaler_path': scaler_path
    }
    
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ metadata saved: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n💡 recommended k: {best_k} (based on validation silhouette score)")
    print(f"💡 used k: {n_clusters}")
    
    if n_clusters != best_k:
        print(f"\n⚠️  note: you can retrain with k={best_k} for potentially better results")
    
    print("\n🚀 next steps:")
    print("  1. review the metadata file")
    print("  2. run 'streamlit run app.py' to visualize results")
    print("  3. experiment with different k values if needed\n")
    
    return kmeans, scaler, metadata


if __name__ == "__main__":
    # parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='train mumbai ward clustering model')
    parser.add_argument('--k', type=int, default=4, help='number of clusters (default: 4)')
    parser.add_argument('--test-size', type=float, default=0.2, help='test set proportion (default: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.1, help='validation set proportion (default: 0.1)')
    parser.add_argument('--max-k-eval', type=int, default=10, help='max k for evaluation (default: 10)')
    parser.add_argument('--random-state', type=int, default=42, help='random seed (default: 42)')
    
    args = parser.parse_args()
    
    # run training
    train_model(
        n_clusters=args.k,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        max_k_eval=args.max_k_eval
    )
