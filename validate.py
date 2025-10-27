"""
validation script for saved clustering models
loads a trained model and evaluates it on the test set
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from tabulate import tabulate

from src.features import load_data, aggregate_by_ward, prepare_features
from src.modeling import load_model, evaluate_clustering
from sklearn.preprocessing import StandardScaler


def list_saved_models(models_dir='models'):
    """
    list all saved models in the models directory
    
    args:
        models_dir: directory containing saved models
    
    returns:
        list of model metadata dictionaries
    """
    # find all metadata files
    metadata_files = glob.glob(os.path.join(models_dir, '*_metadata.json'))
    
    models = []
    for metadata_file in metadata_files:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            metadata['metadata_path'] = metadata_file
            models.append(metadata)
    
    return sorted(models, key=lambda x: x['timestamp'], reverse=True)


def validate_model(model_path=None, scaler_path=None, models_dir='models'):
    """
    validate a saved clustering model
    
    args:
        model_path: path to saved model file (if None, uses latest)
        scaler_path: path to saved scaler file (if None, uses latest)
        models_dir: directory containing saved models
    """
    print("=" * 60)
    print("MUMBAI WARD CLUSTERING - MODEL VALIDATION")
    print("=" * 60)
    
    # if no paths provided, use the latest model
    if model_path is None or scaler_path is None:
        models = list_saved_models(models_dir)
        
        if not models:
            print("\n❌ no saved models found in 'models/' directory")
            print("   run 'python train.py' first to train a model\n")
            return
        
        latest_model = models[0]
        model_path = latest_model['model_path']
        scaler_path = latest_model['scaler_path']
        metadata_path = latest_model['metadata_path']
        
        print(f"\n📦 loading latest model:")
        print(f"   timestamp: {latest_model['timestamp']}")
        print(f"   clusters: {latest_model['n_clusters']}")
    else:
        metadata_path = model_path.replace('.pkl', '_metadata.json')
    
    # load model and scaler
    print(f"\n[1/4] loading model and scaler...")
    kmeans, scaler = load_model(model_path, scaler_path)
    
    if kmeans is None:
        print(f"\n❌ failed to load model from:")
        print(f"   model: {model_path}")
        print(f"   scaler: {scaler_path}\n")
        return
    
    print(f"  ✓ model loaded: {model_path}")
    print(f"  ✓ scaler loaded: {scaler_path}")
    
    # load metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"  ✓ metadata loaded")
    else:
        metadata = None
        print(f"  ⚠️  metadata not found")
    
    # load and prepare data
    print(f"\n[2/4] loading and preparing data...")
    df = load_data('data/mumbai-suburban.csv')
    ward_df = aggregate_by_ward(df)
    print(f"  ✓ loaded {len(ward_df)} wards")
    
    # prepare features using loaded scaler
    X, feature_cols, _ = prepare_features(ward_df)
    print(f"  ✓ {len(feature_cols)} features prepared")
    
    # make predictions
    print(f"\n[3/4] making predictions...")
    labels = kmeans.predict(X)
    print(f"  ✓ predicted clusters for {len(labels)} wards")
    
    # evaluate
    print(f"\n[4/4] evaluating model performance...")
    metrics = evaluate_clustering(X, labels)
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    # display metrics
    print(f"\n📊 overall performance metrics:")
    print(f"  • silhouette score: {metrics['silhouette_score']:.4f}")
    print(f"  • davies-bouldin index: {metrics['davies_bouldin_score']:.4f}")
    
    # compare with training metrics if available
    if metadata and 'metrics' in metadata:
        print("\n📈 comparison with training:")
        
        comparison_data = []
        for dataset in ['train', 'validation', 'test']:
            if dataset in metadata['metrics']:
                dataset_metrics = metadata['metrics'][dataset]
                comparison_data.append([
                    dataset.capitalize(),
                    f"{dataset_metrics['silhouette_score']:.4f}",
                    f"{dataset_metrics['davies_bouldin_score']:.4f}"
                ])
        
        comparison_data.append([
            'Full Dataset',
            f"{metrics['silhouette_score']:.4f}",
            f"{metrics['davies_bouldin_score']:.4f}"
        ])
        
        headers = ['Dataset', 'Silhouette ↑', 'Davies-Bouldin ↓']
        print("\n" + tabulate(comparison_data, headers=headers, tablefmt='grid'))
    
    # cluster distribution
    print("\n📊 cluster distribution:")
    cluster_dist = pd.Series(labels).value_counts().sort_index()
    
    dist_data = []
    for cluster_id, count in cluster_dist.items():
        dist_data.append([
            f'Cluster {cluster_id}',
            count,
            f'{count/len(labels)*100:.1f}%'
        ])
    
    headers = ['Cluster', 'Count', 'Percentage']
    print("\n" + tabulate(dist_data, headers=headers, tablefmt='grid'))
    
    # cluster statistics
    print("\n📋 cluster statistics:")
    ward_df_labeled = ward_df.copy()
    ward_df_labeled['cluster'] = labels
    
    cluster_stats = ward_df_labeled.groupby('cluster').agg({
        'total_workers': 'sum',
        'total_establishments': 'sum',
        'avg_workers_per_est': 'mean',
        'pct_tertiary': 'mean',
        'pct_female': 'mean'
    }).round(2)
    
    print("\n" + cluster_stats.to_string())
    
    # model summary
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    
    if metadata:
        print(f"\n🏷️  model information:")
        print(f"  • trained: {metadata['timestamp']}")
        print(f"  • clusters: {metadata['n_clusters']}")
        print(f"  • features: {metadata['n_features']}")
        print(f"  • training set size: {metadata['data_split']['train_size']} wards")
        
        if 'evaluation' in metadata:
            print(f"\n💡 training insights:")
            print(f"  • recommended k: {metadata['evaluation']['best_k_by_silhouette']}")
            print(f"  • best silhouette: {metadata['evaluation']['best_silhouette_score']:.4f}")
    
    print("\n✅ validation complete!\n")
    
    return metrics, labels, ward_df_labeled


def compare_models(models_dir='models'):
    """
    compare all saved models
    
    args:
        models_dir: directory containing saved models
    """
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    models = list_saved_models(models_dir)
    
    if not models:
        print("\n❌ no saved models found in 'models/' directory\n")
        return
    
    print(f"\n📦 found {len(models)} saved model(s)\n")
    
    comparison_data = []
    for i, model in enumerate(models, 1):
        timestamp = model['timestamp'].split('T')[0] + ' ' + model['timestamp'].split('T')[1][:8]
        
        # get test metrics if available
        if 'metrics' in model and 'test' in model['metrics']:
            sil_score = f"{model['metrics']['test']['silhouette_score']:.4f}"
            db_score = f"{model['metrics']['test']['davies_bouldin_score']:.4f}"
        else:
            sil_score = 'N/A'
            db_score = 'N/A'
        
        comparison_data.append([
            i,
            timestamp,
            model['n_clusters'],
            sil_score,
            db_score,
            model['data_split']['train_size']
        ])
    
    headers = ['#', 'Timestamp', 'K', 'Silhouette ↑', 'DB Index ↓', 'Train Size']
    print(tabulate(comparison_data, headers=headers, tablefmt='grid'))
    
    print("\n💡 higher silhouette score and lower davies-bouldin index are better\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='validate mumbai ward clustering model')
    parser.add_argument('--model', type=str, default=None, help='path to model file (default: latest)')
    parser.add_argument('--scaler', type=str, default=None, help='path to scaler file (default: latest)')
    parser.add_argument('--compare', action='store_true', help='compare all saved models')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    else:
        validate_model(model_path=args.model, scaler_path=args.scaler)
