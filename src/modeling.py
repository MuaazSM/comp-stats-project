import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA


def compute_elbow_curve(X, max_k=10):
    """
    compute SSE for different k values (elbow method)
    """
    # calculate SSE for each k
    k_values = range(2, max_k + 1)
    sse_values = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        sse_values.append(kmeans.inertia_)
    
    return list(k_values), sse_values


def compute_silhouette_scores(X, max_k=10):
    """
    compute silhouette score for different k values
    """
    # calculate silhouette score for each k
    k_values = range(2, max_k + 1)
    silhouette_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    return list(k_values), silhouette_scores


def fit_kmeans(X, n_clusters=4):
    """
    fit k-means clustering model
    """
    # train k-means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    return kmeans, labels


def evaluate_clustering(X, labels):
    """
    compute clustering evaluation metrics
    """
    # compute evaluation metrics
    sil_score = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    
    metrics = {
        'silhouette_score': sil_score,
        'davies_bouldin_score': db_score
    }
    
    return metrics


def reduce_dimensions_pca(X, n_components=2):
    """
    reduce dimensions using PCA for visualization
    """
    # apply PCA for 2D visualization
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    
    return X_pca, pca


def get_cluster_profiles(ward_df, labels, feature_cols):
    """
    compute cluster-wise feature averages
    """
    # add cluster labels to dataframe
    ward_df_copy = ward_df.copy()
    ward_df_copy['cluster'] = labels
    
    # compute mean features per cluster
    cluster_profiles = ward_df_copy.groupby('cluster')[feature_cols].mean()
    
    return cluster_profiles


def save_model(kmeans, scaler, n_clusters, output_dir='models'):
    """
    save trained k-means model and scaler to disk
    
    args:
        kmeans: trained kmeans model
        scaler: fitted scaler object
        n_clusters: number of clusters
        output_dir: directory to save models
    
    returns:
        paths to saved model and scaler files
    """
    # create models directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # create simple filenames with k value only
    model_filename = f'model_k{n_clusters}.pkl'
    scaler_filename = f'scaler_k{n_clusters}.pkl'
    
    # save model and scaler
    model_path = os.path.join(output_dir, model_filename)
    scaler_path = os.path.join(output_dir, scaler_filename)
    
    joblib.dump(kmeans, model_path)
    joblib.dump(scaler, scaler_path)
    
    return model_path, scaler_path


def load_model(model_path, scaler_path):
    """
    load saved k-means model and scaler from disk
    
    args:
        model_path: path to saved model file
        scaler_path: path to saved scaler file
    
    returns:
        loaded kmeans model and scaler, or (None, None) if files don't exist
    """
    # check if files exist
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        kmeans = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return kmeans, scaler
    else:
        return None, None

