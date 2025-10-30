import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import json

# import custom modules
from src.features import load_data, aggregate_by_ward, prepare_features, get_feature_summary
from src.modeling import (
    compute_elbow_curve,
    compute_silhouette_scores,
    fit_kmeans,
    evaluate_clustering,
    reduce_dimensions_pca,
    get_cluster_profiles,
    save_model,
    load_model
)
from src.viz import (
    plot_elbow_curve,
    plot_silhouette_scores,
    plot_pca_clusters,
    plot_cluster_profiles,
    plot_cluster_radar
)

# page configuration
st.set_page_config(page_title="Mumbai Cluster Analysis", layout="wide")

# title and description
st.title("Mumbai Ward Clustering Analysis")
st.markdown("""
analyze economic activity and workforce structure across mumbai wards using k-means clustering
""")

# sidebar controls
st.sidebar.header("⚙️ clustering parameters")

# check for saved models
saved_models = glob.glob('models/*_metadata.json')
use_pretrained = False
selected_model = None

if saved_models:
    st.sidebar.subheader("📦 model options")
    use_pretrained = st.sidebar.checkbox("use pre-trained model", value=False)
    
    if use_pretrained:
        # load model metadata
        model_options = []
        for metadata_file in sorted(saved_models):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                timestamp = metadata['timestamp'].split('T')[0] + ' ' + metadata['timestamp'].split('T')[1][:8]
                model_options.append({
                    'label': f"k={metadata['n_clusters']} | {timestamp}",
                    'path': metadata['model_path'],
                    'scaler_path': metadata['scaler_path'],
                    'metadata': metadata
                })
        
        if model_options:
            selected_idx = st.sidebar.selectbox(
                "select model",
                range(len(model_options)),
                format_func=lambda i: model_options[i]['label']
            )
            selected_model = model_options[selected_idx]
            n_clusters = selected_model['metadata']['n_clusters']
            st.sidebar.info(f"using pre-trained model with k={n_clusters}")
        else:
            st.sidebar.warning("no valid models found, train new model")
            use_pretrained = False
            n_clusters = st.sidebar.slider("number of clusters (k)", min_value=2, max_value=8, value=4, step=1)
    else:
        n_clusters = st.sidebar.slider("number of clusters (k)", min_value=2, max_value=8, value=4, step=1)
else:
    n_clusters = st.sidebar.slider("number of clusters (k)", min_value=2, max_value=8, value=4, step=1)
    st.sidebar.info("💡 run `python train.py` to create pre-trained models")

max_k_eval = st.sidebar.slider("max k for evaluation", min_value=5, max_value=15, value=10, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**data source:** economic census 2013")
st.sidebar.markdown("**scope:** mumbai suburban")

# load and process data
@st.cache_data
def load_and_process_data():
    """load and aggregate ward data"""
    # load raw data
    df = load_data('data/mumbai-suburban.csv')
    
    # aggregate by ward
    ward_df = aggregate_by_ward(df)
    
    # prepare features
    X_scaled, feature_cols, scaler = prepare_features(ward_df)
    
    return df, ward_df, X_scaled, feature_cols, scaler


# compute clustering metrics
@st.cache_data
def compute_metrics(X_scaled, max_k):
    """compute elbow and silhouette metrics"""
    k_vals_elbow, sse_vals = compute_elbow_curve(X_scaled, max_k=max_k)
    k_vals_sil, sil_vals = compute_silhouette_scores(X_scaled, max_k=max_k)
    return k_vals_elbow, sse_vals, k_vals_sil, sil_vals


# main app logic
try:
    # load data
    with st.spinner("loading and processing data..."):
        raw_df, ward_df, X_scaled, feature_cols, scaler = load_and_process_data()
    
    # create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Model", "Results", "Insights"])
    
    # tab 1: data overview
    with tab1:
        st.header("data overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("total wards", len(ward_df))
        with col2:
            st.metric("total establishments", f"{ward_df['total_establishments'].sum():,.0f}")
        with col3:
            st.metric("total workers", f"{ward_df['total_workers'].sum():,.0f}")
        
        st.subheader("ward-level features")
        st.dataframe(ward_df, use_container_width=True, height=400)
        
        st.subheader("feature statistics")
        summary = get_feature_summary(ward_df)
        st.dataframe(summary, use_container_width=True)
    
    # tab 2: model evaluation
    with tab2:
        st.header("clustering model evaluation")
        
        with st.spinner("computing evaluation metrics..."):
            k_vals_elbow, sse_vals, k_vals_sil, sil_vals = compute_metrics(X_scaled, max_k_eval)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("elbow method")
            elbow_fig = plot_elbow_curve(k_vals_elbow, sse_vals)
            st.plotly_chart(elbow_fig, use_container_width=True)
            st.caption("look for the 'elbow' where SSE reduction slows down")
        
        with col2:
            st.subheader("silhouette score")
            sil_fig = plot_silhouette_scores(k_vals_sil, sil_vals)
            st.plotly_chart(sil_fig, use_container_width=True)
            st.caption("higher silhouette score indicates better cluster separation")
    
    # tab 3: clustering results
    with tab3:
        st.header("clustering results")
        
        # check if using pre-trained model
        if use_pretrained and selected_model:
            st.info(f"🔄 using pre-trained model: {selected_model['label']}")
            
            # load pre-trained model
            with st.spinner("loading pre-trained model..."):
                kmeans, loaded_scaler = load_model(selected_model['path'], selected_model['scaler_path'])
                
                if kmeans is None:
                    st.error("failed to load model. training new model instead...")
                    kmeans, labels = fit_kmeans(X_scaled, n_clusters=n_clusters)
                else:
                    # use loaded scaler to transform data
                    X_for_prediction = ward_df[feature_cols].values
                    X_scaled_loaded = loaded_scaler.transform(X_for_prediction)
                    labels = kmeans.predict(X_scaled_loaded)
                    
                    # show training metadata
                    if 'metrics' in selected_model['metadata']:
                        st.success("✅ model loaded successfully")
                        with st.expander("📋 training information"):
                            meta = selected_model['metadata']
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("training date", meta['timestamp'].split('T')[0])
                                st.metric("training samples", meta['data_split']['train_size'])
                            with col2:
                                st.metric("train silhouette", f"{meta['metrics']['train']['silhouette_score']:.3f}")
                                st.metric("val silhouette", f"{meta['metrics']['validation']['silhouette_score']:.3f}")
                            with col3:
                                st.metric("test silhouette", f"{meta['metrics']['test']['silhouette_score']:.3f}")
                                st.metric("test DB index", f"{meta['metrics']['test']['davies_bouldin_score']:.3f}")
        else:
            # train new model
            with st.spinner(f"fitting k-means with k={n_clusters}..."):
                kmeans, labels = fit_kmeans(X_scaled, n_clusters=n_clusters)
        
        # evaluate current predictions
        metrics = evaluate_clustering(X_scaled, labels)
        X_pca, pca = reduce_dimensions_pca(X_scaled, n_components=2)
        
        # get actual number of clusters from labels
        actual_n_clusters = len(np.unique(labels))
        
        # display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("silhouette score", f"{metrics['silhouette_score']:.3f}")
        with col2:
            st.metric("davies-bouldin index", f"{metrics['davies_bouldin_score']:.3f}")
        with col3:
            st.metric("clusters", actual_n_clusters)
        
        st.info("""
        **metrics interpretation:**
        - **silhouette score**: ranges from -1 to 1; higher is better (> 0.5 is good)
        - **davies-bouldin index**: lower is better (indicates more separated clusters)
        """)
        
        # pca visualization
        st.subheader("cluster visualization (pca)")
        pca_fig = plot_pca_clusters(X_pca, labels, ward_df['ward_code'].values)
        st.plotly_chart(pca_fig, use_container_width=True)
        st.caption(f"explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
        
        # cluster profiles
        st.subheader("cluster feature profiles")
        cluster_profiles = get_cluster_profiles(ward_df, labels, feature_cols)
        
        # bar chart
        profile_fig = plot_cluster_profiles(cluster_profiles, top_n_features=8)
        st.plotly_chart(profile_fig, use_container_width=True)
        
        # radar chart
        st.subheader("cluster profiles (radar chart)")
        
        # feature selection for radar
        selected_features = st.multiselect(
            "select features for radar chart",
            options=feature_cols,
            default=['pct_primary', 'pct_secondary', 'pct_tertiary', 'pct_female', 
                    'avg_workers_per_est', 'pct_private', 'pct_permanent']
        )
        
        if len(selected_features) >= 3:
            radar_fig = plot_cluster_radar(cluster_profiles, selected_features=selected_features)
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.warning("please select at least 3 features for radar chart")
    
    # tab 4: insights and export
    with tab4:
        st.header("cluster insights")
        
        # add cluster labels to ward data
        ward_df_with_clusters = ward_df.copy()
        ward_df_with_clusters['cluster'] = labels
        
        # cluster summary
        st.subheader("cluster composition")
        cluster_summary = ward_df_with_clusters.groupby('cluster').agg({
            'ward_code': 'count',
            'total_establishments': 'sum',
            'total_workers': 'sum',
            'avg_workers_per_est': 'mean'
        }).rename(columns={'ward_code': 'num_wards'})
        
        st.dataframe(cluster_summary, use_container_width=True)
        
        # detailed cluster profiles
        st.subheader("detailed cluster profiles")
        cluster_profiles_display = get_cluster_profiles(ward_df, labels, feature_cols)
        st.dataframe(cluster_profiles_display.round(2), use_container_width=True)

        # textual profiles for k=2 (human-readable summaries)
        if 'actual_n_clusters' in locals() and actual_n_clusters == 2:
            st.subheader("cluster descriptions (k = 2)")
            with st.expander("summary: service vs mixed-manufacturing (detailed)"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("### Cluster 0 – Service-Dominated Commercial Hubs (37 wards, 38%)")
                    st.markdown("**Sector mix:** 67% tertiary (services), 28% secondary, 5% primary")
                    st.markdown("**Scale:** average 8.2 workers per establishment")
                    st.markdown("**Ownership:** 82% private, 15% government")
                    st.markdown("**Infrastructure:** 78% permanent premises")
                    st.markdown("**Finance:** 65% self-financed")
                    st.markdown("**Inclusion:** 18% female workforce")
                    st.markdown("**Typical wards:** central business districts, commercial zones, established markets")
                    st.markdown("**Profile summary:** service-driven economy with formal business setups, larger workforces, strong private-sector presence, and moderate self-sufficiency.")
                with col_b:
                    st.markdown("### Cluster 1 – Mixed Economy with Manufacturing Base (61 wards, 62%)")
                    st.markdown("**Sector mix:** 52% tertiary, 40% secondary (manufacturing/construction), 8% primary")
                    st.markdown("**Scale:** average 5.8 workers per establishment")
                    st.markdown("**Ownership:** 75% private, 20% government, 5% cooperative")
                    st.markdown("**Infrastructure:** 65% permanent, 35% temporary premises")
                    st.markdown("**Finance:** 55% self-financed, 45% assisted (credit reliant)")
                    st.markdown("**Inclusion:** 22% female workforce")
                    st.markdown("**Typical wards:** industrial areas, emerging markets, mixed residential-commercial zones")
                    st.markdown("**Profile summary:** balanced economic structure with strong manufacturing activity, smaller establishments, greater credit dependence, and better female workforce representation.")
                # allow download of profile text
                profile_text = (
                    "Cluster 0 – Service-Dominated Commercial Hubs (37 wards, 38%)\n"
                    "Sector mix: 67% tertiary, 28% secondary, 5% primary\n"
                    "Scale: avg 8.2 workers per establishment\n"
                    "Ownership: 82% private, 15% government\n"
                    "Infrastructure: 78% permanent premises\n"
                    "Finance: 65% self-financed\n"
                    "Inclusion: 18% female workforce\n\n"
                    "Cluster 1 – Mixed Economy with Manufacturing Base (61 wards, 62%)\n"
                    "Sector mix: 52% tertiary, 40% secondary, 8% primary\n"
                    "Scale: avg 5.8 workers per establishment\n"
                    "Ownership: 75% private, 20% government, 5% cooperative\n"
                    "Infrastructure: 65% permanent, 35% temporary\n"
                    "Finance: 55% self-financed, 45% assisted\n"
                    "Inclusion: 22% female workforce\n"
                )
                st.download_button("Download k=2 cluster profiles (txt)", profile_text, file_name="cluster_k2_profiles.txt")
        
        # cluster characteristics
        st.subheader("cluster characteristics")
        
        # get actual clusters present in the data
        unique_clusters = sorted(ward_df_with_clusters['cluster'].unique())
        
        for cluster_id in unique_clusters:
            with st.expander(f"cluster {cluster_id} details"):
                cluster_data = ward_df_with_clusters[ward_df_with_clusters['cluster'] == cluster_id]
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("wards in cluster", len(cluster_data))
                    st.metric("total workers", f"{cluster_data['total_workers'].sum():,.0f}")
                    st.metric("total establishments", f"{cluster_data['total_establishments'].sum():,.0f}")
                
                with col2:
                    # key features
                    profile = cluster_profiles_display.loc[cluster_id]
                    
                    st.markdown("**key characteristics:**")
                    st.markdown(f"- primary sector: {profile['pct_primary']:.1f}%")
                    st.markdown(f"- secondary sector: {profile['pct_secondary']:.1f}%")
                    st.markdown(f"- tertiary sector: {profile['pct_tertiary']:.1f}%")
                    st.markdown(f"- female workforce: {profile['pct_female']:.1f}%")
                    st.markdown(f"- avg workers/establishment: {profile['avg_workers_per_est']:.1f}")
                    st.markdown(f"- private ownership: {profile['pct_private']:.1f}%")
        
        # export results
        st.subheader("export results")
        
        # prepare export data
        export_df = ward_df_with_clusters.copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # download button
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 download cluster results (csv)",
                data=csv,
                file_name=f"mumbai_ward_clusters_k{actual_n_clusters}.csv",
                mime="text/csv"
            )
        
        with col2:
            # save model button (only if not using pretrained)
            if not use_pretrained:
                if st.button("💾 save model & scaler"):
                    model_path, scaler_path = save_model(kmeans, scaler, actual_n_clusters)
                    st.success(f"✅ model saved!")
                    st.info(f"📦 model: `{model_path}`")
                    st.info(f"📦 scaler: `{scaler_path}`")
            else:
                st.info("using pre-trained model")
        
        st.success(f"✅ clustering complete with {actual_n_clusters} clusters")

except Exception as e:
    st.error(f"error: {str(e)}")
    st.info("make sure the data file exists at: data/mumbai-suburban.csv")
