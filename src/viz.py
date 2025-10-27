import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def plot_elbow_curve(k_values, sse_values):
    """
    plot elbow curve for k-means clustering
    
    args:
        k_values: list of k values
        sse_values: list of SSE values
    
    returns:
        plotly figure
    """
    # create elbow plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=k_values,
        y=sse_values,
        mode='lines+markers',
        marker=dict(size=10, color='#3498db', line=dict(width=2, color='white')),
        line=dict(width=3, color='#3498db'),
        hovertemplate='<b>k=%{x}</b><br>SSE=%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='Elbow Method for Optimal K', font=dict(size=20, color='#2c3e50')),
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Sum of Squared Errors (SSE)',
        template='plotly_white',
        hovermode='x unified',
        height=450,
        font=dict(size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#ecf0f1'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#ecf0f1')
    )
    
    return fig


def plot_silhouette_scores(k_values, silhouette_scores):
    """
    plot silhouette scores for different k values
    """
    # create silhouette score bar chart with color gradient
    colors = ['#e74c3c' if s < 0.3 else '#f39c12' if s < 0.5 else '#2ecc71' for s in silhouette_scores]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=k_values,
        y=silhouette_scores,
        marker=dict(
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=[f'{score:.3f}' for score in silhouette_scores],
        textposition='outside',
        hovertemplate='<b>k=%{x}</b><br>Silhouette=%{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='Silhouette Score by Number of Clusters', font=dict(size=20, color='#2c3e50')),
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Silhouette Score',
        template='plotly_white',
        height=450,
        font=dict(size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#ecf0f1'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#ecf0f1', range=[0, max(silhouette_scores) * 1.2])
    )
    
    return fig


def plot_pca_clusters(X_pca, labels, ward_codes):
    """
    plot 2D PCA scatter colored by cluster
    """
    # create dataframe for plotting
    plot_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'cluster': [f'Cluster {label}' for label in labels],
        'ward_code': ward_codes
    })
    
    # create scatter plot
    fig = px.scatter(
        plot_df,
        x='PC1',
        y='PC2',
        color='cluster',
        hover_data=['ward_code'],
        title='Ward Clustering Visualization (PCA 2D Projection)',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        template='plotly_white',
        height=600,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color='white')),
        marker_opacity=0.8
    )
    
    fig.update_layout(
        title=dict(font=dict(size=20, color='#2c3e50')),
        font=dict(size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#ecf0f1', zeroline=True),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#ecf0f1', zeroline=True),
        legend=dict(
            title=dict(text='Clusters', font=dict(size=14)),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#bdc3c7',
            borderwidth=1
        )
    )
    
    return fig


def plot_cluster_profiles(cluster_profiles, top_n_features=8):
    """
    plot cluster feature profiles as grouped bar chart
    """
    # select top features by variance across clusters
    feature_variance = cluster_profiles.var(axis=0).sort_values(ascending=False)
    top_features = feature_variance.head(top_n_features).index.tolist()
    
    # filter to top features
    profiles_subset = cluster_profiles[top_features]
    
    # create grouped bar chart with better colors
    fig = go.Figure()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    
    for idx, cluster_id in enumerate(profiles_subset.index):
        fig.add_trace(go.Bar(
            name=f'Cluster {cluster_id}',
            x=top_features,
            y=profiles_subset.loc[cluster_id],
            text=[f'{val:.1f}' for val in profiles_subset.loc[cluster_id]],
            textposition='outside',
            marker=dict(
                color=colors[idx % len(colors)],
                line=dict(color='white', width=1.5)
            ),
            hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=f'Cluster Feature Profiles (Top {top_n_features} Features)', font=dict(size=20, color='#2c3e50')),
        xaxis_title='Features',
        yaxis_title='Mean Value',
        barmode='group',
        template='plotly_white',
        height=550,
        font=dict(size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, tickangle=-45),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#ecf0f1'),
        legend=dict(
            title=dict(text='Clusters', font=dict(size=14)),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#bdc3c7',
            borderwidth=1
        )
    )
    
    return fig


def plot_cluster_radar(cluster_profiles, selected_features=None):
    """
    plot cluster profiles as radar chart
    """
    # use selected features or all
    if selected_features is None:
        selected_features = cluster_profiles.columns.tolist()
    
    # normalize to 0-1 scale for radar chart
    profiles_norm = cluster_profiles[selected_features].copy()
    for col in profiles_norm.columns:
        min_val = profiles_norm[col].min()
        max_val = profiles_norm[col].max()
        if max_val > min_val:
            profiles_norm[col] = (profiles_norm[col] - min_val) / (max_val - min_val)
        else:
            profiles_norm[col] = 0.5
    
    # create radar chart
    fig = go.Figure()
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, cluster_id in enumerate(profiles_norm.index):
        fig.add_trace(go.Scatterpolar(
            r=profiles_norm.loc[cluster_id].tolist() + [profiles_norm.loc[cluster_id].iloc[0]],
            theta=selected_features + [selected_features[0]],
            fill='toself',
            name=f'cluster {cluster_id}',
            line=dict(color=colors[idx % len(colors)])
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title='cluster profiles (normalized)',
        template='plotly_white',
        height=500
    )
    
    return fig
