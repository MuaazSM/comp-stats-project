"""
feature engineering and preprocessing for ward-level clustering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    # load dataset
    df = pd.read_csv(filepath)
    return df


def aggregate_by_ward(df):
    """
    aggregate establishment and worker data by ward
    """
    # create ward-level aggregates
    ward_data = []
    
    # group by ward code
    for ward_code, group in df.groupby('WC'):
        if ward_code == 0:  # skip unassigned ward
            continue
            
        # total establishments and workers
        total_establishments = group['EB'].nunique() if 'EB' in group.columns else len(group)
        total_workers = group['TOTAL_WORKER'].sum()
        
        # avg workers per establishment
        avg_workers_per_est = total_workers / total_establishments if total_establishments > 0 else 0
        
        # sector distribution (SECTOR: 1=primary, 2=secondary, 3=tertiary)
        sector_counts = group.groupby('SECTOR')['TOTAL_WORKER'].sum()
        total_sector_workers = sector_counts.sum()
        
        pct_primary = (sector_counts.get(1, 0) / total_sector_workers * 100) if total_sector_workers > 0 else 0
        pct_secondary = (sector_counts.get(2, 0) / total_sector_workers * 100) if total_sector_workers > 0 else 0
        pct_tertiary = (sector_counts.get(3, 0) / total_sector_workers * 100) if total_sector_workers > 0 else 0
        
        # ownership distribution (OWN_SHIP_C: 1=private, 2=govt, 3=cooperative)
        ownership_counts = group.groupby('OWN_SHIP_C')['TOTAL_WORKER'].sum()
        pct_govt = (ownership_counts.get(2, 0) / total_workers * 100) if total_workers > 0 else 0
        pct_private = (ownership_counts.get(1, 0) / total_workers * 100) if total_workers > 0 else 0
        pct_cooperative = (ownership_counts.get(3, 0) / total_workers * 100) if total_workers > 0 else 0
        
        # premise type (C_HOUSE: 1=permanent, 2=temporary, 3=mobile)
        premise_counts = group.groupby('C_HOUSE')['TOTAL_WORKER'].sum()
        pct_permanent = (premise_counts.get(1, 0) / total_workers * 100) if total_workers > 0 else 0
        pct_temporary = (premise_counts.get(2, 0) / total_workers * 100) if total_workers > 0 else 0
        
        # finance type (SOF: 1=self, 2=assisted, 9=not reported)
        finance_counts = group.groupby('SOF')['TOTAL_WORKER'].sum()
        finance_total = finance_counts.get(1, 0) + finance_counts.get(2, 0)
        pct_self_financed = (finance_counts.get(1, 0) / finance_total * 100) if finance_total > 0 else 0
        pct_assisted = (finance_counts.get(2, 0) / finance_total * 100) if finance_total > 0 else 0
        
        # female workforce share
        female_workers = group['F_H'].sum() + group['F_NH'].sum()
        male_workers = group['M_H'].sum() + group['M_NH'].sum()
        total_gendered_workers = female_workers + male_workers
        pct_female = (female_workers / total_gendered_workers * 100) if total_gendered_workers > 0 else 0
        
        # compile ward features
        ward_data.append({
            'ward_code': int(ward_code),
            'total_establishments': total_establishments,
            'total_workers': total_workers,
            'avg_workers_per_est': avg_workers_per_est,
            'pct_primary': pct_primary,
            'pct_secondary': pct_secondary,
            'pct_tertiary': pct_tertiary,
            'pct_govt': pct_govt,
            'pct_private': pct_private,
            'pct_cooperative': pct_cooperative,
            'pct_permanent': pct_permanent,
            'pct_temporary': pct_temporary,
            'pct_self_financed': pct_self_financed,
            'pct_assisted': pct_assisted,
            'pct_female': pct_female
        })
    
    # convert to dataframe
    ward_df = pd.DataFrame(ward_data)
    return ward_df


def prepare_features(ward_df):
    # normalization
    # select features for clustering (exclude ward_code)
    feature_cols = [col for col in ward_df.columns if col != 'ward_code']
    X = ward_df[feature_cols].values
    
    # scale features before clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, feature_cols, scaler


def get_feature_summary(ward_df):
    # compute summary stats
    summary = ward_df.describe()
    return summary
