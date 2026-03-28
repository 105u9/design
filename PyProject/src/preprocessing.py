# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return df

def clean_and_impute(df, method='knn'):
    # Drop columns with too many missing values (e.g., > 50%)
    threshold = 0.5 * len(df)
    df = df.dropna(axis=1, thresh=threshold)
    
    # Separate numeric columns for imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df

def normalize_data(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df_normalized = df.copy()
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_normalized, scaler

def calculate_pcc(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr(method='pearson')
    return corr_matrix

def select_features(corr_matrix, threshold=0.5):
    # Requirement: Extract significant correlation (|PCC| > 0.5)
    features = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                features.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    return features

def load_building_data(building_id, site_id, meter_type='chilledwater'):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load Weather Data
    weather_path = os.path.join(base_path, "building-data-genome-project-2", "data", "weather", "weather.csv")
    df_weather = pd.read_csv(weather_path)
    df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
    df_weather = df_weather[df_weather['site_id'] == site_id]
    
    # Load Meter Data
    meter_path = os.path.join(base_path, "building-data-genome-project-2", "data", "meters", "cleaned", f"{meter_type}_cleaned.csv")
    df_meter = pd.read_csv(meter_path)
    df_meter['timestamp'] = pd.to_datetime(df_meter['timestamp'])
    
    # Merge on timestamp
    df_merged = pd.merge(df_weather, df_meter[['timestamp', building_id]], on='timestamp')
    df_merged = df_merged.rename(columns={building_id: 'power_usage'})
    
    return df_merged

if __name__ == "__main__":
    import os
    # Example usage: Panther_office_Karla
    print("Loading building data for Panther_office_Karla...")
    df_building = load_building_data("Panther_office_Karla", "Panther")
    
    # Clean and impute
    print("Cleaning and imputing data...")
    df_cleaned = clean_and_impute(df_building, method='mean')
    
    # Normalize
    print("Normalizing data (Z-Score)...")
    df_normalized, scaler = normalize_data(df_cleaned)
    
    # PCC Analysis
    print("Calculating Pearson Correlation Coefficients...")
    corr_matrix = calculate_pcc(df_normalized)
    
    # Feature Selection
    print("Selecting highly correlated features for power_usage...")
    if 'power_usage' in corr_matrix.columns:
        pcc_with_power = corr_matrix['power_usage'].sort_values(ascending=False)
        print("\nCorrelation with Power Usage:")
        print(pcc_with_power)

    # Save cleaned data
    df_normalized.to_csv('src/cleaned_building_data.csv', index=False)
    print("\nPhase 2 processing complete. Cleaned data saved to 'src/cleaned_building_data.csv'.")
