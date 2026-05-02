# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return df

def clean_and_impute(df, method='knn'):
    # Ensure we work on a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # --- NEW: Extract Time Features for Graduation Requirement ---
    # Help LSTM identify occupancy and daily patterns
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        # Use sin/cos for cyclical features to represent time better
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # Drop columns with too many missing values (e.g., > 50%)
    threshold = 0.5 * len(df)
    df = df.dropna(axis=1, thresh=threshold)
    
    # Separate numeric columns for imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'mean':
        # Use fillna on numeric columns and update via .loc
        df.loc[:, numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        # Use imputer and update via .loc
        df.loc[:, numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
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

def generate_pcc_adj(df, threshold=0.5):
    """
    Generate an adjacency matrix based on Pearson Correlation Coefficients.
    Only connections with |PCC| > threshold are kept.
    """
    corr_matrix = calculate_pcc(df)
    adj = (corr_matrix.abs() > threshold).astype(float).values
    # Set diagonal to 1 (self-loops)
    np.fill_diagonal(adj, 1.0)
    return adj, corr_matrix.columns.tolist()

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
    
    # Load Weather Data (All Features)
    weather_path = os.path.join(base_path, "building-data-genome-project-2", "data", "weather", "weather.csv")
    df_weather = pd.read_csv(weather_path, engine='python', on_bad_lines='skip')
    df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
    df_weather = df_weather[df_weather['site_id'] == site_id]
    
    # Load Primary Meter Data (e.g., Chilled Water for Cooling Load)
    meter_path = os.path.join(base_path, "building-data-genome-project-2", "data", "meters", "cleaned", f"{meter_type}_cleaned.csv")
    df_meter = pd.read_csv(meter_path, engine='python', on_bad_lines='skip')
    df_meter['timestamp'] = pd.to_datetime(df_meter['timestamp'])
    
    # Load Secondary Meter Data (e.g., Electricity for Internal Heat Gain proxy)
    # This addresses "using all appropriate data types"
    elec_path = os.path.join(base_path, "building-data-genome-project-2", "data", "meters", "cleaned", "electricity_cleaned.csv")
    df_elec = pd.read_csv(elec_path, engine='python', on_bad_lines='skip')
    df_elec['timestamp'] = pd.to_datetime(df_elec['timestamp'])
    
    # Merge all datasets
    df_merged = pd.merge(df_weather, df_meter[['timestamp', building_id]], on='timestamp')
    df_merged = df_merged.rename(columns={building_id: 'power_usage'})
    
    if building_id in df_elec.columns:
        df_merged = pd.merge(df_merged, df_elec[['timestamp', building_id]], on='timestamp')
        df_merged = df_merged.rename(columns={building_id: 'total_electricity'})

    # --- Feature Engineering: Enhanced Physical Indicators ---
    # 1. Thermal Lag: 3-hour and 6-hour rolling average of outdoor temperature
    # Buildings have thermal mass and do not respond instantly to outdoor changes.
    df_merged['air_temp_rolling_3h'] = df_merged['airTemperature'].rolling(window=3, min_periods=1).mean()
    df_merged['air_temp_rolling_6h'] = df_merged['airTemperature'].rolling(window=6, min_periods=1).mean()
    
    # 2. Solar Impact Proxy: Cloud coverage combined with hour of day
    if 'cloudCoverage' in df_merged.columns:
        # Simple proxy: during day (8-18), high cloud coverage reduces solar gain
        is_day = (df_merged['timestamp'].dt.hour >= 8) & (df_merged['timestamp'].dt.hour <= 18)
        df_merged['solar_proxy'] = np.where(is_day, 10.0 - df_merged['cloudCoverage'].fillna(5.0), 0.0)

    # 3. Cyclical Wind Direction
    if 'windDirection' in df_merged.columns:
        df_merged['wind_dir_sin'] = np.sin(2 * np.pi * df_merged['windDirection'] / 360.0)
        df_merged['wind_dir_cos'] = np.cos(2 * np.pi * df_merged['windDirection'] / 360.0)

    # --- Simulation of Indoor Environment (Enhanced with occupancy proxy) ---
    np.random.seed(42)
    # Indoor Temp: slightly lag and dampen outdoor temp
    df_merged['indoor_temp'] = df_merged['airTemperature'].rolling(window=3, min_periods=1).mean() + np.random.normal(0, 0.5, len(df_merged))
    # Indoor Humidity: derived from dew point and air temp
    df_merged['indoor_humidity'] = 100 * (np.exp((17.625 * df_merged['dewTemperature']) / (243.04 + df_merged['dewTemperature'])) / 
                                        np.exp((17.625 * df_merged['airTemperature']) / (243.04 + df_merged['airTemperature'])))
    # Indoor CO2: occupancy-driven simulation
    # Using total_electricity as a better proxy for occupancy than just cooling power
    occ_proxy = df_merged['total_electricity'] if 'total_electricity' in df_merged.columns else df_merged['power_usage']
    df_merged['indoor_co2'] = 400 + (occ_proxy / occ_proxy.max()) * 600 + np.random.normal(0, 20, len(df_merged))
    
    # --- PHASE 6: Introduce outdoor_temp explicitly ---
    if 'airTemperature' in df_merged.columns:
        df_merged = df_merged.rename(columns={'airTemperature': 'outdoor_temp'})
        
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
