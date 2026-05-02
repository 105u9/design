
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))
from preprocessing import load_building_data, clean_and_impute

building_id = "Panther_office_Karla"
site_id = "Panther"
print(f"Loading data for {building_id}...")
df = load_building_data(building_id, site_id)
print(f"Data loaded. Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

df_cleaned = clean_and_impute(df, method='mean')
print(f"Cleaned data. Shape: {df_cleaned.shape}")
print(f"Numeric columns: {df_cleaned.select_dtypes(include=['number']).columns.tolist()}")
