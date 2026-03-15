import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import json
import os
import pickle

# Paths
BASE_DIR = r"d:\Consultancy"
DATA_PATH = os.path.join(BASE_DIR, "frontend", "src", "data", "crop_yield.csv")
MODEL_DIR = os.path.join(BASE_DIR, "frontend", "ml_models")
EXPORT_JSON = os.path.join(MODEL_DIR, "agri_insights.json")

def train_and_analyze():
    print("Starting Advanced Agricultural Analysis...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # 1. Cleaning & Preprocessing
    df = df.dropna()
    # Normalize strings
    df['Crop'] = df['Crop'].str.strip()
    df['State'] = df['State'].str.strip()
    df['Season'] = df['Season'].str.strip()
    
    print(f"Success: Loaded {len(df)} rows.")

    # 2. Key Statistical Insights (For the Dashboard)
    print("Generating Statistical Insights...")
    
    # Top Crops by Average Yield
    crop_stats = df.groupby('Crop').agg({
        'Yield': 'mean',
        'Pesticide': 'mean',
        'Fertilizer': 'mean'
    }).sort_values(by='Yield', ascending=False).head(10).reset_index()
    
    # Regional Efficiency (Yield per unit Pesticide)
    df['Efficiency'] = df['Yield'] / (df['Pesticide'] + 1)
    state_efficiency = df.groupby('State')['Yield'].mean().sort_values(ascending=False).head(10).reset_index()

    # Yearly Trends
    yearly_avg = df.groupby('Crop_Year')['Yield'].mean().reset_index()

    # 3. ML Model Training (Yield Predictor)
    print("Training Yield Prediction Model...")
    
    # Encode Categorical Features
    le_crop = LabelEncoder()
    le_state = LabelEncoder()
    le_season = LabelEncoder()
    
    df_ml = df.copy()
    df_ml['Crop_ID'] = le_crop.fit_transform(df_ml['Crop'])
    df_ml['State_ID'] = le_state.fit_transform(df_ml['State'])
    df_ml['Season_ID'] = le_season.fit_transform(df_ml['Season'])
    
    features = ['Crop_ID', 'State_ID', 'Season_ID', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    X = df_ml[features]
    y = df_ml['Yield']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using small number of estimators for speed in this environment
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Model Accuracy (R2): {score*100:.2f}%")

    # Save Encoders & Model
    model_data = {
        "model": model,
        "encoders": {
            "crop": le_crop,
            "state": le_state,
            "season": le_season
        }
    }
    with open(os.path.join(MODEL_DIR, 'agri_model.pkl'), 'wb') as f:
        pickle.dump(model_data, f)
    
    # 4. EXPORT INSIGHTS FOR FRONTEND
    final_output = {
        "summary": {
            "total_records": len(df),
            "model_accuracy": f"{score*100:.2f}%",
            "top_crop": crop_stats.iloc[0]['Crop'] if not crop_stats.empty else "N/A",
            "high_yield_state": state_efficiency.iloc[0]['State'] if not state_efficiency.empty else "N/A"
        },
        "cropYieldData": crop_stats.to_dict(orient='records'),
        "regionalYield": state_efficiency.to_dict(orient='records'),
        "yearlyTrends": yearly_avg.to_dict(orient='records'),
        "pesticideImpact": df.groupby('Crop')['Pesticide'].mean().sort_values(ascending=False).head(10).reset_index().to_dict(orient='records')
    }

    with open(EXPORT_JSON, 'w') as f:
        json.dump(final_output, f, indent=4)
    
    print(f"Analysis complete. Insights saved to {EXPORT_JSON}")

if __name__ == "__main__":
    train_and_analyze()
