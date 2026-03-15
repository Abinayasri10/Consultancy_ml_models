"""
Demand Forecasting Model Training Script
Uses ARIMA + Exponential Smoothing ensemble for 6-month demand prediction
Updated to include full 34-product pesticide catalog.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pickle
import json
import os

# Ensure the directory for models exists
os.makedirs('ml_models', exist_ok=True)

# Historical Sales Data (Monthly - 24 months) - Total Platform Sales
HISTORICAL_DATA = {
    'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
              'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'sales': [2200, 2400, 2100, 2800, 2600, 3000,
              2150, 2380, 2050, 2750, 2650, 3100,
              2300, 2450, 2200, 2950, 2750, 3200,
              2250, 2400, 2150, 2900, 2700, 3150]
}

# Full Pesticide List from Frontend
PESTICIDE_CATALOG = [
    'ATRATAE', 'COVER', 'BIOMYCIN', 'CELQUIN', 'CyPEPGAURD', 'SULPIX 807.wP', 
    'ALL CLEAR', 'CLINTON', 'ROUND-UP', 'BEVRATON', 'MONDKEM', 'SPLC-DAP', 
    'KATHIR SUPER PHASPHTL', 'STANES 18 ND', 'STANES 10 ND', 'AGRIYA PLUS', 
    'MICRONAL LOM RASLL', 'ZING SULPHATE', 'POWER TRON', 'HHJACK 200 LITA', 
    'JANBAA2', 'KEMTREK', 'INDOELL', 'AVTHAR', 'TAREEP', 'WOKOVIT', 
    'HITACK', 'EXYGOLD', 'TATA-METRIC', 'D-CELL', 'WEEDLESS SOPER', 
    'MP DUST', 'SUN POWER', 'SMITE'
]

def train_arima_model():
    """Train ARIMA model for seasonal forecasting"""
    print("[v0] Training ARIMA model...")
    
    df = pd.DataFrame(HISTORICAL_DATA)
    sales_series = df['sales'].values
    
    # Fit ARIMA model: (p,d,q) x (P,D,Q,s)
    model = ARIMA(sales_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    arima_result = model.fit()
    
    # Forecast 6 months
    forecast = arima_result.get_forecast(steps=6)
    forecast_values = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()
    
    if hasattr(confidence_intervals, 'iloc'):
        lower_ci = confidence_intervals.iloc[:, 0].tolist()
        upper_ci = confidence_intervals.iloc[:, 1].tolist()
    else:
        lower_ci = confidence_intervals[:, 0].tolist()
        upper_ci = confidence_intervals[:, 1].tolist()
    
    forecast_data = {
        'months': ['Feb (F)', 'Mar (F)', 'Apr (F)', 'May (F)', 'Jun (F)', 'Jul (F)'],
        'forecast': forecast_values.tolist(),
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
    }
    
    with open('ml_models/arima_forecast.pkl', 'wb') as f:
        pickle.dump(arima_result, f)
    
    return forecast_data

def generate_ensemble_forecast():
    """Combine ARIMA and simple exponential smoothing"""
    print("\n[v0] Generating Ensemble Forecast...")
    
    with open('ml_models/arima_forecast.pkl', 'rb') as f:
        arima_model = pickle.load(f)
    
    arima_forecast = arima_model.get_forecast(steps=6).predicted_mean
    
    # Simple exponential smoothing
    sales = np.array(HISTORICAL_DATA['sales'])
    alpha = 0.3
    es_forecast = []
    s = sales[-1]
    for i in range(6):
        s = alpha * sales[-(i+1)] + (1 - alpha) * s
        es_forecast.append(s)
    
    es_forecast = np.array(es_forecast)
    ensemble_forecast = (arima_forecast + es_forecast) / 2
    
    # Accuracy Metrics
    mape = np.mean(np.abs((arima_forecast - sales[-6:]) / sales[-6:])) * 100
    
    forecast_result = {
        'forecast': ensemble_forecast.tolist(),
        'mape': round(float(mape), 2),
        'confidence': f"{max(0, 100 - mape):.1f}%",
        'algorithm': 'ARIMA + Exponential Smoothing Ensemble'
    }
    
    with open('ml_models/demand_forecast.json', 'w') as f:
        json.dump(forecast_result, f, indent=2)
    
    return forecast_result

def generate_full_catalog_forecast():
    """Distribute total demand across all 34 products"""
    print(f"\n[v0] Generating product-level predictions for {len(PESTICIDE_CATALOG)} items...")
    
    with open('ml_models/demand_forecast.json', 'r') as f:
        ensemble = json.load(f)
    
    # We distribute the 'Total' demand equally across all products for this demonstration
    # In production, you would use historical weights (e.g., 'ROUND-UP' is 10% of sales)
    share_per_product = 1.0 / len(PESTICIDE_CATALOG)
    
    full_catalog_data = {
        "metadata": ensemble,
        "products": {}
    }
    
    for product in PESTICIDE_CATALOG:
        product_prediction = [round(f * share_per_product, 2) for f in ensemble['forecast']]
        full_catalog_data["products"][product] = product_prediction

    # Save final JSON for React frontend
    with open('ml_models/full_catalog_forecast.json', 'w') as f:
        json.dump(full_catalog_data, f, indent=2)
    
    print(f"[v0] Success! Full catalog forecast saved to ml_models/full_catalog_forecast.json")

if __name__ == '__main__':
    print("[v0] === STARTING FULL CATALOG DEMAND TRAINING ===\n")
    train_arima_model()
    generate_ensemble_forecast()
    generate_full_catalog_forecast()
    print("\n[v0] === TRAINING COMPLETE ===")