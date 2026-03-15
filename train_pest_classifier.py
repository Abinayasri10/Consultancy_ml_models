"""
Pest Identification Model Training Script
Uses Random Forest/XGBoost for multi-class pest classification
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import json

# Sample Training Data
TRAINING_DATA = {
    'crop': ['Rice', 'Rice', 'Cotton', 'Cotton', 'Wheat', 'Wheat', 'Tomato', 'Tomato'],
    'growth_stage': ['Vegetative', 'Flowering', 'Vegetative', 'Fruiting', 'Vegetative', 'Flowering', 'Fruiting', 'Fruiting'],
    'symptoms': ['Yellowing', 'Spots', 'Holes', 'Rotting', 'Wilting', 'Curling', 'Spots', 'Deformation'],
    'location': ['Bangalore', 'Bangalore', 'Mysore', 'Mysore', 'Belgaum', 'Belgaum', 'Kolar', 'Kolar'],
    'season': ['Kharif', 'Kharif', 'Kharif', 'Kharif', 'Rabi', 'Rabi', 'Zaid', 'Zaid'],
    'temperature': [28, 26, 32, 30, 18, 16, 35, 34],
    'humidity': [85, 80, 75, 78, 65, 60, 70, 72],
    'pest': ['Leaf Miner', 'Brown Spot', 'Bollworm', 'Boll Rot', 'Leaf Rust', 'Powdery Mildew', 'Early Blight', 'Fruit Rot'],
}

def prepare_data(data):
    """Convert raw data to ML-ready format"""
    df = pd.DataFrame(data)
    
    # Encode categorical variables
    label_encoders = {}
    for column in ['crop', 'growth_stage', 'symptoms', 'location', 'season']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    return df, label_encoders

def train_pest_model():
    """Train pest identification model"""
    print("[v0] Preparing training data...")
    df, encoders = prepare_data(TRAINING_DATA)
    
    X = df[['crop', 'growth_stage', 'symptoms', 'location', 'season', 'temperature', 'humidity']]
    y = df['pest']
    
    # Encode target variable
    pest_encoder = LabelEncoder()
    y_encoded = pest_encoder.fit_transform(y)
    
    print("[v0] Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y_encoded)
    
    # Evaluate
    accuracy = model.score(X, y_encoded)
    print(f"[v0] Model Training Complete - Accuracy: {accuracy:.2%}")
    
    # Save model
    model_data = {
        'model': model,
        'encoders': encoders,
        'pest_encoder': pest_encoder,
        'feature_names': X.columns.tolist(),
        'accuracy': float(accuracy)
    }
    
    with open('ml_models/pest_classifier.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("[v0] Model saved to pest_classifier.pkl")
    
    return model, encoders, pest_encoder

def predict_pest(crop, growth_stage, symptoms, location, season, temperature, humidity):
    """Make predictions using trained model"""
    with open('ml_models/pest_classifier.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    encoders = model_data['encoders']
    pest_encoder = model_data['pest_encoder']
    
    # Encode input
    input_data = np.array([[
        encoders['crop'].transform([crop])[0],
        encoders['growth_stage'].transform([growth_stage])[0],
        encoders['symptoms'].transform([symptoms])[0],
        encoders['location'].transform([location])[0],
        encoders['season'].transform([season])[0],
        temperature,
        humidity
    ]])
    
    # Get prediction and confidence
    prediction = model.predict(input_data)[0]
    confidence = max(model.predict_proba(input_data)[0]) * 100
    
    pest_name = pest_encoder.inverse_transform([prediction])[0]
    
    return {
        'pest': pest_name,
        'confidence': confidence,
        'recommendation': get_recommendation(pest_name)
    }

def get_recommendation(pest_name):
    """Get pesticide recommendation based on pest"""
    recommendations = {
        'Leaf Miner': {'product': 'SPRINT', 'dosage': '500-1000 g/acre'},
        'Bollworm': {'product': 'SPRINT', 'dosage': '600-1000 ml/acre'},
        'Brown Spot': {'product': 'ALL CLEAR', 'dosage': '280-400 ml/acre'},
        'Boll Rot': {'product': 'INDOFIL', 'dosage': '320-500 ml/acre'},
        'Leaf Rust': {'product': 'ALL CLEAR', 'dosage': '280-400 ml/acre'},
        'Powdery Mildew': {'product': 'INDOFIL', 'dosage': '320-400 ml/acre'},
        'Early Blight': {'product': 'ALL CLEAR', 'dosage': '300-400 ml/acre'},
        'Fruit Rot': {'product': 'INDOFIL', 'dosage': '400-500 ml/acre'},
    }
    return recommendations.get(pest_name, {'product': 'CROP GUARD', 'dosage': 'Contact support'})

if __name__ == '__main__':
    print("[v0] === PEST IDENTIFICATION MODEL TRAINING ===")
    train_pest_model()
    
    # Test prediction
    print("\n[v0] Testing model prediction...")
    result = predict_pest('Rice', 'Vegetative', 'Yellowing', 'Bangalore', 'Kharif', 28, 85)
    print(f"[v0] Prediction: {result}")
