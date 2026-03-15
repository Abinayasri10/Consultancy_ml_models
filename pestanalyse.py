import pandas as pd
import os
import joblib  # Used to save the trained model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

# --- 1. SET UP THE PATHS ---
# We are in 'ml_models', so we go UP (..) then DOWN into 'frontend/src/data'
csv_path = os.path.join("..", "frontend", "src", "data", "questionsv4.csv")
model_save_path = "pest_model.pkl"

print(f"🔍 Looking for dataset at: {csv_path}")

try:
    # --- 2. LOAD DATA ---
    df = pd.read_csv(csv_path)

    # Clean up: Rename columns to match what the ML model expects
    # Your CSV has 'questions' and 'answers'
    df = df.rename(columns={'questions': 'text', 'answers': 'label'})
    
    # Remove any empty rows just in case
    df = df.dropna()

    print(f"✅ Loaded {len(df)} rows of data.")

    # --- 3. TRAIN THE MODEL ---
    print("⚙️  Training the AI model... (This might take a second)")
    
    # We use KNN (K-Nearest Neighbors) to find the most similar question
    model = make_pipeline(TfidfVectorizer(), KNeighborsClassifier(n_neighbors=1))
    model.fit(df['text'], df['label'])

    print("✅ Training complete!")

    # --- 4. TEST IT ---
    # Let's test it immediately to make sure it works
    test_input = "insect looks like yellow beetle"
    prediction = model.predict([test_input])

    print("\n--- 🧪 TEST RESULTS ---")
    print(f"User Problem: '{test_input}'")
    print(f"AI Suggests : {prediction[0]}")
    print("-----------------------")

    # --- 5. SAVE THE MODEL ---
    joblib.dump(model, model_save_path)
    print(f"💾 Model saved successfully as '{model_save_path}'")

except FileNotFoundError:
    print("\n❌ ERROR: Could not find 'questionsv4.csv'.")
    print("Make sure your terminal is inside the 'ml_models' folder when you run this!")