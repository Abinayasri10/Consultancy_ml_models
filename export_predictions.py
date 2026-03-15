"""
CSV to JavaScript Dataset Converter
Extracts structured agricultural recommendations from questionsv4.csv
for frontend use in pestanalyze.jsx
"""

import pandas as pd
import re
import json
from pathlib import Path

def extract_crop_from_text(text):
    """Extract crop name from question text"""
    crops = ['rice', 'wheat', 'cotton', 'tomato', 'potato', 'maize', 'chili', 'sugarcane', 'coconut', 'brinjal']
    text_lower = text.lower()
    
    for crop in crops:
        if crop in text_lower:
            return crop.capitalize()
    return None

def extract_symptoms_from_text(text):
    """Extract symptoms from question text"""
    symptoms_map = {
        'yellowing': ['yellowing', 'yellow'],
        'spots': ['spots', 'spot', 'spotting'],
        'holes': ['holes', 'hole', 'boring'],
        'wilting': ['wilting', 'wilt'],
        'curling': ['curling', 'curl'],
        'rotting': ['rotting', 'rot', 'rotten'],
        'deformation': ['deformation', 'deformed'],
        'stunted growth': ['stunted', 'weak growth'],
        'lesions': ['lesions', 'lesion'],
        'discoloration': ['discoloration', 'discolored']
    }
    
    text_lower = text.lower()
    found_symptoms = []
    
    for symptom, keywords in symptoms_map.items():
        for keyword in keywords:
            if keyword in text_lower:
                found_symptoms.append(symptom)
                break
    
    return found_symptoms

def extract_pesticide_from_answer(text):
    """Extract pesticide name and dosage from answer text"""
    text_lower = text.lower()
    
    # Common pesticide patterns
    pesticides = {
        'rogor': 'ROGOR',
        'dithane': 'DITHANE M-45',
        'carbendazim': 'ALL CLEAR',
        'mancozeb': 'INDOFIL',
        'malathion': 'SPRINT',
        'tricel': 'TRICEL',
        'streptomycin': 'CROP GUARD',
        'bordeaux mixture': 'COPPER GUARD',
        'captan': 'ALL CLEAR',
        'chlorpyriphos': 'SPRINT'
    }
    
    for key, value in pesticides.items():
        if key in text_lower:
            return value
    
    return 'CROP GUARD'  # Default

def extract_dosage_from_answer(text):
    """Extract dosage information from answer text"""
    # Look for dosage patterns like @2ml/lit, @1gram/liter, etc.
    dosage_pattern = r'@\s*(\d+(?:\.\d+)?)\s*(?:ml|gram|g|kg)\/(?:lit|liter|litre)'
    match = re.search(dosage_pattern, text.lower())
    
    if match:
        return f"{match.group(1)} per litre"
    
    return "As per label directions"

def process_csv_to_dataset():
    """Main function to process CSV and create JavaScript dataset"""
    
    # Read CSV file
    csv_path = "../src/data/questionsv4.csv"
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from CSV")
    except FileNotFoundError:
        print(f"CSV file not found at {csv_path}")
        return
    
    structured_data = []
    
    # Process first 1000 rows for performance
    for index, row in df.head(1000).iterrows():
        question = str(row['questions'])
        answer = str(row['answers'])
        
        # Extract information
        crop = extract_crop_from_text(question)
        symptoms = extract_symptoms_from_text(question)
        pesticide = extract_pesticide_from_answer(answer)
        dosage = extract_dosage_from_answer(answer)
        
        # Only include entries with valid crop and symptoms
        if crop and symptoms:
            # Try to extract pest name from question
            pest_patterns = {
                'aphid': 'Aphid',
                'leaf miner': 'Leaf Miner',
                'bollworm': 'Bollworm',
                'stem borer': 'Stem Borer',
                'blast': 'Leaf Blast',
                'blight': 'Late Blight',
                'wilt': 'Bacterial Wilt',
                'rust': 'Leaf Rust',
                'mildew': 'Powdery Mildew',
                'folder': 'Rice Leaf Folder',
                'hopper': 'Plant Hopper'
            }
            
            pest = 'Unknown Pest'
            question_lower = question.lower()
            for pattern, pest_name in pest_patterns.items():
                if pattern in question_lower:
                    pest = pest_name
                    break
            
            entry = {
                'crop': crop.lower(),
                'symptoms': symptoms,
                'pest': pest,
                'scientificName': 'To be updated',
                'pesticide': pesticide,
                'alternativeName': pesticide,
                'dosage': f"300-500 ml/acre",
                'applicationMethod': 'spray',
                'dosageDetails': dosage,
                'safetyInterval': 14,
                'confidence': 85,
                'originalQuestion': question,
                'originalAnswer': answer
            }
            
            structured_data.append(entry)
    
    print(f"Processed {len(structured_data)} valid entries")
    
    # Save as JSON for inspection
    with open('extracted_dataset.json', 'w') as f:
        json.dump(structured_data, f, indent=2)
    
    print("Dataset saved to extracted_dataset.json")
    
    # Generate JavaScript code
    js_code = generate_javascript_dataset(structured_data)
    
    with open('generated_pestanalyze_data.js', 'w') as f:
        f.write(js_code)
    
    print("JavaScript dataset generated as generated_pestanalyze_data.js")

def generate_javascript_dataset(data):
    """Generate JavaScript code for the dataset"""
    
    js_template = '''// Auto-generated dataset from questionsv4.csv
export const pestDatasetFromCSV = [
'''
    
    for entry in data:
        js_template += f'''  {{
    crop: '{entry["crop"]}',
    symptoms: {json.dumps(entry["symptoms"])},
    pest: '{entry["pest"]}',
    scientificName: '{entry["scientificName"]}',
    pesticide: '{entry["pesticide"]}',
    alternativeName: '{entry["alternativeName"]}',
    dosage: '{entry["dosage"]}',
    applicationMethod: '{entry["applicationMethod"]}',
    dosageDetails: '{entry["dosageDetails"]}',
    safetyInterval: {entry["safetyInterval"]},
    confidence: {entry["confidence"]}
  }},
'''
    
    js_template += '];\n'
    
    return js_template

if __name__ == "__main__":
    process_csv_to_dataset()