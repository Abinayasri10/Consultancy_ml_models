from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import os
import joblib

app = Flask(__name__)
CORS(app) # Allows React to communicate with this API

# 1. Load the Trained Model
print("Loading model... (this takes a few seconds)")
MODEL_PATH = 'rice_vision_model.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load Dosage ML Model
try:
    dosage_data = joblib.load('dosage_model.pkl')
    dosage_model = dosage_data['model']
    le_crop = dosage_data['le_crop']
    le_disease = dosage_data['le_disease']
    le_age = dosage_data['le_age']
    le_out = dosage_data['le_out']
    print("✅ ML Dosage model loaded successfully")
except Exception as e:
    print(f"❌ Error loading ML Dosage model: {e}")
    dosage_model = None

# Load Validation Model (MobileNetV2 ImageNet) for crop detection
try:
    print("Loading ImageNet Validator... (this may take a moment)")
    validator_model = tf.keras.applications.MobileNetV2(weights='imagenet')
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess, decode_predictions
    print("✅ ImageNet Validator loaded successfully!")
except Exception as e:
    print(f"❌ Error loading ImageNet Validator: {e}")
    validator_model = None

# The exact class names your model just learned
CLASS_NAMES = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

# 2. Logic Layer: Map Diseases to Pesticides/Treatments
TREATMENTS = {
    'Bacterialblight': "Apply Copper-based fungicides (e.g., Copper Hydroxide). Avoid excessive nitrogen fertilizer.",
    'Blast': "Apply Tricyclazole or Propiconazole. Manage water levels carefully.",
    'Brownspot': "Improve soil fertility (add Nitrogen/Potassium) and apply Mancozeb or Edifenphos.",
    'Tungro': "Control green leafhoppers using Imidacloprid or Thiamethoxam. Destroy infected stubble."
}

disease_translations = {
    'Bacterialblight': ('Bacterial Blight', 'பாக்டீரியா இலைக்கருகல் நோய்', 'Bacterial leaf blight causes yellowing and drying of leaves.', 'இலைகள் மஞ்சள் நிறமாக மாறி காய்ந்துவிடும்.'),
    'Blast': ('Blast Disease', 'குலை நோய் (பிளாஸ்ட்)', 'Causes spindle-shaped spots on leaves, leading to severe damage.', 'இலைகளில் கண் வடிவ புள்ளிகள் தோன்றி பெரும் சேதத்தை ஏற்படுத்தும்.'),
    'Brownspot': ('Brown Spot', 'பழுப்பு புள்ளி நோய்', 'Brown or reddish-brown spots appear on the leaves.', 'இலைகளில் பழுப்பு அல்லது சிவப்பு நிற புள்ளிகள் தோன்றும்.'),
    'Tungro': ('Tungro Disease', 'துங்ரோ நோய்', 'Viral disease causing stunted growth and yellow-orange leaves.', 'பயிர் வளர்ச்சி குன்றி, இலைகள் மஞ்சள்-ஆரஞ்சு நிறமாக மாறும்.')
}

def is_crop_image(img):
    try:
        # 1. Color Heuristic Check
        rgb = np.array(img.convert('RGB'))
        r, g, b = rgb[:,:,0].astype(int), rgb[:,:,1].astype(int), rgb[:,:,2].astype(int)
        total_pixels = rgb.shape[0] * rgb.shape[1]
        
        green_pixels = np.sum((g > r + 15) & (g > b + 15) & (g > 40))
        green_ratio = green_pixels / total_pixels
        
        brown_yellow_pixels = np.sum((r > b + 20) & (g > b + 10) & (r > 50))
        brown_ratio = brown_yellow_pixels / total_pixels
        
        # If very clearly a leaf texture by color
        if green_ratio > 0.25 or (green_ratio > 0.10 and brown_ratio > 0.20):
            return True, "Passed color heuristic."

        # 2. ImageNet Object Detection Check
        if validator_model:
            img_resized = img.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = tf.expand_dims(img_array, 0)
            img_array = mobilenet_preprocess(img_array)
            
            preds = validator_model.predict(img_array)
            decoded = decode_predictions(preds, top=10)[0]
            
            crop_keywords = [
                'plant', 'leaf', 'flower', 'tree', 'grass', 'fruit', 'vegetable', 
                'crop', 'weed', 'fern', 'moss', 'vine', 'bush', 'shrub', 'forest', 
                'field', 'garden', 'corn', 'wheat', 'rice', 'soybean', 'cotton', 
                'tobacco', 'potato', 'tomato', 'pepper', 'onion', 'garlic', 'carrot', 
                'radish', 'beet', 'turnip', 'cabbage', 'lettuce', 'spinach', 'broccoli', 
                'cauliflower', 'pumpkin', 'squash', 'cucumber', 'melon', 'watermelon', 
                'apple', 'pear', 'peach', 'cherry', 'plum', 'apricot', 'grape', 
                'strawberry', 'raspberry', 'blackberry', 'blueberry', 'cranberry', 
                'lemon', 'lime', 'orange', 'grapefruit', 'banana', 'pineapple', 
                'mango', 'papaya', 'guava', 'kiwi', 'avocado', 'coconut', 'olive', 
                'nut', 'almond', 'pecan', 'walnut', 'cashew', 'peanut', 'seed', 
                'bean', 'pea', 'lentil', 'chickpea', 'mushroom', 'fungus', 'daisy',
                'pot', 'greenhouse', 'hay', 'straw', 'ear', 'cardoon', 'earthstar', 
                'stinkhorn', 'bolete', 'agaric', 'bell pepper', 'zucchini', 'artichoke',
                'pomegranate', 'fig', 'acorn', 'rapeseed', 'macadamia', 'bramble',
                'buckeye', 'hip', 'coral fungus', 'gyromitra'
            ]
            
            plant_score = 0
            extracted_labels = []
            for _, label, prob in decoded:
                label_lower = label.lower().replace('_', ' ')
                extracted_labels.append(f"{label_lower}")
                
                # Check for keywords in label parts
                label_words = set(label_lower.split())
                for kw in crop_keywords:
                    # check if the term completely matches the label or any words in the label
                    if kw == label_lower or kw in label_words:
                        plant_score += prob
                        break
            
            if plant_score > 0.08: # At least 8% confidence it's a plant-related object
                return True, f"Passed ImageNet."
            
            return False, f"Not a crop image. ImageNet detected: {', '.join(extracted_labels[:3])}"
            
        return False, "Image does not appear to be a crop/leaf (lacks green/brown plant colors)."
    except Exception as e:
        print(f"Validation error: {e}")
        # ALWAYS fallback to rejecting the image if validation code throws an error
        return False, f"Validation system check failed internally ({str(e)[:30]})."

def preprocess_image(image_bytes):
    """Resizes the uploaded image to 224x224 to match MobileNetV2 requirements"""
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # Create a batch of 1
    return img, img_array

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if not model:
        return jsonify({'error': 'Model not loaded on server'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    crop = request.form.get('crop', 'Rice')
    plant_age = request.form.get('plantAge', 'Mature (1-3 months)')
    
    try:
        # Read the file bytes
        file_bytes = file.read()
        
        # Process the image to get both PIL object (for validation) and array (for prediction)
        img, img_array = preprocess_image(file_bytes)
        
        # Optional: Validate image content is a crop/plant
        is_valid, val_msg = is_crop_image(img)
        print(f"Image Validation: {val_msg}")
        
        if not is_valid:
            error_msg = f"Diagnosis Error: The uploaded image does not appear to be a crop or plant. ({val_msg.split('Net detected: ')[-1] if 'detected:' in val_msg else val_msg})"
            return jsonify({
                'status': 'invalid',
                'message': error_msg
            }), 200 # Return 200 so frontend parses the structure and shows error
            
        # Make the prediction
        predictions = model.predict(img_array)
        
        # Find the highest percentage match
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100
        disease_name = CLASS_NAMES[predicted_index]
        
        # Get the recommended treatment
        treatment = TREATMENTS.get(disease_name, "Consult a local agricultural expert.")
        name_en, name_ta, desc_en, desc_ta = disease_translations.get(
            disease_name, 
            (disease_name, disease_name, "Analysis completed.", "பகுப்பாய்வு முடிந்தது.")
        )
        
        # Default Pesticide Defaults
        pesticide_en, pesticide_ta = 'Standard Treatment', 'வழக்கமான சிகிச்சை'
        dosage_en, dosage_ta = 'As directed', 'இயல்பான அளவு'
        quantity_en, quantity_ta = 'Consult expert', 'நிபுணரை அணுகவும்'
        
        # Use ML Model to predict specific dosage based on plant age
        if dosage_model:
            try:
                c_code = le_crop.transform([crop])[0]
                d_code = le_disease.transform([disease_name])[0]
                a_code = le_age.transform([plant_age])[0]
                
                pred_class = dosage_model.predict([[c_code, d_code, a_code]])[0]
                pred_str = le_out.inverse_transform([pred_class])[0]
                
                parts = pred_str.split("|")
                if len(parts) == 6:
                    pesticide_en, pesticide_ta, dosage_en, dosage_ta, quantity_en, quantity_ta = parts
            except Exception as e:
                print(f"ML Dosage Prediction Warning: {e}")
        
        return jsonify({
            'disease': disease_name,
            'confidence': f"{confidence:.1f}%",
            'treatment': treatment,
            'name_en': name_en,
            'name_ta': name_ta,
            'description_en': desc_en,
            'description_ta': desc_ta,
            'pesticide_en': pesticide_en,
            'pesticide_ta': pesticide_ta,
            'dosage_en': dosage_en,
            'dosage_ta': dosage_ta,
            'quantity_en': quantity_en,
            'quantity_ta': quantity_ta,
            'scientific_name': f"{disease_name} Pathogen"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask API on port 5001...")
    app.run(port=5001, debug=True)