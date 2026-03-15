
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# 1. Extensive Dataset Generation
data = []
ages = {
    "Young (< 1 month)": 0,
    "Mature (1-3 months)": 1,
    "Old (> 3 months)": 2
}

crops_data = {
    'Rice': {
        'diseases': {
            'Bacterialblight': {
                'pest_en': 'Copper Hydroxide', 'pest_ta': 'காப்பர் ஹைட்ராக்சைடு',
                'dosages': ['1 g/L', '2 g/L', '3 g/L'],
                'quants': ['200 g/Acre', '400 g/Acre', '600 g/Acre'],
                'tips_en': ['Young plants are sensitive to copper; use low concentration.', 'Standard preventive cover.', 'Higher dosage for thick canopy protection.'],
                'tips_ta': ['இளம் செடிகள் தாமிரத்திற்கு உணர்திறன் கொண்டவை; குறைந்த செறிவைப் பயன்படுத்தவும்.', 'நிலையான தடுப்பு நடவடிக்கை.', 'அடர்த்தியான பயிர் பாதுகாப்பிற்கு அதிக அளவு தேவை.']
            },
            'Blast': {
                'pest_en': 'Tricyclazole', 'pest_ta': 'ட்ரைசைக்ளாசோல்',
                'dosages': ['0.5 g/L', '0.6 g/L', '1.0 g/L'],
                'quants': ['150 g/Acre', '200 g/Acre', '300 g/Acre'],
                'tips_en': ['Spray early in nursery stage to prevent neck blast.', 'Ensure uniform coverage on tillers.', 'Vital for protecting the grain head.'],
                'tips_ta': ['நாற்றங்கால் நிலையில் கழுத்து கருகலைத் தடுக்க முன்கூட்டியே தெளிக்கவும்.', 'தூர்களில் சீரான தெளிப்பை உறுதி செய்யவும்.', 'தானியக் கதிர்களைப் பாதுகாக்க மிக முக்கியமானது.']
            }
        }
    },
    'Tomato': {
        'diseases': {
            'EarlyBlight': {
                'pest_en': 'Chlorothalonil', 'pest_ta': 'குளோரோதாலோனில்',
                'dosages': ['1.5 g/L', '2.0 g/L', '2.5 g/L'],
                'quants': ['350 g/Acre', '500 g/Acre', '750 g/Acre'],
                'tips_en': ['Protect young leaves from fungal spotting.', 'Key stage for maintaining leaf area for fruiting.', 'Keep older foliage protected from soil-borne spores.'],
                'tips_ta': ['இளம் இலைகளை பூஞ்சை புள்ளிகளிலிருந்து பாதுகாக்கவும்.', 'காய் பிடிப்பதற்கு இலைப்பரப்பைப் பராமரிக்கும் முக்கிய நிலை.', 'மண்ணிலிருந்து வரும் பூஞ்சை காளான்களிலிருந்து பழைய இலைகளைப் பாதுகாக்கவும்.']
            }
        }
    },
    'Chili': {
        'diseases': {
            'Anthracnose': {
                'pest_en': 'Azoxystrobin', 'pest_ta': 'அசோக்சிஸ்ட்ரோபின்',
                'dosages': ['0.8 ml/L', '1.0 ml/L', '1.5 ml/L'],
                'quants': ['150 ml/Acre', '200 ml/Acre', '300 ml/Acre'],
                'tips_en': ['Protect emerging shoots.', 'Maximize coverage during flower formation.', 'Critical for preventing fruit rot in mature pods.'],
                'tips_ta': ['புதிய தளிர் இலைகளைப் பாதுகாக்கவும்.', 'பூக்கள் உருவாகும் போது தெளிப்பை அதிகரிக்கவும்.', 'முதிர்ந்த காய்களில் அழுகலைத் தடுக்க இது அவசியம்.']
            }
        }
    },
    'Cotton': {
        'diseases': {
            'Bollworm': {
                'pest_en': 'Emamectin Benzoate', 'pest_ta': 'எமாமெக்டின் பென்சோயேட்',
                'dosages': ['0.4 g/L', '0.5 g/L', '0.8 g/L'],
                'quants': ['80 g/Acre', '100 g/Acre', '160 g/Acre'],
                'tips_en': ['Focus on protecting small squares.', 'Protect flowers and developing bolls.', 'Heavy dosage required for dense cotton canopy.'],
                'tips_ta': ['சிறிய பூ மொட்டுகளை பாதுகாப்பதில் கவனம் செலுத்துங்கள்.', 'பூக்கள் மற்றும் வளரும் காய் மொட்டுகளை பாதுகாக்கவும்.', 'அடர்த்தியான பயிர் வளர்ச்சிக்கு அதிக அளவு தேவைப்படுகிறது.']
            }
        }
    },
    'Sugarcane': {
        'diseases': {
            'Red Rot': {
                'pest_en': 'Carbendazim', 'pest_ta': 'கார்பென்டாசிம்',
                'dosages': ['1.0 g/L', '1.5 g/L', '2.0 g/L'],
                'quants': ['250 g/Acre', '350 g/Acre', '500 g/Acre'],
                'tips_en': ['Set treatment is primary; follow up on young shoots.', 'Protect the stalk integrity.', 'Ensure solution reaches the root zone.'],
                'tips_ta': ['நடவு முறை சிகிச்சை முதன்மையானது; இளம் தளிர்களைக் கண்காணிக்கவும்.', 'தண்டுப் பகுதியின் வலிமையைப் பாதுகாக்கவும்.', 'கரைசல் வேர் பகுதியை அடைவதை உறுதி செய்யவும்.']
            }
        }
    }
}

# Expand dataset with more samples
for _ in range(50):
    for crop_name, c_info in crops_data.items():
        for dis_name, d_info in c_info['diseases'].items():
            for age_label, age_idx in ages.items():
                row = {
                    'Crop': crop_name,
                    'Disease': dis_name,
                    'PlantAge': age_label,
                    'Pesticide_en': d_info['pest_en'],
                    'Pesticide_ta': d_info['pest_ta'],
                    'Dosage_en': d_info['dosages'][age_idx],
                    'Dosage_ta': d_info['dosages'][age_idx], # Using EN dosage for simplicity in TA column if needed, or map appropriately
                    'Quantity_en': d_info['quants'][age_idx],
                    'Quantity_ta': d_info['quants'][age_idx],
                    'Tip_en': d_info['tips_en'][age_idx],
                    'Tip_ta': d_info['tips_ta'][age_idx]
                }
                data.append(row)

df = pd.DataFrame(data)
csv_path = 'refined_dosage_dataset.csv'
df.to_csv(csv_path, index=False)
print(f"✅ Generated dataset {csv_path} with {len(df)} rows")

# 2. Train and Save Model
le_crop = LabelEncoder()
le_disease = LabelEncoder()
le_age = LabelEncoder()

X = pd.DataFrame()
X['Crop_Code'] = le_crop.fit_transform(df['Crop'])
X['Disease_Code'] = le_disease.fit_transform(df['Disease'])
X['Age_Code'] = le_age.fit_transform(df['PlantAge'])

# Encode combined output including age-specific tips
df['Output_Class'] = (
    df['Pesticide_en'] + "|" + 
    df['Pesticide_ta'] + "|" + 
    df['Dosage_en'] + "|" + 
    df['Quantity_en'] + "|" + 
    df['Tip_en'] + "|" + 
    df['Tip_ta']
)

le_out = LabelEncoder()
y = le_out.fit_transform(df['Output_Class'])

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

model_path = 'dosage_model.pkl'
joblib.dump({
    'le_crop': le_crop,
    'le_disease': le_disease,
    'le_age': le_age,
    'le_out': le_out,
    'model': clf
}, model_path)

print(f"✅ Refined model trained and saved to {model_path}")
