import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# 1. Define the exact path to your images based on your project structure
# We go UP one level (..) then DOWN into src -> data -> Rice Leaf Disease Images
DATA_DIR = os.path.join("..", "src", "data", "Rice Leaf Disease Images")

print(f"📂 Looking for images in: {DATA_DIR}")

# Check if directory exists and list contents
if os.path.exists(DATA_DIR):
    print(f" Directory found! Contents: {os.listdir(DATA_DIR)}")
else:
    print(f"Directory not found: {DATA_DIR}")
    print("Current working directory:", os.getcwd())
    print("Available paths:")
    for root, dirs, files in os.walk(".."):
        if "Rice Leaf Disease Images" in root:
            print(f"  Found: {root}")
    exit(1)

# 2. Load the Images into TensorFlow
print("🖼️ Loading and preprocessing images...")

try:
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2, # Use 20% of images for testing
        subset="training",
        seed=123,
        image_size=(224, 224), # MobileNet requires exactly 224x224 pixels
        batch_size=32
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )
    
    print("✅ Successfully loaded image datasets!")
    
except Exception as e:
    print(f"❌ Error loading images: {e}")
    print("Make sure your image folders contain at least 2 images each.")
    exit(1)

# Extract the names of the diseases from your folder names
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"✅ Found {num_classes} classes: {class_names}")

# 3. Build the CNN Model (Transfer Learning)
print("🧠 Building the AI Brain...")
# We use a pre-trained model (MobileNetV2) that already knows how to see shapes
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Freeze the base model so we don't ruin its general knowledge

# Add our custom layers to guess the specific rice diseases
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the Model!
print("🚀 Starting training process... (This may take a few minutes)")
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5 # We run through the data 5 times to learn
)

# 6. Save the results
model_path = 'rice_vision_model.h5'
model.save(model_path)
print(f"💾 Model saved successfully as '{model_path}' in your ml_models folder.")

# Save the disease names to a text file so our API knows what they are later
with open('disease_classes.txt', 'w') as f:
    for item in class_names:
        f.write("%s\n" % item)
print("📄 Class names saved to 'disease_classes.txt'.")