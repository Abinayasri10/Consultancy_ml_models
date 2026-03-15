import numpy as np
from PIL import Image
import api

# Create a white image (like a document)
img = Image.new('RGB', (500, 500), color=(255, 255, 255))
is_valid, msg = api.is_crop_image(img)
print("Is valid:", is_valid)
print("Msg:", msg)
