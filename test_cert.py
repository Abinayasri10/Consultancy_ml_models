import api
from PIL import Image, ImageDraw, ImageFont

# Create a mock certificate image
img = Image.new('RGB', (800, 600), color=(255, 255, 255))
d = ImageDraw.Draw(img)

# add some blue banner
d.rectangle([0, 0, 800, 150], fill=(50, 100, 150))
d.text((100, 50), "Internship Completion Certificate", fill=(255,255,255))
d.text((100, 200), "This is to certify that User has completed...", fill=(0,0,0))
d.ellipse([50, 200, 150, 300], outline=(0,0,255))

is_valid, msg = api.is_crop_image(img)
print(f"\n--- Result ---\nIs valid: {is_valid}\nMessage: {msg}\n")
