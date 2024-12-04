# Start this app using: python3 app.py

from flask import Flask, request, render_template
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

app = Flask(__name__)

# Load BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        # Preprocess the image
        image = Image.open(file).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Generate the caption
        outputs = model.generate(**inputs, max_new_tokens=20)
        description = processor.decode(outputs[0], skip_special_tokens=True)

        return f"<h1>Description:</h1><p>{description}</p>"

if __name__ == "__main__":
    app.run(debug=True)
