import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import base64

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = 'fixed_model.keras'   # use your re-saved compatible model
LABELS_PATH = 'class_names.json'
IMG_SIZE = 128
MAX_FILE_MB = 10

# ── Load model & class names ──────────────────────────────────────────────────
print("Loading model...")

model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print(f"✅ Model loaded: {MODEL_PATH}")

with open(LABELS_PATH, 'r') as f:
    CLASS_NAMES = json.load(f)

print(f"✅ {len(CLASS_NAMES)} classes loaded")

# ── Disposal metadata ─────────────────────────────────────────────────────────
WASTE_META = {
    'aerosol_cans': ('♻️ Recyclable', 'metal', 'Empty the can completely. Do NOT puncture.'),
    'aluminum_food_cans': ('♻️ Recyclable', 'metal', 'Rinse clean and recycle.'),
    'aluminum_soda_cans': ('♻️ Recyclable', 'metal', 'Rinse and crush before recycling.'),
    'cardboard_boxes': ('♻️ Recyclable', 'paper', 'Flatten before recycling.'),
    'cardboard_packaging': ('♻️ Recyclable', 'paper', 'Remove plastic inserts first.'),
    'clothing': ('👕 Donate/Textile', 'textile', 'Donate if reusable.'),
    'coffee_grounds': ('🌿 Compostable', 'organic', 'Add to compost bin.'),
    'disposable_plastic_cutlery': ('🚫 General Waste', 'plastic', 'Usually not recyclable.'),
    'eggshells': ('🌿 Compostable', 'organic', 'Crush and compost.'),
    'food_waste': ('🌿 Compostable', 'organic', 'Place in food waste bin.'),
    'glass_beverage_bottles': ('♻️ Recyclable', 'glass', 'Rinse and recycle.'),
    'glass_cosmetic_containers': ('♻️ Recyclable', 'glass', 'Remove lids before recycling.'),
    'glass_food_jars': ('♻️ Recyclable', 'glass', 'Clean before recycling.'),
    'magazines': ('♻️ Recyclable', 'paper', 'Recycle with paper waste.'),
    'newspaper': ('♻️ Recyclable', 'paper', 'Keep dry before recycling.'),
    'office_paper': ('♻️ Recyclable', 'paper', 'Recycle clean paper only.'),
    'paper_cups': ('🚫 General Waste', 'paper', 'Plastic lining makes recycling difficult.'),
    'plastic_cup_lids': ('🚫 General Waste', 'plastic', 'Usually non-recyclable.'),
    'plastic_detergent_bottles': ('♻️ Recyclable', 'plastic', 'Rinse thoroughly first.'),
    'plastic_food_containers': ('♻️ Recyclable', 'plastic', 'Clean before recycling.'),
    'plastic_shopping_bags': ('♻️ Soft Plastic', 'plastic', 'Return to soft plastic collection.'),
    'plastic_soda_bottles': ('♻️ Recyclable', 'plastic', 'Crush and recycle.'),
    'plastic_straws': ('🚫 General Waste', 'plastic', 'Too small for recycling.'),
    'plastic_trash_bags': ('🚫 General Waste', 'plastic', 'Dispose in general waste.'),
    'plastic_water_bottles': ('♻️ Recyclable', 'plastic', 'Recycle after rinsing.'),
    'shoes': ('👟 Donate/Textile', 'textile', 'Donate if wearable.'),
    'steel_food_cans': ('♻️ Recyclable', 'metal', 'Rinse before recycling.'),
    'styrofoam_cups': ('🚫 General Waste', 'styrofoam', 'Not widely recyclable.'),
    'styrofoam_food_containers': ('🚫 General Waste', 'styrofoam', 'Dispose in general waste.'),
    'tea_bags': ('🌿 Compostable', 'organic', 'Compost if biodegradable.')
}

CATEGORY_COLOR = {
    'metal': '#4ade80',
    'paper': '#a3e635',
    'organic': '#86efac',
    'glass': '#38bdf8',
    'plastic': '#fbbf24',
    'textile': '#c084fc',
    'styrofoam': '#f87171',
}

# ── Image preprocessing ───────────────────────────────────────────────────────
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))

    arr = np.array(img, dtype=np.float32) / 255.0

    arr = np.expand_dims(arr, axis=0)

    return arr

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    file_bytes = file.read()

    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        return jsonify({'error': 'File too large'}), 413

    try:
        img_input = preprocess_image(file_bytes)

        predictions = model.predict(img_input, verbose=0)[0]

        top5_idx = np.argsort(predictions)[::-1][:5]

        top5 = []

        for i in top5_idx:
            top5.append({
                'class': CLASS_NAMES[i],
                'label': CLASS_NAMES[i].replace('_', ' ').title(),
                'confidence': round(float(predictions[i]) * 100, 2)
            })

        best_class = CLASS_NAMES[top5_idx[0]]

        meta = WASTE_META.get(
            best_class,
            ('🗑️ General Waste', 'other', 'Dispose properly.')
        )

        img_preview = Image.open(io.BytesIO(file_bytes)).convert('RGB')

        img_preview.thumbnail((300, 300))

        buffer = io.BytesIO()

        img_preview.save(buffer, format='JPEG')

        thumb_b64 = base64.b64encode(buffer.getvalue()).decode()

        return jsonify({
            'success': True,
            'top5': top5,
            'best': {
                'class': best_class,
                'label': best_class.replace('_', ' ').title(),
                'confidence': round(float(predictions[top5_idx[0]]) * 100, 2),
                'disposal': meta[0],
                'category': meta[1],
                'tip': meta[2],
                'color': CATEGORY_COLOR.get(meta[1], '#888888')
            },
            'thumbnail': f'data:image/jpeg;base64,{thumb_b64}',
            'model_info': {
                'name': 'MobileNetV2',
                'classes': len(CLASS_NAMES),
                'accuracy': '82.40%'
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():

    classes = []

    for name in CLASS_NAMES:

        meta = WASTE_META.get(
            name,
            ('🗑️ General Waste', 'other', '')
        )

        classes.append({
            'class': name,
            'label': name.replace('_', ' ').title(),
            'disposal': meta[0],
            'category': meta[1],
            'tip': meta[2],
            'color': CATEGORY_COLOR.get(meta[1], '#888888')
        })

    return jsonify({
        'classes': classes,
        'total': len(classes)
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': 'MobileNetV2',
        'classes': len(CLASS_NAMES)
    })

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
