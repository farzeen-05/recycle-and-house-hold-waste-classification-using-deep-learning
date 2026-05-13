"""
EcoSort AI — Flask Backend
Model  : MobileNetV2  (waste_classification_model.h5)
Classes: 30
Input  : 128x128 RGB
Acc    : 82.40% validation
"""

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
MODEL_PATH  = 'waste_classification_model.h5'
LABELS_PATH = 'class_names.json'
IMG_SIZE    = 128          # must match training (128x128)
MAX_FILE_MB = 10

# ── Load model & class names ──────────────────────────────────────────────────
print('Loading model...')
model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)
print(f'✅ Model loaded: {MODEL_PATH}')

with open(LABELS_PATH) as f:
    CLASS_NAMES = json.load(f)   # list ordered by class index
print(f'✅ {len(CLASS_NAMES)} classes loaded')

# ── Disposal metadata for all 30 classes ─────────────────────────────────────
WASTE_META = {
    'aerosol_cans':               ('♻️ Recyclable',    'metal',    'Empty the can completely. Do NOT puncture. Place in metal recycling.'),
    'aluminum_food_cans':         ('♻️ Recyclable',    'metal',    'Rinse clean. Crush to save space. Place in metal/aluminum bin.'),
    'aluminum_soda_cans':         ('♻️ Recyclable',    'metal',    'Rinse and crush. Place in aluminum recycling bin.'),
    'cardboard_boxes':            ('♻️ Recyclable',    'paper',    'Flatten completely. Remove tape/staples. Place in cardboard bin.'),
    'cardboard_packaging':        ('♻️ Recyclable',    'paper',    'Remove plastic inserts. Flatten and place in cardboard recycling.'),
    'clothing':                   ('👕 Donate/Textile','textile',  'If wearable, donate. Otherwise take to textile recycling drop-off.'),
    'coffee_grounds':             ('🌿 Compostable',   'organic',  'Add to compost bin. Excellent nitrogen-rich compost material.'),
    'disposable_plastic_cutlery': ('🚫 General Waste', 'plastic',  'Usually not recyclable. Place in general waste bin.'),
    'eggshells':                  ('🌿 Compostable',   'organic',  'Crush and add to compost. Great calcium source for soil.'),
    'food_waste':                 ('🌿 Compostable',   'organic',  'Place in food/organic waste bin or home compost.'),
    'glass_beverage_bottles':     ('♻️ Recyclable',    'glass',    'Rinse clean. Remove caps. Place in glass recycling bin.'),
    'glass_cosmetic_containers':  ('♻️ Recyclable',    'glass',    'Rinse out residue. Remove pumps/lids. Place in glass recycling.'),
    'glass_food_jars':            ('♻️ Recyclable',    'glass',    'Remove lid and rinse. Place in glass recycling bin.'),
    'magazines':                  ('♻️ Recyclable',    'paper',    'Remove plastic covers if any. Place in paper/magazine recycling.'),
    'newspaper':                  ('♻️ Recyclable',    'paper',    'Keep dry. Bundle or place loosely in paper recycling bin.'),
    'office_paper':               ('♻️ Recyclable',    'paper',    'Shred sensitive documents. Place in paper recycling bin.'),
    'paper_cups':                 ('🚫 General Waste', 'paper',    'Most paper cups have plastic lining — check locally. Usually general waste.'),
    'plastic_cup_lids':           ('🚫 General Waste', 'plastic',  'Hard to recycle. Place in general waste unless local scheme accepts them.'),
    'plastic_detergent_bottles':  ('♻️ Recyclable',    'plastic',  'Rinse thoroughly. Replace cap. Place in plastic recycling (HDPE #2).'),
    'plastic_food_containers':    ('♻️ Recyclable',    'plastic',  'Rinse clean. Check resin code. Most #1 and #2 are recyclable.'),
    'plastic_shopping_bags':      ('♻️ Soft Plastic',  'plastic',  'Do NOT put in kerbside bin. Return to supermarket soft-plastic drop-off.'),
    'plastic_soda_bottles':       ('♻️ Recyclable',    'plastic',  'Empty, rinse, replace cap. Place in plastic recycling (PET #1).'),
    'plastic_straws':             ('🚫 General Waste', 'plastic',  'Too small for most recycling. Place in general waste.'),
    'plastic_trash_bags':         ('🚫 General Waste', 'plastic',  'Soft plastic — return to supermarket drop-off or place in general waste.'),
    'plastic_water_bottles':      ('♻️ Recyclable',    'plastic',  'Rinse, crush, replace cap. Place in plastic recycling (PET #1).'),
    'shoes':                      ('👟 Donate/Textile','textile',  'If wearable, donate. Otherwise take to shoe/textile recycling scheme.'),
    'steel_food_cans':            ('♻️ Recyclable',    'metal',    'Rinse clean. Leave lid attached or place inside. Metal recycling bin.'),
    'styrofoam_cups':             ('🚫 General Waste', 'styrofoam','Most councils do not accept EPS foam. Place in general waste.'),
    'styrofoam_food_containers':  ('🚫 General Waste', 'styrofoam','Not widely recyclable. Place in general waste bin.'),
    'tea_bags':                   ('🌿 Compostable',   'organic',  'Most tea bags are compostable. Add to compost or food waste bin.'),
}

CATEGORY_COLOR = {
    'metal':    '#4ade80',
    'paper':    '#a3e635',
    'organic':  '#86efac',
    'glass':    '#38bdf8',
    'plastic':  '#fbbf24',
    'textile':  '#c084fc',
    'styrofoam':'#f87171',
}

# ── Helper: preprocess image exactly as training ──────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0   # rescale=1./255 matches datagen
    return np.expand_dims(arr, axis=0)               # shape (1, 128, 128, 3)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # ── Validate request ──────────────────────────────────────────────────────
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    file_bytes = file.read()
    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        return jsonify({'error': f'File too large (max {MAX_FILE_MB} MB)'}), 413

    try:
        # ── Inference ─────────────────────────────────────────────────────────
        img_input = preprocess_image(file_bytes)
        preds     = model.predict(img_input, verbose=0)[0]   # shape (30,)

        # Top-5 predictions
        top5_idx  = np.argsort(preds)[::-1][:5]
        top5      = [
            {
                'class':      CLASS_NAMES[i],
                'label':      CLASS_NAMES[i].replace('_', ' ').title(),
                'confidence': round(float(preds[i]) * 100, 2),
            }
            for i in top5_idx
        ]

        # Best prediction
        best_class = CLASS_NAMES[top5_idx[0]]
        meta       = WASTE_META.get(best_class, ('🗑️ General Waste', 'other', 'Dispose via your local waste guidelines.'))

        # Return preview thumbnail (base64) so frontend can show it
        img_preview = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        img_preview.thumbnail((300, 300))
        buf = io.BytesIO()
        img_preview.save(buf, format='JPEG', quality=85)
        thumb_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            'success':     True,
            'top5':        top5,
            'best': {
                'class':       best_class,
                'label':       best_class.replace('_', ' ').title(),
                'confidence':  round(float(preds[top5_idx[0]]) * 100, 2),
                'disposal':    meta[0],
                'category':    meta[1],
                'tip':         meta[2],
                'color':       CATEGORY_COLOR.get(meta[1], '#888888'),
            },
            'thumbnail': f'data:image/jpeg;base64,{thumb_b64}',
            'model_info': {
                'name':     'MobileNetV2',
                'classes':  len(CLASS_NAMES),
                'accuracy': '82.40%',
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/classes', methods=['GET'])
def get_classes():
    """Return all 30 class names + metadata."""
    classes = []
    for name in CLASS_NAMES:
        meta = WASTE_META.get(name, ('🗑️ General Waste', 'other', ''))
        classes.append({
            'class':    name,
            'label':    name.replace('_', ' ').title(),
            'disposal': meta[0],
            'category': meta[1],
            'tip':      meta[2],
            'color':    CATEGORY_COLOR.get(meta[1], '#888888'),
        })
    return jsonify({'classes': classes, 'total': len(classes)})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'MobileNetV2', 'classes': len(CLASS_NAMES)})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
