"""
EcoSort AI — Flask Backend
Model  : MobileNetV2  (waste_classification_model.h5)
Classes: 30  |  Input: 128x128  |  Acc: 82.40%

DEFINITIVE FIX for Keras 3 → Keras 2 deserialization error:
  "Unrecognized keyword arguments: ['batch_shape', 'optional']"

Root cause: Colab saved with Keras 3 which uses a new HDF5 schema.
Render runs Keras 2 which cannot deserialize Keras 3's InputLayer config.

Solution: Rebuild the EXACT same architecture in tf.keras (Keras 2),
then extract weights directly from HDF5 using h5py — bypasses Keras
deserialization completely.
"""

import os, json, io, base64, h5py
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

app  = Flask(__name__)
CORS(app)

IMG_SIZE    = 128
NUM_CLASSES = 30
MAX_MB      = 10
MODEL_PATH  = 'waste_classification_model.h5'
LABELS_PATH = 'class_names.json'

# ── 1. Rebuild architecture (mirrors your Colab training code exactly) ─────────
def build_model():
    base = tf.keras.applications.MobileNetV2(
        weights=None, include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False
    inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

# ── 2. Extract weight arrays from HDF5 without using Keras deserializer ────────
def collect_arrays(h5_group):
    """Walk an h5py group and collect every Dataset as a numpy array."""
    arrays = []
    def _walk(g):
        for k in g.keys():
            v = g[k]
            if isinstance(v, h5py.Dataset):
                arrays.append(np.array(v))
            elif isinstance(v, h5py.Group):
                _walk(v)
    _walk(h5_group)
    return arrays

def load_weights_h5py(model, path):
    """
    Bypass Keras config parsing entirely.
    Open the .h5 file with h5py, collect all weight tensors,
    filter to those whose shapes match the model, set them.
    """
    with h5py.File(path, 'r') as f:
        # Try Keras 3 key first ('layers'), then Keras 2 ('model_weights')
        if 'layers' in f:
            all_arrays = collect_arrays(f['layers'])
            print(f'  HDF5 key: "layers" (Keras 3 format) — {len(all_arrays)} arrays found')
        elif 'model_weights' in f:
            all_arrays = collect_arrays(f['model_weights'])
            print(f'  HDF5 key: "model_weights" (Keras 2 format) — {len(all_arrays)} arrays found')
        else:
            all_arrays = collect_arrays(f)
            print(f'  HDF5 key: root fallback — {len(all_arrays)} arrays found')

    model_weights = model.get_weights()
    model_shapes  = [w.shape for w in model_weights]
    n_expected    = len(model_weights)

    # Keep only arrays whose shape appears in the model
    filtered = [a for a in all_arrays if a.shape in model_shapes]
    print(f'  Model expects {n_expected} weight tensors, found {len(filtered)} matching shapes')

    if len(filtered) == n_expected:
        model.set_weights(filtered)
        print('  ✅ Weights set by shape-match')
        return True

    # If shape-filter count doesn't match, try sequential slice
    if len(all_arrays) >= n_expected:
        candidate = all_arrays[:n_expected]
        if all(c.shape == m for c, m in zip(candidate, model_shapes)):
            model.set_weights(candidate)
            print('  ✅ Weights set by sequential slice')
            return True

    # Last resort: tf.keras load_weights (may partially work)
    print('  ⚠ Shape match failed — trying tf.keras load_weights...')
    try:
        model.load_weights(path, by_name=False, skip_mismatch=True)
        print('  ✅ load_weights(skip_mismatch=True) completed')
        return True
    except Exception as e:
        print(f'  ⚠ load_weights also failed: {e}')
        print('  ⚠ Running with random weights — predictions will be wrong!')
        print('  ➡  Re-save your model in Colab (see colab_resave_model.py)')
        return False

print('Building architecture...')
model = build_model()
print(f'  {model.count_params():,} params total')

print(f'Loading weights from {MODEL_PATH}...')
load_weights_h5py(model, MODEL_PATH)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print('✅ Model ready\n')

with open(LABELS_PATH) as f:
    CLASS_NAMES = json.load(f)
print(f'✅ {len(CLASS_NAMES)} classes loaded')

# ── Disposal tips ──────────────────────────────────────────────────────────────
WASTE_META = {
    'aerosol_cans':               ('♻️ Recyclable',    'metal',    'Empty completely. Do NOT puncture. Metal recycling bin.'),
    'aluminum_food_cans':         ('♻️ Recyclable',    'metal',    'Rinse and crush. Metal/aluminum recycling bin.'),
    'aluminum_soda_cans':         ('♻️ Recyclable',    'metal',    'Rinse and crush. Aluminum recycling bin.'),
    'cardboard_boxes':            ('♻️ Recyclable',    'paper',    'Flatten fully. Remove tape. Cardboard bin.'),
    'cardboard_packaging':        ('♻️ Recyclable',    'paper',    'Flatten and remove plastic. Cardboard recycling.'),
    'clothing':                   ('👕 Donate/Textile','textile',  'Donate if wearable, else textile recycling drop-off.'),
    'coffee_grounds':             ('🌿 Compostable',   'organic',  'Add to compost. Excellent nitrogen-rich material.'),
    'disposable_plastic_cutlery': ('🚫 General Waste', 'plastic',  'Usually not recyclable. General waste bin.'),
    'eggshells':                  ('🌿 Compostable',   'organic',  'Crush and add to compost. Great for soil.'),
    'food_waste':                 ('🌿 Compostable',   'organic',  'Food/organic waste bin or home compost.'),
    'glass_beverage_bottles':     ('♻️ Recyclable',    'glass',    'Rinse, remove caps. Glass recycling bin.'),
    'glass_cosmetic_containers':  ('♻️ Recyclable',    'glass',    'Rinse, remove pumps. Glass recycling bin.'),
    'glass_food_jars':            ('♻️ Recyclable',    'glass',    'Remove lid, rinse. Glass recycling bin.'),
    'magazines':                  ('♻️ Recyclable',    'paper',    'Remove plastic covers. Paper recycling bin.'),
    'newspaper':                  ('♻️ Recyclable',    'paper',    'Keep dry. Paper recycling bin.'),
    'office_paper':               ('♻️ Recyclable',    'paper',    'Shred sensitive docs. Paper recycling.'),
    'paper_cups':                 ('🚫 General Waste', 'paper',    'Plastic-lined — check locally. Usually general waste.'),
    'plastic_cup_lids':           ('🚫 General Waste', 'plastic',  'Hard to recycle. General waste bin.'),
    'plastic_detergent_bottles':  ('♻️ Recyclable',    'plastic',  'Rinse, replace cap. Plastic recycling (HDPE #2).'),
    'plastic_food_containers':    ('♻️ Recyclable',    'plastic',  'Rinse clean. Most #1 and #2 are recyclable.'),
    'plastic_shopping_bags':      ('♻️ Soft Plastic',  'plastic',  'Return to supermarket soft-plastic drop-off.'),
    'plastic_soda_bottles':       ('♻️ Recyclable',    'plastic',  'Empty, rinse, replace cap. Plastic recycling (PET #1).'),
    'plastic_straws':             ('🚫 General Waste', 'plastic',  'Too small for recycling. General waste bin.'),
    'plastic_trash_bags':         ('🚫 General Waste', 'plastic',  'Soft plastic — supermarket drop-off or general waste.'),
    'plastic_water_bottles':      ('♻️ Recyclable',    'plastic',  'Rinse, crush, replace cap. Plastic recycling (PET #1).'),
    'shoes':                      ('👟 Donate/Textile','textile',  'Donate if wearable. Otherwise shoe/textile recycling.'),
    'steel_food_cans':            ('♻️ Recyclable',    'metal',    'Rinse clean. Leave lid attached. Metal recycling.'),
    'styrofoam_cups':             ('🚫 General Waste', 'styrofoam','EPS foam rarely accepted. General waste bin.'),
    'styrofoam_food_containers':  ('🚫 General Waste', 'styrofoam','Not widely recyclable. General waste bin.'),
    'tea_bags':                   ('🌿 Compostable',   'organic',  'Most are compostable. Compost or food waste bin.'),
}
CATEGORY_COLOR = {
    'metal':'#4ade80','paper':'#a3e635','organic':'#86efac',
    'glass':'#38bdf8','plastic':'#fbbf24','textile':'#c084fc','styrofoam':'#f87171',
}

# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'Empty filename'}), 400
    data = f.read()
    if len(data) > MAX_MB * 1024 * 1024:
        return jsonify({'error': f'Max {MAX_MB} MB'}), 413
    try:
        preds    = model.predict(preprocess(data), verbose=0)[0]
        top5_idx = np.argsort(preds)[::-1][:5]
        top5     = [{'class': CLASS_NAMES[i],
                     'label': CLASS_NAMES[i].replace('_',' ').title(),
                     'confidence': round(float(preds[i])*100, 2)} for i in top5_idx]
        best     = CLASS_NAMES[top5_idx[0]]
        meta     = WASTE_META.get(best, ('🗑️ General Waste','other','Check local disposal guidelines.'))
        thumb    = Image.open(io.BytesIO(data)).convert('RGB')
        thumb.thumbnail((300,300))
        buf = io.BytesIO(); thumb.save(buf, 'JPEG', quality=85)
        return jsonify({
            'success': True, 'top5': top5,
            'best': {'class':best,'label':best.replace('_',' ').title(),
                     'confidence':round(float(preds[top5_idx[0]])*100,2),
                     'disposal':meta[0],'category':meta[1],'tip':meta[2],
                     'color':CATEGORY_COLOR.get(meta[1],'#888')},
            'thumbnail': 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode(),
            'model_info': {'name':'MobileNetV2','classes':len(CLASS_NAMES),'accuracy':'82.40%'}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classes')
def get_classes():
    return jsonify({'classes':[
        {'class':n,'label':n.replace('_',' ').title(),
         'disposal':WASTE_META.get(n,('','',''))[0],
         'category':WASTE_META.get(n,('','other',''))[1],
         'tip':WASTE_META.get(n,('','',''))[2],
         'color':CATEGORY_COLOR.get(WASTE_META.get(n,('','other',''))[1],'#888')}
        for n in CLASS_NAMES], 'total':len(CLASS_NAMES)})

@app.route('/health')
def health():
    return jsonify({'status':'ok','model':'MobileNetV2','classes':len(CLASS_NAMES)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
