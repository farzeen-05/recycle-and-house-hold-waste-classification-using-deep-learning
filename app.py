"""
EcoSort AI — Flask Backend
MobileNetV2 | 30 classes | 128x128 | 82.40% accuracy

FIX: Keras 3 -> Keras 2 weight loading via h5py.
Rebuilds architecture from scratch, then matches weights
by exact shape sequence using a greedy ordered scan.
"""

import os, json, io, base64, h5py, traceback
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

tf.config.set_visible_devices([], 'GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

app  = Flask(__name__)
CORS(app)

IMG_SIZE    = 128
NUM_CLASSES = 30
MAX_MB      = 10
MODEL_PATH  = 'waste_classification_model.h5'
LABELS_PATH = 'class_names.json'

# ── 1. Rebuild exact architecture ─────────────────────────────────────────────
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

# ── 2. Collect arrays from h5 in DEPTH-FIRST order, preserving sequence ───────
def collect_arrays_ordered(h5_path):
    """
    Walk the HDF5 file in a consistent depth-first order and
    return every Dataset as a numpy array, in file-traversal order.
    Keras stores weights in layer order inside the file, so
    preserving that order is essential.
    """
    arrays = []
    def _walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            arrays.append((name, np.array(obj)))
    with h5py.File(h5_path, 'r') as f:
        # Keras 3 uses 'layers', Keras 2 uses 'model_weights'
        root = None
        for key in ['layers', 'model_weights']:
            if key in f:
                root = f[key]
                print(f'  HDF5 root key: "{key}"')
                break
        if root is None:
            root = f
            print('  HDF5 root key: / (fallback)')
        root.visititems(_walk)
    print(f'  Total arrays in file: {len(arrays)}')
    return arrays  # list of (name, ndarray)

# ── 3. Match file arrays to model weight shapes in order ──────────────────────
def match_weights(model, all_arrays):
    """
    The model's get_weights() returns weights in a fixed order.
    We scan the file arrays in order and greedily pick the first
    array whose shape matches the next expected model weight shape.
    This handles extra optimizer/metadata tensors in the file.
    """
    model_weights = model.get_weights()
    expected      = [w.shape for w in model_weights]
    matched       = []
    file_idx      = 0

    for i, shape in enumerate(expected):
        found = False
        while file_idx < len(all_arrays):
            name, arr = all_arrays[file_idx]
            file_idx += 1
            if arr.shape == shape:
                matched.append(arr)
                found = True
                break
            # Skip arrays that don't match (optimizer states, etc.)
        if not found:
            print(f'  ⚠ Could not find array for weight {i} shape={shape}')
            return None

    print(f'  ✅ Matched {len(matched)}/{len(expected)} weight tensors')
    return matched

# ── 4. Load ───────────────────────────────────────────────────────────────────
def load_weights_safe(model, path):
    all_arrays = collect_arrays_ordered(path)
    matched    = match_weights(model, all_arrays)

    if matched and len(matched) == len(model.get_weights()):
        model.set_weights(matched)
        print('  ✅ Weights loaded successfully via h5py shape-match')
        return

    # Fallback: tf.keras load_weights with by_name (needs layer names to match)
    print('  Shape-match incomplete, trying load_weights by_name=False...')
    try:
        model.load_weights(path, by_name=False, skip_mismatch=True)
        print('  ✅ load_weights completed (skip_mismatch=True)')
    except Exception as e:
        print(f'  ⚠ All strategies failed: {e}')
        print('  ➡ Re-save model in Colab with: model.save("model.keras")')

print('Building MobileNetV2...')
model = build_model()
print(f'  Params: {model.count_params():,}')
print(f'  Expected weight shapes: {[w.shape for w in model.get_weights()][:5]}...')

print(f'Loading weights from {MODEL_PATH}...')
load_weights_safe(model, MODEL_PATH)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print('✅ Model ready\n')

with open(LABELS_PATH) as f:
    CLASS_NAMES = json.load(f)
print(f'✅ {len(CLASS_NAMES)} classes loaded')

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

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

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
        # Preprocess
        img_array = preprocess(data)
        print(f'Input shape: {img_array.shape}, dtype: {img_array.dtype}', flush=True)

        # Predict
        preds = model.predict(img_array, verbose=0)[0]
        print(f'Prediction done, top class idx: {int(np.argmax(preds))}, conf: {float(np.max(preds)):.3f}', flush=True)

        top5_idx = np.argsort(preds)[::-1][:5]
        top5 = [
            {
                'class': CLASS_NAMES[i],
                'label': CLASS_NAMES[i].replace('_', ' ').title(),
                'confidence': round(float(preds[i]) * 100, 2)
            }
            for i in top5_idx
        ]
        best = CLASS_NAMES[top5_idx[0]]
        meta = WASTE_META.get(best, ('🗑️ General Waste', 'other', 'Check local disposal guidelines.'))

        # Small thumbnail (skip if it causes issues)
        try:
            thumb = Image.open(io.BytesIO(data)).convert('RGB')
            thumb.thumbnail((200, 200))
            buf = io.BytesIO()
            thumb.save(buf, 'JPEG', quality=70)
            thumb_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            thumb_b64 = ''

        response = {
            'success': True,
            'top5': top5,
            'best': {
                'class':      best,
                'label':      best.replace('_', ' ').title(),
                'confidence': round(float(preds[top5_idx[0]]) * 100, 2),
                'disposal':   meta[0],
                'category':   meta[1],
                'tip':        meta[2],
                'color':      CATEGORY_COLOR.get(meta[1], '#888'),
            },
            'thumbnail':  thumb_b64,
            'model_info': {
                'name':     'MobileNetV2',
                'classes':  len(CLASS_NAMES),
                'accuracy': '82.40%'
            }
        }
        print('Returning response OK', flush=True)
        return jsonify(response), 200

    except Exception as e:
        tb = traceback.format_exc()
        print(f'PREDICT ERROR:\n{tb}', flush=True)
        return jsonify({'success': False, 'error': str(e), 'traceback': tb}), 500

@app.route('/classes')
def get_classes():
    return jsonify({'classes':[
        {'class':n,'label':n.replace('_',' ').title(),
         'disposal':WASTE_META.get(n,('','',''))[0],
         'category':WASTE_META.get(n,('','other',''))[1],
         'tip':WASTE_META.get(n,('','',''))[2],
         'color':CATEGORY_COLOR.get(WASTE_META.get(n,('','other',''))[1],'#888')}
        for n in CLASS_NAMES],'total':len(CLASS_NAMES)})

@app.route('/health')
def health():
    return jsonify({'status':'ok','model':'MobileNetV2','classes':len(CLASS_NAMES)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
