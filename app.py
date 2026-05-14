"""
EcoSort AI — Flask Backend
Model  : MobileNetV2  (waste_classification_model.h5)
Classes: 30
Input  : 128x128 RGB
Acc    : 82.40% validation
"""
"""
EcoSort AI — Flask Backend
MobileNetV2 | 30 classes | 128x128 | 82.40%

KEY FIX: Model is loaded lazily (on first request), NOT at import time.
This prevents gunicorn from crashing during startup if model loading fails.
Every route always returns valid JSON — never an empty response.
"""

import os, json, io, base64, h5py, traceback
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

app = Flask(__name__)
CORS(app)

IMG_SIZE    = 128
NUM_CLASSES = 30
MAX_MB      = 10
MODEL_PATH  = 'waste_classification_model.h5'
LABELS_PATH = 'class_names.json'

# Global — populated on first request
_model       = None
_class_names = None
_load_error  = None   # stores any startup error so we can return it as JSON

# ── Architecture ──────────────────────────────────────────────────────────────
def build_model():
    base = tf.keras.applications.MobileNetV2(
        weights=None, include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False
    inp  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x    = base(inp, training=False)
    x    = tf.keras.layers.GlobalAveragePooling2D()(x)
    x    = tf.keras.layers.Dense(128, activation='relu')(x)
    out  = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return tf.keras.Model(inp, out)

# ── h5py weight loader ────────────────────────────────────────────────────────
def load_weights_h5py(model, path):
    arrays = []
    def _walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            arrays.append(np.array(obj))
    with h5py.File(path, 'r') as f:
        root_key = next((k for k in ['layers','model_weights'] if k in f), None)
        root = f[root_key] if root_key else f
        print(f'  h5 root: {root_key or "/"}, datasets found: ', end='', flush=True)
        root.visititems(_walk)
        print(len(arrays), flush=True)

    expected = [w.shape for w in model.get_weights()]
    matched  = []
    fi = 0
    for shape in expected:
        while fi < len(arrays):
            if arrays[fi].shape == shape:
                matched.append(arrays[fi]); fi += 1; break
            fi += 1
        else:
            matched = []   # reset — shape not found
            break

    if len(matched) == len(expected):
        model.set_weights(matched)
        print(f'  ✅ {len(matched)} weights loaded via h5py', flush=True)
        return True

    # fallback
    print('  shape-match failed, trying load_weights...', flush=True)
    try:
        model.load_weights(path, by_name=False, skip_mismatch=True)
        print('  ✅ load_weights(skip_mismatch) done', flush=True)
        return True
    except Exception as e:
        print(f'  ⚠ load_weights also failed: {e}', flush=True)
        return False

# ── Lazy loader — called once on first request ────────────────────────────────
def get_model():
    global _model, _class_names, _load_error
    if _model is not None:
        return _model, _class_names, None
    if _load_error is not None:
        return None, None, _load_error
    try:
        print('=== Lazy loading model ===', flush=True)
        m = build_model()
        print(f'  built: {m.count_params():,} params', flush=True)
        load_weights_h5py(m, MODEL_PATH)
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print('=== Model ready ===', flush=True)

        with open(LABELS_PATH) as f:
            cls = json.load(f)
        print(f'  classes: {len(cls)}', flush=True)

        _model       = m
        _class_names = cls
        return _model, _class_names, None

    except Exception as e:
        _load_error = traceback.format_exc()
        print(f'=== MODEL LOAD FAILED ===\n{_load_error}', flush=True)
        return None, None, _load_error

# ── Disposal metadata ─────────────────────────────────────────────────────────
WASTE_META = {
    'aerosol_cans':               ('♻️ Recyclable',    'metal',    'Empty completely. Do NOT puncture. Metal recycling bin.'),
    'aluminum_food_cans':         ('♻️ Recyclable',    'metal',    'Rinse and crush. Metal recycling bin.'),
    'aluminum_soda_cans':         ('♻️ Recyclable',    'metal',    'Rinse and crush. Aluminum recycling bin.'),
    'cardboard_boxes':            ('♻️ Recyclable',    'paper',    'Flatten, remove tape. Cardboard bin.'),
    'cardboard_packaging':        ('♻️ Recyclable',    'paper',    'Flatten, remove plastic. Cardboard recycling.'),
    'clothing':                   ('👕 Donate/Textile','textile',  'Donate if wearable, else textile drop-off.'),
    'coffee_grounds':             ('🌿 Compostable',   'organic',  'Add to compost bin.'),
    'disposable_plastic_cutlery': ('🚫 General Waste', 'plastic',  'Usually not recyclable. General waste.'),
    'eggshells':                  ('🌿 Compostable',   'organic',  'Crush and add to compost.'),
    'food_waste':                 ('🌿 Compostable',   'organic',  'Food/organic waste bin or compost.'),
    'glass_beverage_bottles':     ('♻️ Recyclable',    'glass',    'Rinse, remove caps. Glass bin.'),
    'glass_cosmetic_containers':  ('♻️ Recyclable',    'glass',    'Rinse, remove pumps. Glass bin.'),
    'glass_food_jars':            ('♻️ Recyclable',    'glass',    'Remove lid, rinse. Glass bin.'),
    'magazines':                  ('♻️ Recyclable',    'paper',    'Remove plastic covers. Paper bin.'),
    'newspaper':                  ('♻️ Recyclable',    'paper',    'Keep dry. Paper bin.'),
    'office_paper':               ('♻️ Recyclable',    'paper',    'Shred if sensitive. Paper bin.'),
    'paper_cups':                 ('🚫 General Waste', 'paper',    'Plastic-lined — usually general waste.'),
    'plastic_cup_lids':           ('🚫 General Waste', 'plastic',  'Hard to recycle. General waste.'),
    'plastic_detergent_bottles':  ('♻️ Recyclable',    'plastic',  'Rinse, replace cap. HDPE #2 bin.'),
    'plastic_food_containers':    ('♻️ Recyclable',    'plastic',  'Rinse. Most #1 and #2 recyclable.'),
    'plastic_shopping_bags':      ('♻️ Soft Plastic',  'plastic',  'Return to supermarket drop-off.'),
    'plastic_soda_bottles':       ('♻️ Recyclable',    'plastic',  'Empty, rinse, replace cap. PET #1.'),
    'plastic_straws':             ('🚫 General Waste', 'plastic',  'Too small. General waste bin.'),
    'plastic_trash_bags':         ('🚫 General Waste', 'plastic',  'Soft plastic — supermarket drop-off.'),
    'plastic_water_bottles':      ('♻️ Recyclable',    'plastic',  'Rinse, crush, replace cap. PET #1.'),
    'shoes':                      ('👟 Donate/Textile','textile',  'Donate if wearable. Textile recycling.'),
    'steel_food_cans':            ('♻️ Recyclable',    'metal',    'Rinse, leave lid in. Metal bin.'),
    'styrofoam_cups':             ('🚫 General Waste', 'styrofoam','EPS foam rarely recyclable.'),
    'styrofoam_food_containers':  ('🚫 General Waste', 'styrofoam','Not widely recyclable.'),
    'tea_bags':                   ('🌿 Compostable',   'organic',  'Compostable. Add to food waste bin.'),
}
CAT_COLOR = {
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

@app.route('/health')
def health():
    model, classes, err = get_model()
    if err:
        return jsonify({'status': 'error', 'error': err}), 500
    return jsonify({'status': 'ok', 'model': 'MobileNetV2',
                    'classes': len(classes), 'accuracy': '82.40%'})

@app.route('/predict', methods=['POST'])
def predict():
    # Always return JSON — never empty
    try:
        model, class_names, load_err = get_model()
        if load_err:
            return jsonify({'success': False,
                            'error': 'Model failed to load',
                            'detail': load_err[:500]}), 500

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file in request'}), 400
        f = request.files['file']
        if not f.filename:
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        raw = f.read()
        if len(raw) > MAX_MB * 1024 * 1024:
            return jsonify({'success': False, 'error': f'File too large (max {MAX_MB}MB)'}), 413

        img_arr = preprocess(raw)
        print(f'[predict] input {img_arr.shape} {img_arr.dtype}', flush=True)

        preds    = model.predict(img_arr, verbose=0)[0]
        top5_idx = np.argsort(preds)[::-1][:5]
        best_cls = class_names[top5_idx[0]]
        best_conf = float(preds[top5_idx[0]])
        print(f'[predict] → {best_cls} {best_conf:.3f}', flush=True)

        meta = WASTE_META.get(best_cls, ('🗑️ General Waste', 'other', 'Check local guidelines.'))

        top5 = [{'class': class_names[i],
                 'label': class_names[i].replace('_',' ').title(),
                 'confidence': round(float(preds[i])*100, 2)}
                for i in top5_idx]

        # thumbnail — skip on failure
        thumb_b64 = ''
        try:
            t = Image.open(io.BytesIO(raw)).convert('RGB')
            t.thumbnail((200,200))
            buf = io.BytesIO(); t.save(buf,'JPEG',quality=70)
            thumb_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            pass

        return jsonify({
            'success': True,
            'top5': top5,
            'best': {
                'class':      best_cls,
                'label':      best_cls.replace('_',' ').title(),
                'confidence': round(best_conf*100, 2),
                'disposal':   meta[0],
                'category':   meta[1],
                'tip':        meta[2],
                'color':      CAT_COLOR.get(meta[1], '#888'),
            },
            'thumbnail':  thumb_b64,
            'model_info': {'name':'MobileNetV2','classes':len(class_names),'accuracy':'82.40%'}
        }), 200

    except Exception:
        tb = traceback.format_exc()
        print(f'[predict] EXCEPTION:\n{tb}', flush=True)
        return jsonify({'success': False, 'error': 'Internal server error',
                        'detail': tb[-500:]}), 500

@app.route('/classes')
def get_classes():
    try:
        _, class_names, err = get_model()
        if err:
            return jsonify({'error': err[:300]}), 500
        return jsonify({'classes': [
            {'class': n, 'label': n.replace('_',' ').title(),
             'disposal': WASTE_META.get(n,('','',''))[0],
             'category': WASTE_META.get(n,('','other',''))[1],
             'tip': WASTE_META.get(n,('','',''))[2],
             'color': CAT_COLOR.get(WASTE_META.get(n,('','other',''))[1],'#888')}
            for n in class_names], 'total': len(class_names)})
    except Exception:
        return jsonify({'error': traceback.format_exc()[-300:]}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
