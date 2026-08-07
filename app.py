
import os, json, io, base64, h5py, traceback, sqlite3, datetime
from functools import wraps
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, g
from flask_cors import CORS
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ecosort-dev-secret-key-change-in-production')
CORS(app)

# ── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE    = 128
NUM_CLASSES = 30
MAX_MB      = 10
MODEL_PATH  = 'waste_classification_model.h5'
LABELS_PATH = 'class_names.json'
DB_PATH     = 'ecosort.db'

# ── Database ─────────────────────────────────────────────────────────────────
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                image_name TEXT,
                predicted_class TEXT,
                confidence REAL,
                disposal_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id);
        """)
        db.commit()
        print('Database initialized', flush=True)

# ── Auth helpers ─────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            if request.is_json or request.path.startswith('/predict'):
                return jsonify({'success': False, 'error': 'Authentication required', 'redirect': '/login'}), 401
            return redirect(url_for('login_page', next=request.path))
        return f(*args, **kwargs)
    return decorated

def get_current_user():
    if 'user_id' not in session:
        return None
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    return dict(user) if user else None

# ── Model globals ────────────────────────────────────────────────────────────
_model       = None
_class_names = None
_load_error  = None

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

def load_weights_h5py(model, path):
    arrays = []
    def _walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            arrays.append(np.array(obj))
    with h5py.File(path, 'r') as f:
        root_key = next((k for k in ['layers','model_weights'] if k in f), None)
        root = f[root_key] if root_key else f
        root.visititems(_walk)

    expected = [w.shape for w in model.get_weights()]
    matched  = []
    fi = 0
    for shape in expected:
        while fi < len(arrays):
            if arrays[fi].shape == shape:
                matched.append(arrays[fi]); fi += 1; break
            fi += 1
        else:
            matched = []
            break

    if len(matched) == len(expected):
        model.set_weights(matched)
        return True

    try:
        model.load_weights(path, by_name=False, skip_mismatch=True)
        return True
    except Exception:
        return False

def get_model():
    global _model, _class_names, _load_error
    if _model is not None:
        return _model, _class_names, None
    if _load_error is not None:
        return None, None, _load_error
    try:
        m = build_model()
        load_weights_h5py(m, MODEL_PATH)
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        with open(LABELS_PATH) as f:
            cls = json.load(f)
        _model = m
        _class_names = cls
        return _model, _class_names, None
    except Exception as e:
        _load_error = traceback.format_exc()
        return None, None, _load_error

# ── Disposal metadata ────────────────────────────────────────────────────────
WASTE_META = {
    'aerosol_cans':               ('Recyclable',    'metal',    'Empty completely. Do NOT puncture. Metal recycling bin.'),
    'aluminum_food_cans':         ('Recyclable',    'metal',    'Rinse and crush. Metal recycling bin.'),
    'aluminum_soda_cans':         ('Recyclable',    'metal',    'Rinse and crush. Aluminum recycling bin.'),
    'cardboard_boxes':            ('Recyclable',    'paper',    'Flatten, remove tape. Cardboard bin.'),
    'cardboard_packaging':        ('Recyclable',    'paper',    'Flatten, remove plastic. Cardboard recycling.'),
    'clothing':                   ('Donate/Textile','textile',  'Donate if wearable, else textile drop-off.'),
    'coffee_grounds':             ('Compostable',   'organic',  'Add to compost bin.'),
    'disposable_plastic_cutlery': ('General Waste', 'plastic',  'Usually not recyclable. General waste.'),
    'eggshells':                  ('Compostable',   'organic',  'Crush and add to compost.'),
    'food_waste':                 ('Compostable',   'organic',  'Food/organic waste bin or compost.'),
    'glass_beverage_bottles':     ('Recyclable',    'glass',    'Rinse, remove caps. Glass bin.'),
    'glass_cosmetic_containers':  ('Recyclable',    'glass',    'Rinse, remove pumps. Glass bin.'),
    'glass_food_jars':            ('Recyclable',    'glass',    'Remove lid, rinse. Glass bin.'),
    'magazines':                  ('Recyclable',    'paper',    'Remove plastic covers. Paper bin.'),
    'newspaper':                  ('Recyclable',    'paper',    'Keep dry. Paper bin.'),
    'office_paper':               ('Recyclable',    'paper',    'Shred if sensitive. Paper bin.'),
    'paper_cups':                 ('General Waste', 'paper',    'Plastic-lined — usually general waste.'),
    'plastic_cup_lids':           ('General Waste', 'plastic',  'Hard to recycle. General waste.'),
    'plastic_detergent_bottles':  ('Recyclable',    'plastic',  'Rinse, replace cap. HDPE #2 bin.'),
    'plastic_food_containers':    ('Recyclable',    'plastic',  'Rinse. Most #1 and #2 recyclable.'),
    'plastic_shopping_bags':      ('Soft Plastic',  'plastic',  'Return to supermarket drop-off.'),
    'plastic_soda_bottles':       ('Recyclable',    'plastic',  'Empty, rinse, replace cap. PET #1.'),
    'plastic_straws':             ('General Waste', 'plastic',  'Too small. General waste bin.'),
    'plastic_trash_bags':         ('General Waste', 'plastic',  'Soft plastic — supermarket drop-off.'),
    'plastic_water_bottles':      ('Recyclable',    'plastic',  'Rinse, crush, replace cap. PET #1.'),
    'shoes':                      ('Donate/Textile','textile',  'Donate if wearable. Textile recycling.'),
    'steel_food_cans':            ('Recyclable',    'metal',    'Rinse, leave lid in. Metal bin.'),
    'styrofoam_cups':             ('General Waste', 'styrofoam','EPS foam rarely recyclable.'),
    'styrofoam_food_containers':  ('General Waste', 'styrofoam','Not widely recyclable.'),
    'tea_bags':                   ('Compostable',   'organic',  'Compostable. Add to food waste bin.'),
}
CAT_COLOR = {
    'metal':'#4ade80','paper':'#a3e635','organic':'#86efac',
    'glass':'#38bdf8','plastic':'#fbbf24','textile':'#c084fc','styrofoam':'#f87171',
}

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    user = get_current_user()
    return render_template('index.html', user=user)

# ── Auth Pages ───────────────────────────────────────────────────────────────
@app.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/register')
def register_page():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    data = request.get_json()
    username = (data.get('username') or '').strip().lower()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    full_name = (data.get('full_name') or '').strip()

    errors = {}
    if not username or len(username) < 3:
        errors['username'] = 'Username must be at least 3 characters'
    elif not username.isalnum():
        errors['username'] = 'Username must be alphanumeric'
    if not email or '@' not in email:
        errors['email'] = 'Valid email required'
    if not password or len(password) < 6:
        errors['password'] = 'Password must be at least 6 characters'

    if errors:
        return jsonify({'success': False, 'errors': errors}), 400

    db = get_db()
    existing = db.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email)).fetchone()
    if existing:
        return jsonify({'success': False, 'errors': {'general': 'Username or email already exists'}}), 409

    pw_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
    cursor = db.execute(
        'INSERT INTO users (username, email, password_hash, full_name) VALUES (?, ?, ?, ?)',
        (username, email, pw_hash, full_name)
    )
    db.commit()
    session['user_id'] = cursor.lastrowid
    session['username'] = username

    return jsonify({'success': True, 'user': {'id': cursor.lastrowid, 'username': username, 'email': email}}), 201

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.get_json()
    identifier = (data.get('identifier') or '').strip().lower()
    password = data.get('password') or ''

    if not identifier or not password:
        return jsonify({'success': False, 'errors': {'general': 'Username/email and password required'}}), 400

    db = get_db()
    user = db.execute(
        'SELECT * FROM users WHERE username = ? OR email = ?', (identifier, identifier)
    ).fetchone()

    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'success': False, 'errors': {'general': 'Invalid credentials'}}), 401

    db.execute('UPDATE users SET last_login = ? WHERE id = ?', (datetime.datetime.now(), user['id']))
    db.commit()

    session['user_id'] = user['id']
    session['username'] = user['username']

    return jsonify({'success': True, 'user': {'id': user['id'], 'username': user['username'], 'email': user['email']}}), 200

@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'success': True}), 200

@app.route('/api/auth/me')
def api_me():
    user = get_current_user()
    if not user:
        return jsonify({'authenticated': False}), 200
    return jsonify({
        'authenticated': True,
        'user': {'id': user['id'], 'username': user['username'], 'email': user['email'], 'full_name': user['full_name']}
    }), 200

# ── Predict (auth required) ──────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        model, class_names, load_err = get_model()
        if load_err:
            return jsonify({'success': False, 'error': 'Model failed to load', 'detail': load_err[:500]}), 500

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file in request'}), 400
        f = request.files['file']
        if not f.filename:
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        raw = f.read()
        if len(raw) > MAX_MB * 1024 * 1024:
            return jsonify({'success': False, 'error': 'File too large (max %dMB)' % MAX_MB}), 413

        img_arr = preprocess(raw)
        preds = model.predict(img_arr, verbose=0)[0]
        top5_idx = np.argsort(preds)[::-1][:5]
        best_cls = class_names[top5_idx[0]]
        best_conf = float(preds[top5_idx[0]])

        meta = WASTE_META.get(best_cls, ('General Waste', 'other', 'Check local guidelines.'))

        top5 = [{'class': class_names[i],
                 'label': class_names[i].replace('_',' ').title(),
                 'confidence': round(float(preds[i])*100, 2)}
                for i in top5_idx]

        # Save prediction to DB
        db = get_db()
        db.execute(
            'INSERT INTO predictions (user_id, image_name, predicted_class, confidence, disposal_type) VALUES (?, ?, ?, ?, ?)',
            (session['user_id'], f.filename, best_cls, round(best_conf*100, 2), meta[0])
        )
        db.commit()

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
                'class': best_cls,
                'label': best_cls.replace('_',' ').title(),
                'confidence': round(best_conf*100, 2),
                'disposal': meta[0],
                'category': meta[1],
                'tip': meta[2],
                'color': CAT_COLOR.get(meta[1], '#888'),
            },
            'thumbnail': thumb_b64,
            'model_info': {'name':'MobileNetV2','classes':len(class_names),'accuracy':'82.40%'}
        }), 200

    except Exception:
        tb = traceback.format_exc()
        return jsonify({'success': False, 'error': 'Internal server error', 'detail': tb[-500:]}), 500

# ── Other routes ─────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    model, classes, err = get_model()
    if err:
        return jsonify({'status': 'error', 'error': err}), 500
    return jsonify({'status': 'ok', 'model': 'MobileNetV2', 'classes': len(classes), 'accuracy': '82.40%'})

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

@app.route('/api/history')
@login_required
def get_history():
    db = get_db()
    rows = db.execute(
        'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 50',
        (session['user_id'],)
    ).fetchall()
    return jsonify({'success': True, 'history': [dict(r) for r in rows]})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
