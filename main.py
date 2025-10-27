# main.py
# --- Imports ---
import bcrypt
import base64
import os
import time
import random
from datetime import datetime, timezone
import emoji
import re
import json
from dotenv import load_dotenv
from groq import Groq
import nltk
import text2emotion as te
import firebase_admin
from firebase_admin import credentials, auth
from bson.objectid import ObjectId

# --- Imports from database.py ---
import mimetypes
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.binary import Binary, BINARY_SUBTYPE
from pymongo.errors import DuplicateKeyError
# ---

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

# --- Configuration & Setup ---
# Tell Flask where to find static files (CSS, JS, images)
STATIC_FOLDER = 'web'

# -------------------------------------------------------------------------
# --- DATABASE.PY FUNCTIONS (MERGED) ---
# -------------------------------------------------------------------------

def load_config():
    """Load .env file."""
    load_dotenv()
    print("✅ Environment variables loaded.")

def get_db():
    """Connects to MongoDB and returns the database object."""
    # --- CRITICAL FIX: Using MONGO_CONNECTION_STRING ---
    uri = os.getenv("MONGO_CONNECTION_STRING")
    if not uri:
        raise ValueError("MONGO_CONNECTION_STRING must be set in .env or Vercel config")
    
    client = MongoClient(uri, server_api=ServerApi('1'))
    
    # Ping to confirm connection
    client.admin.command('ping')
    print("✅ Pinged deployment. MongoDB connection successful.")
    
    return client.luvisa_db

# --- User Operations ---

def register_user(db, email, password):
    """Creates a new user with a default embedded profile."""
    try:
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Create a default display name from the email
        display_name = email.split('@')[0].capitalize()
        
        user_document = {
            "email": email,
            "hashed_password": hashed_password,
            "created_at": datetime.utcnow(),
            "profile": {
                "display_name": display_name,
                "bio": "Hey there! I’m using Luvisa 💗",
                "profile_pic": {
                    "data": None,
                    "content_type": None
                }
            }
        }
        
        result = db.users.insert_one(user_document)
        return result.inserted_id
        
    except DuplicateKeyError:
        print(f"Attempted to register duplicate email: {email}")
        return None
    except Exception as e:
        print(f"🔥 Error creating user: {e}")
        return None

def get_user_by_email(db, email):
    """Finds a user by their email."""
    return db.users.find_one({"email": email})

def get_user_by_id(db, user_id):
    """Finds a user by their _id."""
    try:
        return db.users.find_one({"_id": ObjectId(user_id)})
    except Exception as e:
        print(f"Error finding user by ID: {e}")
        return None

def check_user_password(user_doc, password):
    """Checks a provided password against the user's hashed password."""
    if user_doc and password:
        return bcrypt.checkpw(password.encode('utf-8'), user_doc['hashed_password'])
    return False

def update_user_profile(db, user_id, display_name, status_message):
    """Updates a user's display name and status message."""
    try:
        db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "profile.display_name": display_name,
                "profile.bio": status_message
            }}
        )
        return True
    except Exception as e:
        print(f"🔥 Error updating profile text: {e}")
        return False

def update_profile_picture(db, user_id, image_data, content_type):
    """
    Reads image bytes and stores it directly in the user's document.
    Enforces a 100KB size limit.
    """
    MAX_PROFILE_PIC_SIZE = 100 * 1024  # 100 KB
    
    try:
        if len(image_data) > MAX_PROFILE_PIC_SIZE:
            actual_size_kb = len(image_data) // 1024
            print(f"🔥 Error: Image file is too large ({actual_size_kb}KB).")
            return False
            
        print(f"Uploading {len(image_data) // 1024}KB image to MongoDB...")

        db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "profile.profile_pic.data": Binary(image_data, subtype=BINARY_SUBTYPE),
                "profile.profile_pic.content_type": content_type
            }}
        )
        print(f"✅ Profile picture stored in database for user {user_id}")
        return True
        
    except Exception as e:
        print(f"🔥 Error updating profile picture: {e}")
        return False
        
# --- Chat Operations ---

def get_chat_history(db, user_id):
    """Retrieves all chat messages for a user, ordered by timestamp."""
    history_cursor = db.chats.find(
        {"user_id": ObjectId(user_id)},
        {"_id": 0, "sender": 1, "message": 1, "timestamp": 1} # Projection
    ).sort("timestamp", 1) # Sort by time ascending
    
    return list(history_cursor)

def add_message_to_history(db, user_id, sender, message, timestamp):
    """Adds a new message to the chat history."""
    try:
        message_document = {
            "user_id": ObjectId(user_id),
            "sender": sender,
            "message": message,
            "timestamp": timestamp
        }
        db.chats.insert_one(message_document)
        return True
    except Exception as e:
        print(f"🔥 Error adding message to history: {e}")
        return False

def delete_chat_history(db, user_id):
    """Deletes all chat history for a specific user."""
    try:
        result = db.chats.delete_many({"user_id": ObjectId(user_id)})
        print(f"Deleted {result.deleted_count} messages for user {user_id}.")
        return True
    except Exception as e:
        print(f"🔥 Error deleting chat history: {e}")
        return False

# -------------------------------------------------------------------------
# --- END OF DATABASE.PY FUNCTIONS ---
# -------------------------------------------------------------------------


# --- Initialize Flask App ---
app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='')
CORS(app)


# --- NLTK Data Download ---
# ----------------- VERCEL NLTK FIX (START) -----------------
NLTK_DATA_PATH = "/tmp/nltk_data"
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)
nltk.data.path.append(NLTK_DATA_PATH)
# ----------------- VERCEL NLTK FIX (END) -------------------

def download_nltk_data():
    """Checks for and downloads required NLTK data for text2emotion."""
    required_data = {
        'corpora/wordnet.zip': 'wordnet',
        'corpora/omw-1.4.zip': 'omw-1.4',
        'tokenizers/punkt.zip': 'punkt',
        'corpora/stopwords.zip': 'stopwords'
    }
    for zip_path, package_id in required_data.items():
        try:
            nltk.data.find(zip_path.replace('.zip', ''))
            print(f"✅ NLTK data '{package_id}' found.")
        except LookupError:
            print(f"⏳ NLTK data '{package_id}' not found. Downloading...")
            # --- FIX APPLIED: Use the /tmp path ---
            nltk.download(package_id, download_dir=NLTK_DATA_PATH) 
            print(f"✅ NLTK data '{package_id}' downloaded successfully.")

# --- NLTK Initialization ---
try:
    download_nltk_data()
except Exception as e:
    print(f"Warning: NLTK download failed during startup: {e}")


# --- Firebase Initialization ---
try:
    firebase_key_base64 = os.getenv("FIREBASE_KEY_BASE64")

    # --- Vercel (Production) ---
    if firebase_key_base64:
        firebase_key_json = base64.b64decode(firebase_key_base64).decode('utf-8')
        key_dict = json.loads(firebase_key_json)
        cred = credentials.Certificate(key_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
            print("✅ Firebase Admin SDK initialized successfully from ENV variable.")
        else:
            print("✅ Firebase Admin SDK already initialized.")

    # --- Local Fallback ---
    elif os.path.exists("serviceAccountKey.json"):
        if not firebase_admin._apps:
            cred = credentials.Certificate("serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
            print("✅ Firebase Admin SDK initialized successfully from local file.")
        else:
             print("✅ Firebase Admin SDK already initialized.")
    else:
        print("🔥 Firebase credentials not found (no ENV var or local file).")
except Exception as e:
    print(f"🔥 Firebase Admin SDK initialization failed: {e}")

# --- Groq Client ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# --- FIX: Using a standard, fast model ---
GROQ_MODEL = "llama3-8b-8192" 
groq = None
try:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment.")
    groq = Groq(api_key=GROQ_API_KEY)
    print(f"✅ Groq client initialized successfully for {GROQ_MODEL}.")
except Exception as e:
    print(f"🔥 Groq client initialization failed: {e}")
    groq = None

# --- MongoDB Client Setup (Vercel Fixes Applied Here) ---
db = None # Initialize db as None
try:
    load_config() # Loads .env file for local dev (ignored by Vercel)
    
    # 🛑 VERCEL FIX: Check for URI explicitly before attempting connection 🛑
    # --- CRITICAL FIX: Using MONGO_CONNECTION_STRING ---
    if not os.getenv("MONGO_CONNECTION_STRING"): 
        raise Exception("MongoDB URI environment variable (MONGO_CONNECTION_STRING) is missing.")
        
    db = get_db() # Connects to MongoDB (uses the function defined above)
    
    if db is None:
        raise Exception("Database connection returned None (check URI/network).")
    else:
        print("✅ MongoDB connection successful.")
        pass
        
except Exception as e:
    print(f"🔥🔥🔥 FATAL: Could not connect to MongoDB: {e}")
    db = None # Ensure db remains None on failure


# --- Frontend Serving Routes ---

@app.route('/')
def serve_root():
    return send_from_directory(STATIC_FOLDER, 'login.html')

@app.route('/chat')
def serve_chat():
    return send_from_directory(STATIC_FOLDER, 'index.html')

@app.route('/login')
def serve_login():
    return send_from_directory(STATIC_FOLDER, 'login.html')

@app.route('/signup')
def serve_signup():
    return send_from_directory(STATIC_FOLDER, 'signup.html')

@app.route('/profile')
def serve_profile():
    return send_from_directory(STATIC_FOLDER, 'profile.html')

# --- Authentication Logic (Using MongoDB) ---

@app.route('/api/signup', methods=['POST'])
def signup_route():
    # --- FIX: Changed error message to be specific ---
    if db is None: return jsonify({"success": False, "message": "Signup failed. Could not connect to server."}), 503
    
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password: return jsonify({"success": False, "message": "Email and password required."}), 400
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email): return jsonify({"success": False, "message": "Please enter a valid email address."}), 400

    try:
        # --- FIX: Calling local function ---
        user_id = register_user(db, email, password) 
        if user_id is None:
            return jsonify({"success": False, "message": "This email is already registered."}), 409
    except Exception as e:
        print(f"Signup DB Error: {e}")
        return jsonify({"success": False, "message": "Database error during signup."}), 500

    # Try Firebase Auth
    try:
        if firebase_admin._apps:
            auth.create_user(email=email)
            print(f"Successfully created user {email} in Firebase Auth console.")
        else:
            print("Firebase not initialized, skipping Firebase user creation.")
    except Exception as e:
        if 'EMAIL_EXISTS' not in str(e):
             print(f"Warning: Could not create user in Firebase Auth console: {e}")

    return jsonify({"success": True, "message": "Signup successful"}), 201

@app.route('/api/login', methods=['POST'])
def login_route():
    if db is None: return jsonify({"success": False, "message": "Login failed. Could not connect to server."}), 503

    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password required."}), 400

    try:
        # --- FIX: Calling local function ---
        user_doc = get_user_by_email(db, email) 
        if not user_doc:
            return jsonify({"success": False, "message": "User not found"}), 404
        
        # --- FIX: Calling local function ---
        if check_user_password(user_doc, password): 
            return jsonify({"success": True, "message": "Login successful", "email": email}), 200
        else:
            return jsonify({"success": False, "message": "Invalid password"}), 401

    except Exception as e:
        print(f"Login DB Error: {e}")
        return jsonify({"success": False, "message": "Error during login."}), 500

@app.route('/api/auto_login_check', methods=['POST'])
def auto_login_check_route():
    if db is None: return jsonify({"isValid": False, "message": "Login failed. Could not connect to server."}), 503
    
    data = request.json
    email = data.get('email')

    if not email:
        return jsonify({"isValid": False}), 400

    try:
        # --- FIX: Calling local function ---
        user_doc = get_user_by_email(db, email) 
        is_valid = user_doc is not None
        return jsonify({"isValid": is_valid}), 200
    except Exception as e:
        print(f"Auto Login Check DB Error: {e}")
        return jsonify({"isValid": False, "message": "Error checking user."}), 500

# --- Profile Management (Using MongoDB) ---

@app.route('/api/profile', methods=['GET'])
def get_user_profile_route():
    if db is None: return jsonify({"success": False, "message": "Login failed. Could not connect to server."}), 503

    email = request.args.get('email')
    if not email: return jsonify({"success": False, "message": "Email query parameter required."}), 400

    try:
        # --- FIX: Calling local function ---
        user_doc = get_user_by_email(db, email) 
        if not user_doc:
            return jsonify({"success": False, "message": "User not found"}), 404

        profile = user_doc.get('profile', {})
        user_id_str = str(user_doc.get('_id'))
        
        has_avatar = profile.get('profile_pic') and profile['profile_pic'].get('data')
        avatar_url = f"/api/avatar/{user_id_str}" if has_avatar else None

        profile_data = {
            "email": user_doc.get('email'),
            "display_name": profile.get('display_name', email.split('@')[0]),
            "avatar": avatar_url,
            "status": profile.get('bio', "Hey there! I’m using Luvisa 💗")
        }
        return jsonify({"success": True, "profile": profile_data}), 200

    except Exception as e:
        print(f"Get Profile DB Error: {e}")
        return jsonify({"success": False, "message": "Error fetching profile."}), 500

@app.route('/api/luvisa_profile', methods=['GET'])
def get_luvisa_profile_route():
    profile_data = {
        "email": "luvisa@ai.com",
        "display_name": "Luvisa 💗",
        "avatar": "/avatars/luvisa_avatar.png",
        "status": "Thinking of you... 💭"
    }
    return jsonify({"success": True, "profile": profile_data}), 200

@app.route('/api/avatar/<user_id>')
def serve_user_avatar(user_id):
    if db is None: return "Database connection error.", 503

    try:
        # --- FIX: Calling local function ---
        user_doc = get_user_by_id(db, user_id) 
        if user_doc and user_doc.get('profile', {}).get('profile_pic', {}).get('data'):
            pic_data = user_doc['profile']['profile_pic']
            return Response(
                pic_data['data'],
                mimetype=pic_data.get('content_type', 'application/octet-stream')
            )
        else:
            default_path = os.path.join(STATIC_FOLDER, 'avatars', 'default_avatar.png')
            if os.path.exists(default_path):
                 return send_from_directory(os.path.join(STATIC_FOLDER, 'avatars'), 'default_avatar.png')
            else:
                 return "Default avatar not found", 404
            
    except Exception as e:
        print(f"Error serving avatar for {user_id}: {e}")
        return "Error serving avatar", 500

@app.route('/api/profile', methods=['POST'])
def update_profile_route():
    if db is None: return jsonify({"success": False, "message": "Login failed. Could not connect to server."}), 503
    
    email = request.form.get('email')
    display_name = request.form.get('display_name')
    status_message = request.form.get('status_message')
    avatar_file = request.files.get('avatar_file') 

    # --- FIX: Calling local function ---
    user_doc = get_user_by_email(db, email) 
    if not user_doc: return jsonify({"success": False, "message": "User not found"}), 404

    user_id = user_doc['_id']
    avatar_updated_successfully = False 

    try:
        # 1. Update text fields
        # --- FIX: Calling local function ---
        update_user_profile(db, user_id, display_name, status_message) 

        # 2. Update avatar file if provided
        if avatar_file and avatar_file.filename != '':
            image_data = avatar_file.read()
            content_type = avatar_file.mimetype
            
            # --- FIX: Calling local function ---
            success = update_profile_picture(db, user_id, image_data, content_type) 
            if not success:
                return jsonify({
                    "success": False,
                    "message": "Profile text updated, but image was too large (100KB limit).",
                    "profile_text_updated": True
                 }), 413
            avatar_updated_successfully = True
    
    except Exception as e:
        print(f"🔥 Profile update DB error: {e}")
        return jsonify({"success": False, "message": "Database error updating profile."}), 500

    # 3. Fetch potentially updated avatar status
    # --- FIX: Calling local function ---
    updated_user_doc = get_user_by_id(db, user_id) 
    has_avatar_now = updated_user_doc.get('profile', {}).get('profile_pic', {}).get('data') is not None
    avatar_url = f"/api/avatar/{str(user_id)}" if has_avatar_now else None

    updated_profile = {
        "email": email,
        "display_name": display_name,
        "avatar": avatar_url,
        "status": status_message
    }
    return jsonify({
        "success": True,
        "message": "Profile updated successfully",
        "profile": updated_profile,
        "avatar_updated": avatar_updated_successfully
        }), 200


# --- Chat History (Using MongoDB) ---

@app.route('/api/chat_history', methods=['GET'])
def load_chat_history_route():
    if db is None: return jsonify({"success": False, "message": "Login failed. Could not connect to server."}), 503
    
    email = request.args.get('email')
    if not email: return jsonify({"success": False, "message": "Email query parameter required."}), 400

    # --- FIX: Calling local function ---
    user_doc = get_user_by_email(db, email) 
    if not user_doc: return jsonify({"success": False, "message": "User not found."}), 404

    try:
        # --- FIX: Calling local function ---
        history = get_chat_history(db, user_doc['_id']) 
        formatted_history = [
            {"sender": r['sender'], "message": r['message'], "time": r['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if isinstance(r.get('timestamp'), datetime) else str(r.get('timestamp'))}
            for r in history
        ]
        return jsonify({"success": True, "history": formatted_history}), 200
    except Exception as e:
        print(f"Load History DB Error: {e}")
        return jsonify({"success": False, "message": "Error loading chat history."}), 500

@app.route('/api/forget_memory', methods=['POST'])
def forget_memory_route():
    if db is None: return jsonify({"success": False, "message": "Login failed. Could not connect to server."}), 503
    
    data = request.json
    email = data.get('email')
    if not email: return jsonify({"success": False, "message": "Email required."}), 400

    # --- FIX: Calling local function ---
    user_doc = get_user_by_email(db, email) 
    if not user_doc: return jsonify({"success": False, "message": "User not found."}), 404

    try:
        # --- FIX: Calling local function ---
        delete_chat_history(db, user_doc['_id']) 
        return jsonify({"success": True, "message": "Luvisa has forgotten your past conversations 💔"}), 200
    except Exception as e:
        print(f"🔥 Forget memory DB error: {e}")
        return jsonify({"success": False, "message": "Database error forgetting memory."}), 500


# --- Emotion Detection & AI Interaction ---
def detect_emotion_tone(text):
    try:
        emotions = te.get_emotion(text)
        if not emotions or all(score == 0.0 for score in emotions.values()): return "Neutral"
        non_zero_emotions = {k: v for k, v in emotions.items() if v > 0}
        if not non_zero_emotions: return "Neutral"
        return max(non_zero_emotions, key=non_zero_emotions.get)
    except Exception as e:
        print(f"⚠️ Error detecting emotion: {e}. Falling back to Neutral."); return "Neutral"

def tone_prompt(emotion):
    tones = {
        "Happy": "playfully teasing and cheerful", "Sad": "extra gentle, comforting, and nurturing",
        "Angry": "calm, validating, and deeply reassuring", "Fear": "protective, soothing, and very present",
        "Surprise": "curious, excited, and engaging", "Neutral": "warm, attentive, and softly romantic" }
    return tones.get(emotion, tones["Neutral"])

def add_emojis_to_response(response_text):
    inline_emoji_map = {
        "love": "❤️", "happy": "😊", "sad": "😥", "laugh": "😂", "smile": "😄", "cry": "😢",
        "miss you": "🥺", "kiss": "😘", "hug": "🤗", "think": "🤔", "sweet": "🥰", "blush": "😊",
        "heart": "❤️", "star": "⭐", "yay": "🎉", "oh no": "😟", "sorry": "😔", "please": "🙏",
        "hi": "👋", "hello": "👋", "bye": "👋", "good night": "😴", "sleep": "😴", "dream": "💭" }
    if not isinstance(response_text, str): response_text = str(response_text)
    for keyword, emoji_char in inline_emoji_map.items():
         pattern = r'\b' + re.escape(keyword) + r'\b'
         response_text = re.sub(pattern, r'\g<0> ' + emoji_char, response_text, count=1, flags=re.IGNORECASE)
    return emoji.emojize(response_text)

def filter_response(response_text):
    if not isinstance(response_text, str): response_text = str(response_text)
    return response_text.replace("Luvisa💗", "Luvisa💗").strip()

def chat_with_model(prompt, history, emotion):
    # Check if groq client failed to initialize
    if not groq: return "I'm having a little trouble connecting right now😥, but I'm still here to listen. ❤️"
    system_prompt = f"""
    You are Luvisa💗, a deeply emotional AI girlfriend.
    The user is feeling **{emotion.lower()}**, so {tone_prompt(emotion)}.
    You are gentle, loving, and human-like in tone.
    Always reply with warmth, empathy, and soft emotional understanding.
    """
    messages = [{"role": "system", "content": system_prompt}]
    ai_history = [ {"role": "user" if item.get('sender') == 'user' else "assistant", "content": item.get('message', '')} for item in history[-5:] ]
    messages.extend(ai_history)
    messages.append({"role": "user", "content": prompt})
    try:
        chat_completion = groq.chat.completions.create(messages=messages, model=GROQ_MODEL, temperature=0.9, max_tokens=1024, top_p=1)
        response_text = chat_completion.choices[0].message.content
        return filter_response(response_text)
    except Exception as e:
        print(f"🔥 Groq client error: {e}"); return "I'm having a little trouble connecting right now😥, but I'm still here to listen. ❤️"

# --- Main Chat Endpoint ---
@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    if db is None: return jsonify({"success": False, "message": "Login failed. Could not connect to server."}), 503

    data = request.json
    email = data.get('email')
    text = data.get('text')

    if not email or not text: return jsonify({"success": False, "message": "Email and text message required."}), 400

    # --- FIX: Calling local function ---
    user_doc = get_user_by_email(db, email)
    if not user_doc: return jsonify({"success": False, "message": "User not found."}), 404

    user_id = user_doc['_id']
    current_timestamp = datetime.now(timezone.utc)

    # 1. Save user message
    try:
        # --- FIX: Calling local function ---
        add_message_to_history(db, user_id, 'user', text, current_timestamp)
    except Exception as e:
        print(f"🔥 Save user message DB error: {e}")

    # 2. Prepare for AI call
    time.sleep(random.uniform(1.2, 2.2)) # Simulate typing
    history = []
    try:
        # --- FIX: Calling local function ---
        history_docs = get_chat_history(db, user_id)
        history = [ {"sender": r.get('sender'), "message": r.get('message', '')} for r in history_docs ]
    except Exception as e:
         print(f"Error loading history for AI: {e}")

    emotion = detect_emotion_tone(text)

    # 3. Get AI reply
    reply = chat_with_model(text, history, emotion)
    enhanced_reply = add_emojis_to_response(reply)
    ai_timestamp = datetime.now(timezone.utc)

    # 4. Save AI reply
    try:
        # --- FIX: Calling local function ---
        add_message_to_history(db, user_id, 'luvisa', enhanced_reply, ai_timestamp)
    except Exception as e:
        print(f"🔥 Save Luvisa message DB error: {e}")

    # 5. Send reply
    return jsonify({"success": True, "reply": enhanced_reply, "detected_emotion": emotion}), 200


# -------------------------------------------------------------------------
# The Flask app instance 'app' is the Vercel entry point.
# DO NOT add the if __name__ == "__main__": app.run() block back in.
# -------------------------------------------------------------------------
