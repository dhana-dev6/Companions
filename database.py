import os
import bcrypt
import mimetypes
from datetime import datetime
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
from bson.binary import Binary, BINARY_SUBTYPE
from pymongo.errors import DuplicateKeyError

# --- Config and Connection ---

def load_config():
    """Load .env file."""
    load_dotenv()
    print("✅ Environment variables loaded.")

def get_db():
    """Connects to MongoDB and returns the database object."""
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise ValueError("MONGO_URI must be set in .env")
    
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
    Enforces a 50KB size limit.
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
