import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Load environment variables from a .env file
load_dotenv()


def get_db():
    """Establishes connection to MongoDB and returns the database object."""
    uri = os.getenv("MONGO_CONNECTION_STRING")
    if not uri:
        raise ValueError("MONGO_CONNECTION_STRING must be set in .env")
    
    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))
    
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("✅ Pinged your deployment. You successfully connected to MongoDB!")
        
        # Return your specific database. 
        # I'm using 'luvisa_db' as in your original script.
        return client.luvisa_db
        
    except Exception as e:
        print(f"🔥 Failed to connect to MongoDB: {e}")
        return None

def setup_indexes():
    """Creates recommended indexes on the collections for fast lookups."""
    db = get_db()
    if not db:
        print("🔥 Database connection failed. Cannot setup indexes.")
        return

    try:
        print("Applying indexes...")
        
        # --- Users Collection ---
        # Create a unique index on 'email'
        # This prevents duplicate accounts and speeds up login lookups
        db.users.create_index("email", unique=True)
        print("  -> Index created for: users.email (unique)")

        # --- Chats Collection ---
        # (Renamed from chat_history)
        # Create an index on 'user_id' to quickly find all chats for a specific user
        db.chats.create_index("user_id")
        print("  -> Index created for: chats.user_id")
        
        # --- Note ---
        # The 'profiles' collection is no longer indexed because
        # it will be embedded directly into the 'users' collection.
        
        print("\n✅ MongoDB indexes created/updated successfully.")
        
    except Exception as e:
        print(f"🔥 An error occurred while creating indexes: {e}")

if __name__ == '__main__':
    setup_indexes()