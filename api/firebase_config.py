"""
Firebase configuration and initialization module.
This module handles Firebase Admin SDK setup and provides database access.
"""
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load environment variables
# Try to load from api/.env first, then root .env
_app_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_app_dir, '.env')
if os.path.exists(_env_path):
    load_dotenv(_env_path)
else:
    # Fallback to root .env
    load_dotenv()

# Firebase configuration
FIREBASE_CREDENTIALS_JSON = os.getenv('FIREBASE_CREDENTIALS_JSON')
FIREBASE_CREDENTIALS_PATH = os.getenv('FIREBASE_CREDENTIALS_PATH')
FIREBASE_PROJECT_ID = os.getenv('FIREBASE_PROJECT_ID')

# Initialize Firebase Admin SDK
_firebase_app = None
_db = None


def initialize_firebase():
    """
    Initialize Firebase Admin SDK.
    Supports both JSON string (from environment variable) and file path.
    """
    global _firebase_app, _db
    
    if _firebase_app is not None:
        return _firebase_app, _db
    
    try:
        # Check if Firebase is already initialized
        if firebase_admin._apps:
            _firebase_app = firebase_admin.get_app()
        else:
            # Initialize with credentials
            if FIREBASE_CREDENTIALS_JSON:
                # Use JSON string from environment variable (recommended for serverless)
                cred_dict = json.loads(FIREBASE_CREDENTIALS_JSON)
                cred = credentials.Certificate(cred_dict)
            elif FIREBASE_CREDENTIALS_PATH:
                # Use file path (for local development)
                if os.path.exists(FIREBASE_CREDENTIALS_PATH):
                    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
                else:
                    raise FileNotFoundError(
                        f"Firebase credentials file not found: {FIREBASE_CREDENTIALS_PATH}"
                    )
            else:
                # Try default credentials (for Google Cloud environments)
                cred = credentials.ApplicationDefault()
            
            # Initialize the app
            _firebase_app = firebase_admin.initialize_app(
                cred,
                options={'projectId': FIREBASE_PROJECT_ID} if FIREBASE_PROJECT_ID else {}
            )
        
        # Get Firestore database
        _db = firestore.client()
        
        print("✅ Firebase initialized successfully")
        return _firebase_app, _db
        
    except Exception as e:
        print(f"❌ Error initializing Firebase: {e}")
        raise


def get_db():
    """Get Firestore database instance."""
    if _db is None:
        initialize_firebase()
    return _db


def get_firebase_app():
    """Get Firebase app instance."""
    if _firebase_app is None:
        initialize_firebase()
    return _firebase_app
