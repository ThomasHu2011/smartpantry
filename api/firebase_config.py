"""
Firebase configuration and initialization.
Handles both local development (credentials file) and serverless (JSON string) environments.
"""
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

# Global variable to store the Firestore database instance
_db = None


def initialize_firebase():
    """
    Initialize Firebase Admin SDK.
    Supports both local development (credentials file) and serverless (JSON string) environments.
    """
    global _db
    
    # If Firebase is already initialized, return early
    try:
        if firebase_admin._apps:
            _db = firestore.client()
            return
    except Exception:
        pass
    
    try:
        # Check if running in serverless environment (Vercel, Render, etc.)
        credentials_json = os.getenv('FIREBASE_CREDENTIALS_JSON')
        project_id = os.getenv('FIREBASE_PROJECT_ID')
        
        if credentials_json:
            # Serverless environment: credentials are provided as a JSON string
            try:
                # Parse the JSON string
                cred_dict = json.loads(credentials_json)
                cred = credentials.Certificate(cred_dict)
                
                # Initialize with explicit project_id if provided
                if project_id:
                    firebase_admin.initialize_app(cred, {
                        'projectId': project_id
                    })
                else:
                    firebase_admin.initialize_app(cred)
                
                _db = firestore.client()
                print("✅ Firebase initialized from environment variable (serverless)")
                return
            except json.JSONDecodeError as e:
                print(f"⚠️ Error parsing FIREBASE_CREDENTIALS_JSON: {e}")
                raise
        else:
            # Local development: try to load from credentials file
            credentials_path = os.getenv('FIREBASE_CREDENTIALS_PATH')
            
            if not credentials_path:
                # Try common locations
                _app_file_dir = os.path.dirname(os.path.abspath(__file__))
                _root_dir = os.path.dirname(_app_file_dir)
                
                possible_paths = [
                    os.path.join(_app_file_dir, 'firebase-credentials.json'),
                    os.path.join(_root_dir, 'firebase-credentials.json'),
                    os.path.join(_app_file_dir, 'serviceAccountKey.json'),
                    os.path.join(_root_dir, 'serviceAccountKey.json'),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        credentials_path = path
                        break
            
            if credentials_path and os.path.exists(credentials_path):
                cred = credentials.Certificate(credentials_path)
                
                # Initialize with explicit project_id if provided
                if project_id:
                    firebase_admin.initialize_app(cred, {
                        'projectId': project_id
                    })
                else:
                    firebase_admin.initialize_app(cred)
                
                _db = firestore.client()
                print(f"✅ Firebase initialized from credentials file: {credentials_path}")
                return
            else:
                raise FileNotFoundError(
                    f"Firebase credentials not found. "
                    f"Set FIREBASE_CREDENTIALS_JSON (serverless) or "
                    f"FIREBASE_CREDENTIALS_PATH (local) environment variable."
                )
    
    except Exception as e:
        print(f"❌ Error initializing Firebase: {e}")
        raise


def get_db():
    """
    Get the Firestore database instance.
    Initializes Firebase if not already initialized.
    
    🔥 PERFORMANCE: This function uses a singleton pattern to reuse the same
    Firestore client instance across all requests, avoiding connection overhead.
    The client is thread-safe and can be reused in serverless environments.
    
    Returns:
        firestore.Client: The Firestore database client (reused instance)
    """
    global _db
    
    if _db is None:
        initialize_firebase()
    
    return _db
