"""
Firebase helper functions for user and pantry management.
These functions replace the file-based storage with Firestore.
"""
from datetime import datetime
from firebase_config import get_db
import hashlib


def hash_password(password):
    """Hash password using SHA-256 (same as original implementation)."""
    return hashlib.sha256(password.encode()).hexdigest()


# ==================== USER MANAGEMENT ====================

def load_users(use_cache=True):
    """
    Load all users from Firestore.
    Returns a dictionary with user_id as key and user data as value.
    """
    try:
        db = get_db()
        users_ref = db.collection('users')
        docs = users_ref.stream()
        
        users = {}
        for doc in docs:
            user_data = doc.to_dict()
            users[doc.id] = user_data
        
        return users
    except Exception as e:
        print(f"Error loading users from Firestore: {e}")
        return {}


def save_users(users):
    """
    Save users dictionary to Firestore.
    This function updates all users in the 'users' collection.
    """
    try:
        db = get_db()
        users_ref = db.collection('users')
        
        # Get existing user IDs
        existing_docs = users_ref.stream()
        existing_ids = {doc.id for doc in existing_docs}
        
        # Batch write for efficiency
        batch = db.batch()
        batch_count = 0
        max_batch_size = 500  # Firestore batch limit
        
        for user_id, user_data in users.items():
            user_ref = users_ref.document(user_id)
            batch.set(user_ref, user_data)
            batch_count += 1
            
            # Commit batch if it reaches the limit
            if batch_count >= max_batch_size:
                batch.commit()
                batch = db.batch()
                batch_count = 0
        
        # Commit remaining updates
        if batch_count > 0:
            batch.commit()
        
        # Delete users that are no longer in the dictionary
        to_delete = existing_ids - set(users.keys())
        if to_delete:
            delete_batch = db.batch()
            delete_count = 0
            for user_id in to_delete:
                user_ref = users_ref.document(user_id)
                delete_batch.delete(user_ref)
                delete_count += 1
                
                if delete_count >= max_batch_size:
                    delete_batch.commit()
                    delete_batch = db.batch()
                    delete_count = 0
            
            if delete_count > 0:
                delete_batch.commit()
        
        return True
    except Exception as e:
        print(f"Error saving users to Firestore: {e}")
        return False


def get_user_by_id(user_id):
    """Get a single user by ID from Firestore."""
    try:
        db = get_db()
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if user_doc.exists:
            return user_doc.to_dict()
        return None
    except Exception as e:
        print(f"Error getting user {user_id} from Firestore: {e}")
        return None


def create_user_in_firestore(username, email, password, client_type='web'):
    """Create a new user in Firestore."""
    import uuid
    
    try:
        db = get_db()
        users_ref = db.collection('users')
        
        # Normalize inputs
        username_normalized = username.strip().lower() if username else ""
        email_normalized = email.strip().lower() if email else ""
        
        # Validate inputs
        if not username_normalized:
            return None, "Username cannot be empty"
        if not email_normalized:
            return None, "Email cannot be empty"
        
        # Check if username or email already exists
        existing_users = users_ref.stream()
        for user_doc in existing_users:
            user_data = user_doc.to_dict()
            stored_username = user_data.get('username', '').strip().lower()
            stored_email = user_data.get('email', '').strip().lower()
            
            if stored_username == username_normalized:
                return None, f"Username '{username}' already exists"
            if stored_email == email_normalized:
                return None, f"Email '{email}' already exists"
        
        # Create new user
        user_id = str(uuid.uuid4())
        user_data = {
            'id': user_id,
            'username': username,
            'email': email,
            'password_hash': hash_password(password),
            'client_type': client_type,
            'pantry': [],
            'created_at': datetime.now().isoformat(),
            'last_login': None
        }
        
        # Save to Firestore
        users_ref.document(user_id).set(user_data)
        
        return user_id, None
    except Exception as e:
        print(f"Error creating user in Firestore: {e}")
        return None, str(e)


def update_user(user_id, updates):
    """Update a user's data in Firestore."""
    try:
        db = get_db()
        user_ref = db.collection('users').document(user_id)
        
        # Update only provided fields
        user_ref.update(updates)
        return True
    except Exception as e:
        print(f"Error updating user {user_id} in Firestore: {e}")
        return False


# ==================== PANTRY MANAGEMENT ====================

def get_user_pantry(user_id):
    """Get a user's pantry items from Firestore."""
    try:
        user_data = get_user_by_id(user_id)
        if user_data:
            return user_data.get('pantry', [])
        return []
    except Exception as e:
        print(f"Error getting pantry for user {user_id}: {e}")
        return []


def update_user_pantry(user_id, pantry_items):
    """Update a user's pantry items in Firestore."""
    try:
        db = get_db()
        user_ref = db.collection('users').document(user_id)
        
        # pantry_items should already be normalized by app.py's normalize_pantry_item function
        # Just ensure it's a list and save it
        if not isinstance(pantry_items, list):
            pantry_items = []
        
        # Update pantry in Firestore
        user_ref.update({'pantry': pantry_items})
        
        return True
    except Exception as e:
        print(f"Error updating pantry for user {user_id} in Firestore: {e}")
        import traceback
        traceback.print_exc()
        return False


def add_pantry_item(user_id, item):
    """Add a single item to a user's pantry."""
    try:
        pantry = get_user_pantry(user_id)
        pantry.append(item)
        return update_user_pantry(user_id, pantry)
    except Exception as e:
        print(f"Error adding pantry item for user {user_id}: {e}")
        return False


def delete_pantry_item(user_id, item_id):
    """Delete a single item from a user's pantry."""
    try:
        pantry = get_user_pantry(user_id)
        pantry = [item for item in pantry if item.get('id') != item_id]
        return update_user_pantry(user_id, pantry)
    except Exception as e:
        print(f"Error deleting pantry item for user {user_id}: {e}")
        return False
