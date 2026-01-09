import os
import json
import hashlib
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from openai import OpenAI
from dotenv import load_dotenv

# Check if running on Vercel (serverless environment)
IS_VERCEL = os.getenv('VERCEL') == '1' or os.getenv('VERCEL_ENV') is not None
# Check if running on Render (persistent server environment)
IS_RENDER = os.getenv('RENDER') == 'true' or 'render.com' in os.getenv('RENDER_EXTERNAL_HOSTNAME', '')

# CORS will be handled via manual headers (no flask-cors dependency needed)
# This approach works perfectly for all use cases and doesn't require additional packages
CORS_AVAILABLE = False

# Load environment variables from .env file (optional - safe for serverless)
# In Vercel, environment variables are set directly, so .env file is optional
try:
    load_dotenv()
except Exception as e:
    # Silently ignore if .env file doesn't exist (common in serverless)
    if not IS_VERCEL:
        print(f"Note: Could not load .env file: {e}")

# Initialize Flask app with explicit configuration for serverless
# Use absolute paths for templates and static files to work in serverless
# Templates and static are now in api/ directory (copied from SmartPantryWeb)
_app_file_dir = os.path.dirname(os.path.abspath(__file__))  # api/ directory
_template_folder = os.path.join(_app_file_dir, 'templates')
_static_folder = os.path.join(_app_file_dir, 'static')

app = Flask(__name__, 
            template_folder=_template_folder,  # Absolute path to api/templates
            static_folder=_static_folder,      # Absolute path to api/static
            static_url_path='/static')         # Static URL path

# Add custom Jinja2 filter to check if item is a dict
@app.template_filter('is_dict')
def is_dict_filter(value):
    """Check if value is a dictionary-like object"""
    return isinstance(value, dict)

# Use environment variable for secret key if available, otherwise use default
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')  # Needed for flash messages

# Configure Flask for serverless environment
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching in serverless (better for debugging)
if IS_VERCEL:
    app.config['TESTING'] = False
    app.config['DEBUG'] = False  # Disable debug in production
else:
    app.config['DEBUG'] = True

# Add 404 error handler for better debugging
@app.errorhandler(404)
def handle_404(e):
    """Handle 404 Not Found errors"""
    from werkzeug.exceptions import NotFound
    
    error_msg = str(e)
    request_path = request.path if hasattr(request, 'path') else 'unknown'
    
    # Log the 404 error
    print(f"\n{'='*60}")
    print(f"404 NOT FOUND: {request_path}")
    print(f"Method: {request.method if hasattr(request, 'method') else 'unknown'}")
    print(f"Headers: {dict(request.headers) if hasattr(request, 'headers') else 'unknown'}")
    print(f"{'='*60}\n")
    
    # Check if this is an API request
    is_api_request = request_path.startswith('/api/') if request_path else False
    
    if is_api_request:
        return jsonify({
            'success': False,
            'error': f'Route not found: {request_path}',
            'path': request_path
        }), 404
    else:
        # For HTML routes, redirect to home page with a flash message
        # This ensures users can navigate back to a working page
        try:
            flash(f"Page not found: {request_path}. Redirecting to home.", "warning")
            return redirect(url_for('index'))
        except Exception as e:
            # Fallback if redirect fails
            print(f"Error in 404 handler: {e}")
            return f"""
            <html>
                <head><title>404 - Page Not Found</title>
                <meta http-equiv="refresh" content="3;url=/">
                </head>
                <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1>404 - Page Not Found</h1>
                    <p>The requested URL <strong>{request_path}</strong> was not found.</p>
                    <p>Redirecting to <a href="/">home page</a>...</p>
                </body>
            </html>
            """, 404

# Add global error handler for better debugging in serverless
@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler - logs errors for debugging in Vercel"""
    from werkzeug.exceptions import NotFound, HTTPException
    
    # Don't handle 404 here - it's handled by handle_404
    if isinstance(e, NotFound):
        return handle_404(e)
    
    error_msg = str(e)
    error_type = type(e).__name__
    
    # Log the error (will appear in Vercel logs) - critical for debugging
    print(f"\n{'='*60}")
    print(f"ERROR [{error_type}]: {error_msg}")
    print(f"{'='*60}")
    import traceback
    traceback.print_exc()
    print(f"{'='*60}\n")
    
    # Return safe error response
    try:
        # Check if this is an API request
        request_path = request.path if hasattr(request, 'path') else ''
        is_api_request = request_path.startswith('/api/') or \
                        (hasattr(request, 'is_json') and request.is_json)
        
        if is_api_request:
            return jsonify({
                'success': False,
                'error': error_msg,
                'type': error_type
            }), 500
        else:
            # For HTML routes, return simple error page
            return f"""
            <html>
                <head><title>Error</title></head>
                <body>
                    <h1>An error occurred</h1>
                    <p><strong>Error:</strong> {error_msg}</p>
                    <p><strong>Type:</strong> {error_type}</p>
                    <p><a href="/">Go to Home</a></p>
                </body>
            </html>
            """, 500
    except Exception as fallback_error:
        # Ultimate fallback - return plain text error
        print(f"Error handler failed: {fallback_error}")
        return f"Error: {error_msg} (Type: {error_type})", 500

# Configure session for serverless environment
# Flask's default cookie-based sessions work on both Render and Vercel
# On Render, sessions persist in cookies (survives server restarts)
# On Vercel, sessions are stateless but work within a single request
# Make sure secret_key is set (done above) for secure session cookies
if IS_RENDER:
    # On Render, ensure sessions are permanent and secure
    app.config['PERMANENT_SESSION_LIFETIME'] = 86400 * 7  # 7 days
    app.config['SESSION_COOKIE_SECURE'] = False  # Set to True if using HTTPS
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
elif IS_VERCEL:
    # For serverless, we'll use Flask's default cookie-based sessions
    # Note: Sessions persist in cookies, so they work across function invocations
    app.config['PERMANENT_SESSION_LIFETIME'] = 86400 * 7  # 7 days
    app.config['SESSION_COOKIE_SECURE'] = True  # HTTPS required on Vercel
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Enable CORS for all domains on all routes using manual headers
# Manual headers work perfectly and don't require flask-cors package
@app.after_request
def after_request(response):
    """Add CORS headers to all responses - no flask-cors dependency needed"""
    # Allow all origins (for development - restrict in production)
    response.headers.add('Access-Control-Allow-Origin', '*')
    # Allow common headers including custom ones
    response.headers.add('Access-Control-Allow-Headers', 
                       'Content-Type,Authorization,X-User-ID,X-Client-Type')
    # Allow all common HTTP methods
    response.headers.add('Access-Control-Allow-Methods', 
                       'GET,PUT,POST,DELETE,OPTIONS,PATCH,HEAD')
    # Handle preflight OPTIONS requests properly
    if request.method == 'OPTIONS':
        response.status_code = 200
    return response

# ✅ Initialize OpenAI client with API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("⚠️  WARNING: OPENAI_API_KEY not found in environment variables")
    print("   Some features (photo recognition, recipe suggestions) will not work.")
    if IS_RENDER:
        print("   On Render: Go to Dashboard → Your Service → Environment → Add OPENAI_API_KEY")
    else:
        print("   Create a .env file with: OPENAI_API_KEY=your_api_key_here")
    client = None
else:
    try:
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully")
    except Exception as e:
        print(f"⚠️  ERROR: Failed to initialize OpenAI client: {e}")
        client = None

 

# Separate pantry lists for different clients (for non-authenticated users)
web_pantry = []
mobile_pantry = []

# User management system
# In serverless, use /tmp directory for temporary storage or in-memory storage
# On Render, use persistent file storage
if IS_VERCEL:
    # Use /tmp directory which is writable in Vercel serverless functions
    USERS_FILE = os.path.join('/tmp', 'users.json')
    # In-memory fallback if file operations fail
    _in_memory_users = {}
elif IS_RENDER:
    # On Render, use /tmp for persistent storage (survives restarts but not redeploys)
    # For production, consider using a database
    USERS_FILE = os.path.join('/tmp', 'users.json')
    _in_memory_users = {}
else:
    # Local development: use api directory (where app.py is located)
    # Use the same _app_file_dir that's already defined for templates
    USERS_FILE = os.path.join(_app_file_dir, 'users.json')
    _in_memory_users = {}
    print(f"📁 Local development: Using USERS_FILE = {USERS_FILE}")
    print(f"   App directory: {_app_file_dir}")

# Performance optimization: Add caching for users and pantry data
_users_cache = {}
_users_cache_timestamp = {}
_users_cache_ttl = 5  # Cache for 5 seconds (balance between performance and freshness)
_pantry_cache = {}  # Cache normalized pantry items per user
_pantry_cache_timestamp = {}

# Enable/disable verbose logging (set to False for production)
VERBOSE_LOGGING = os.getenv('VERBOSE_LOGGING', 'false').lower() == 'true'

def load_users(use_cache=True):
    """Load users from JSON file or in-memory storage with caching"""
    global _in_memory_users, _users_cache, _users_cache_timestamp  # Declare global at the start of the function
    
    # Check cache first
    if use_cache:
        import time
        current_time = time.time()
        if '_all_users' in _users_cache:
            cache_age = current_time - _users_cache_timestamp.get('_all_users', 0)
            if cache_age < _users_cache_ttl:
                if VERBOSE_LOGGING:
                    print(f"Using cached users data (age: {cache_age:.2f}s)")
                return _users_cache['_all_users'].copy()
    
    if IS_VERCEL or IS_RENDER:
        # Try to load from /tmp, fallback to in-memory
        users = {}
        
        # First, try to load from new location (/tmp/users.json)
        try:
            if os.path.exists(USERS_FILE):
                with open(USERS_FILE, 'r') as f:
                    users = json.load(f)
                    if VERBOSE_LOGGING:
                        print(f"Loaded {len(users)} users from {USERS_FILE}")
        except (IOError, OSError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load users file from {USERS_FILE}: {e}")
        
        # Also check old location (users.json in current directory) and migrate
        old_users_file = 'users.json'
        try:
            if os.path.exists(old_users_file):
                with open(old_users_file, 'r') as f:
                    old_users = json.load(f)
                    print(f"Found {len(old_users)} users in old location ({old_users_file}), migrating...")
                    # Merge old users into current users (old users take precedence if duplicate)
                    for user_id, user_data in old_users.items():
                        if user_id not in users:
                            users[user_id] = user_data
                            print(f"Migrated user: {user_data.get('username', 'unknown')}")
                    # Save merged users to new location
                    if old_users:
                        _in_memory_users = users.copy()
                        try:
                            os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
                            with open(USERS_FILE, 'w') as f:
                                json.dump(users, f, indent=2)
                            print(f"Migrated {len(old_users)} users to {USERS_FILE}")
                        except Exception as e:
                            print(f"Warning: Could not save migrated users: {e}")
        except (IOError, OSError, json.JSONDecodeError) as e:
            print(f"Note: Could not check old users file: {e}")
        
        # Update in-memory cache
        _in_memory_users = users.copy()
        
        if not users:
            if VERBOSE_LOGGING:
                print(f"No users found, using in-memory storage")
            users = _in_memory_users.copy() if _in_memory_users else {}
        
        # Update cache
        import time
        _users_cache['_all_users'] = users.copy()
        _users_cache_timestamp['_all_users'] = time.time()
        
        return users
    else:
        # Local development: use file system
        users = {}
        
        # First, try to load from the new location (api/users.json)
        try:
            if os.path.exists(USERS_FILE):
                with open(USERS_FILE, 'r') as f:
                    users = json.load(f)
                    if VERBOSE_LOGGING:
                        print(f"✅ Loaded {len(users)} users from {USERS_FILE}")
                    return users
        except (IOError, json.JSONDecodeError) as e:
            print(f"⚠️ Warning: Could not load users file from {USERS_FILE}: {e}")
        
        # Also check old locations and migrate
        old_locations = [
            'users.json',  # Current directory
            os.path.join('SmartPantryWeb', 'users.json'),  # Old SmartPantryWeb location
            os.path.join('..', 'SmartPantryWeb', 'users.json'),  # Relative path
        ]
        
        for old_file in old_locations:
            try:
                if os.path.exists(old_file):
                    with open(old_file, 'r') as f:
                        old_users = json.load(f)
                        if old_users and isinstance(old_users, dict):
                            print(f"📦 Found {len(old_users)} users in old location ({old_file}), migrating...")
                            # Merge old users into current users
                            for user_id, user_data in old_users.items():
                                if user_id not in users:
                                    users[user_id] = user_data
                                    print(f"   Migrated user: {user_data.get('username', 'unknown')} (ID: {user_id})")
                            
                            # Save migrated users to new location
                            if old_users:
                                try:
                                    file_dir = os.path.dirname(USERS_FILE)
                                    if file_dir and file_dir != '.':
                                        os.makedirs(file_dir, exist_ok=True)
                                    with open(USERS_FILE, 'w') as f:
                                        json.dump(users, f, indent=2)
                                    print(f"✅ Migrated {len(old_users)} users to {USERS_FILE}")
                                except Exception as e:
                                    print(f"⚠️ Warning: Could not save migrated users: {e}")
            except (IOError, json.JSONDecodeError) as e:
                # Continue checking other locations
                pass
        
        if not users:
            if VERBOSE_LOGGING:
                print(f"📁 Users file does not exist at {USERS_FILE}, starting with empty users")
                print(f"   Checked locations: {[USERS_FILE] + old_locations}")
        
        # Update cache
        import time
        _users_cache['_all_users'] = users.copy()
        _users_cache_timestamp['_all_users'] = time.time()
        
        return users

def save_users(users):
    """Save users to JSON file or in-memory storage and update cache"""
    global _users_cache, _users_cache_timestamp
    
    # Invalidate cache on save
    if '_all_users' in _users_cache:
        del _users_cache['_all_users']
    # Also invalidate all pantry caches since users data changed
    _pantry_cache.clear()
    _pantry_cache_timestamp.clear()
    
    if IS_VERCEL or IS_RENDER:
        # Try to save to /tmp, always update in-memory cache
        global _in_memory_users
        _in_memory_users = users.copy()  # Always update in-memory first
        
        try:
            # Ensure /tmp directory exists
            os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
            # Use atomic write: write to temp file first, then rename
            temp_file = USERS_FILE + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(users, f, indent=2)
                f.flush()  # Force write to disk
                os.fsync(f.fileno())  # Ensure data is written to disk
            
            # Atomic rename (works on Unix-like systems)
            os.replace(temp_file, USERS_FILE)
            if VERBOSE_LOGGING:
                print(f"Saved {len(users)} users to {USERS_FILE}")
        except (IOError, OSError) as e:
            # Fallback to in-memory storage if file write fails
            print(f"Warning: Could not save users file: {e}. Using in-memory storage.")
    else:
        # Local development: use file system with atomic write
        try:
            # Ensure directory exists
            file_dir = os.path.dirname(USERS_FILE)
            if file_dir and file_dir != '.':
                os.makedirs(file_dir, exist_ok=True)
                if VERBOSE_LOGGING:
                    print(f"📁 Ensured directory exists: {file_dir}")
            elif not file_dir or file_dir == '':
                # If no directory, file is in current directory
                if VERBOSE_LOGGING:
                    print(f"📁 Saving to current directory: {os.getcwd()}")
            
            if VERBOSE_LOGGING:
                print(f"💾 Attempting to save {len(users)} users to {USERS_FILE}")
                print(f"   Absolute path: {os.path.abspath(USERS_FILE)}")
                print(f"   Current working directory: {os.getcwd()}")
                print(f"   File directory exists: {os.path.exists(file_dir) if file_dir else 'N/A'}")
            
            # Use atomic write: write to temp file first, then rename
            temp_file = USERS_FILE + '.tmp'
            if VERBOSE_LOGGING:
                print(f"   Writing to temp file: {temp_file}")
            
            with open(temp_file, 'w') as f:
                json.dump(users, f, indent=2)
                f.flush()  # Force write to disk
                os.fsync(f.fileno())  # Ensure data is written to disk
            
            if VERBOSE_LOGGING:
                print(f"   Temp file written, size: {os.path.getsize(temp_file)} bytes")
            
            # Atomic rename
            os.replace(temp_file, USERS_FILE)
            if VERBOSE_LOGGING:
                print(f"✅ Saved {len(users)} users to {USERS_FILE}")
            
            # Verify the save immediately (only in verbose mode for performance)
            if VERBOSE_LOGGING and os.path.exists(USERS_FILE):
                file_size = os.path.getsize(USERS_FILE)
                print(f"   File exists, size: {file_size} bytes")
                with open(USERS_FILE, 'r') as f:
                    verify = json.load(f)
                    if len(verify) == len(users):
                        print(f"✅ Verified: {len(verify)} users saved correctly to {USERS_FILE}")
                        # Print user IDs for debugging
                        if users:
                            print(f"   User IDs in file: {list(verify.keys())}")
                            # Print first user as sample
                            first_user_id = list(verify.keys())[0]
                            first_user = verify[first_user_id]
                            print(f"   Sample user: {first_user.get('username', 'unknown')} ({first_user_id})")
                    else:
                        print(f"⚠️ Warning: Saved {len(users)} users but file contains {len(verify)} users")
                        print(f"   Expected IDs: {list(users.keys())}")
                        print(f"   Found IDs: {list(verify.keys())}")
            else:
                print(f"❌ Error: File {USERS_FILE} does not exist after save!")
                print(f"   Attempting direct write...")
                # Try direct write as fallback
                try:
                    with open(USERS_FILE, 'w') as f:
                        json.dump(users, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    print(f"   ✅ Direct write succeeded")
                except Exception as e3:
                    print(f"   ❌ Direct write also failed: {e3}")
        except (IOError, OSError, PermissionError) as e:
            print(f"❌ Error: Could not save users file to {USERS_FILE}: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            # Try to save to a fallback location
            try:
                fallback_file = 'users.json'
                print(f"   Attempting fallback save to: {fallback_file}")
                with open(fallback_file, 'w') as f:
                    json.dump(users, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                print(f"⚠️ Saved to fallback location: {fallback_file}")
            except Exception as e2:
                print(f"❌ Could not save to fallback location either: {e2}")
                import traceback
                traceback.print_exc()

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, email, password, client_type='web'):
    """Create a new user"""
    users = load_users()
    
    # Normalize inputs for case-insensitive comparison
    username_normalized = username.strip().lower() if username else ""
    email_normalized = email.strip().lower() if email else ""
    
    # Validate inputs
    if not username_normalized:
        return None, "Username cannot be empty"
    if not email_normalized:
        return None, "Email cannot be empty"
    
    # Check if username or email already exists (case-insensitive, regardless of client_type)
    for user_id, user_data in users.items():
        stored_username = user_data.get('username', '').strip().lower()
        stored_email = user_data.get('email', '').strip().lower()
        
        if stored_username == username_normalized:
            return None, f"Username '{username}' already exists (case-insensitive)"
        if stored_email == email_normalized:
            return None, f"Email '{email}' already exists (case-insensitive)"
    
    # Create new user
    user_id = str(uuid.uuid4())
    users[user_id] = {
        'id': user_id,
        'username': username,
        'email': email,
        'password_hash': hash_password(password),
        'client_type': client_type,  # Store for reference but don't restrict login
        'pantry': [],
        'created_at': datetime.now().isoformat(),
        'last_login': None
    }
    
    # Save users immediately and ensure it's written to disk
    print(f"💾 Saving user '{username}' (ID: {user_id}) to {USERS_FILE}...")
    print(f"   Total users to save: {len(users)}")
    print(f"   User data: username={username}, email={email}, pantry={len(users[user_id]['pantry'])} items")
    
    try:
        save_users(users)
        print(f"✅ save_users() completed without exception")
    except Exception as e:
        print(f"❌ ERROR in save_users(): {e}")
        import traceback
        traceback.print_exc()
        # Still try to continue, but log the error
    
    # Verify the save was successful by reloading
    # This ensures the file is actually written before returning
    print(f"🔍 Verifying save by reloading users from {USERS_FILE}...")
    print(f"   File exists: {os.path.exists(USERS_FILE)}")
    if os.path.exists(USERS_FILE):
        print(f"   File size: {os.path.getsize(USERS_FILE)} bytes")
    
    verify_users = load_users()
    print(f"   Loaded {len(verify_users)} users after save")
    
    if user_id not in verify_users:
        print(f"⚠️ Warning: User {user_id} not found after save, retrying...")
        print(f"   Users in file: {list(verify_users.keys())}")
        try:
            save_users(users)  # Retry once
            # Verify again
            verify_users = load_users()
            if user_id not in verify_users:
                print(f"❌ Error: User {user_id} still not found after retry!")
                print(f"   Current users in file: {list(verify_users.keys())}")
                print(f"   Expected user ID: {user_id}")
                print(f"   File path: {os.path.abspath(USERS_FILE)}")
                # Try to write directly as a last resort
                try:
                    with open(USERS_FILE, 'w') as f:
                        json.dump(users, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    print(f"   ✅ Direct write succeeded, verifying...")
                    verify_users = load_users()
                    if user_id in verify_users:
                        print(f"   ✅ User found after direct write!")
                    else:
                        print(f"   ❌ User still not found after direct write")
                except Exception as e2:
                    print(f"   ❌ Direct write also failed: {e2}")
            else:
                print(f"✅ User {user_id} found after retry")
        except Exception as e:
            print(f"❌ Error during retry: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"✅ User {user_id} verified in saved file")
        print(f"   User data in file: username={verify_users[user_id].get('username')}, email={verify_users[user_id].get('email')}")
    
    print(f"✅ User created successfully: {username} (ID: {user_id})")
    return user_id, None

def authenticate_user(username, password, client_type='web'):
    """Authenticate user and return user data"""
    users = load_users()
    
    # Normalize input (strip whitespace and convert to lowercase for comparison)
    username_normalized = username.strip().lower() if username else ""
    password = password.strip() if password else ""
    
    # Debug: print number of users loaded
    print(f"Authenticating user '{username}' (normalized: '{username_normalized}') against {len(users)} users")
    
    if not users:
        print("ERROR: No users found in database!")
        return None, "No users found. Please sign up first."
    
    if not username_normalized or not password:
        print("ERROR: Username or password is empty!")
        return None, "Username and password are required"
    
    # Debug: print all usernames and emails (without passwords)
    user_list = []
    for uid, udata in users.items():
        user_list.append({
            'username': udata.get('username', 'N/A'),
            'email': udata.get('email', 'N/A'),
            'id': uid
        })
    print(f"Available users in database: {user_list}")
    
    password_hash = hash_password(password)
    print(f"Password hash for provided password: {password_hash[:20]}...")
    
    # Track if we found a matching username but wrong password
    found_username = False
    found_email = False
    matched_username = None
    
    for user_id, user_data in users.items():
        stored_username = user_data.get('username', '').strip()
        stored_email = user_data.get('email', '').strip()
        stored_password_hash = user_data.get('password_hash', '')
        
        # Case-insensitive matching for username and email
        username_match = stored_username.lower() == username_normalized
        email_match = stored_email.lower() == username_normalized
        password_match = stored_password_hash == password_hash
        
        print(f"Checking user {user_id}: username='{stored_username}' (match: {username_match}), email='{stored_email}' (match: {email_match}), password_match={password_match}")
        
        if (username_match or email_match) and password_match:
            # Update last login
            user_data['last_login'] = datetime.now().isoformat()
            save_users(users)
            
            print(f"✅ Authentication successful for user '{username}' (ID: {user_id})")
            return user_data, None
        elif username_match or email_match:
            found_username = True
            found_email = email_match
            matched_username = stored_username
            print(f"❌ Password mismatch for user '{stored_username}' - stored hash: {stored_password_hash[:20]}..., provided hash: {password_hash[:20]}...")
            print(f"   Stored hash length: {len(stored_password_hash)}, Provided hash length: {len(password_hash)}")
            if stored_password_hash != password_hash:
                print(f"   Hashes do not match!")
    
    # Provide specific error messages
    if found_username:
        if found_email:
            print(f"❌ Authentication failed: Incorrect password for email '{username}' (username: '{matched_username}')")
            return None, f"Incorrect password for email '{username}'. Please check your password and try again."
        else:
            print(f"❌ Authentication failed: Incorrect password for username '{username}'")
            return None, f"Incorrect password for username '{username}'. Please check your password and try again."
    else:
        print(f"❌ Authentication failed: Username or email '{username}' not found in database")
        print(f"   Searched for (normalized): '{username_normalized}'")
        print(f"   Available usernames: {[u.get('username', 'N/A') for u in user_list]}")
        print(f"   Available emails: {[u.get('email', 'N/A') for u in user_list]}")
        return None, f"Username or email '{username}' not found. Please check your username/email or sign up for a new account."

def is_expiring_soon(exp_date_str, days_threshold=7):
    """Check if expiration date is within threshold days"""
    if not exp_date_str or exp_date_str == 'None' or exp_date_str == '' or exp_date_str == 'null':
        return False
    try:
        exp_date_str = str(exp_date_str).strip()
        if len(exp_date_str) >= 10:
            exp_date_only = exp_date_str[:10]
        else:
            exp_date_only = exp_date_str
        
        # Parse date
        if 'T' in exp_date_only:
            exp_date = datetime.fromisoformat(exp_date_only.replace('Z', '+00:00')).date()
        else:
            exp_date = datetime.strptime(exp_date_only, "%Y-%m-%d").date()
        
        today = datetime.now().date()
        days_diff = (exp_date - today).days
        
        return 0 <= days_diff <= days_threshold
    except:
        return False

# Register template filter AFTER function is defined
@app.template_filter('is_expiring_soon')
def template_is_expiring_soon(exp_date_str):
    """Template filter to check if expiration date is soon"""
    return is_expiring_soon(exp_date_str, 7)

def normalize_expiration_date(date_str):
    """Normalize expiration date to YYYY-MM-DD format"""
    if not date_str or not isinstance(date_str, str):
        return None
    
    date_str = date_str.strip()
    if not date_str:
        return None
    
    try:
        # If already in YYYY-MM-DD format, return as-is
        if len(date_str) == 10 and date_str.count('-') == 2:
            datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        
        # Try parsing ISO format
        if 'T' in date_str:
            parsed = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            parsed = datetime.fromisoformat(date_str)
        
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, TypeError, AttributeError):
        return None

def normalize_pantry_item(item):
    """Normalize a pantry item to ensure consistent format"""
    if item is None:
        # Return a default item if None is passed
        return {
            'id': str(uuid.uuid4()),
            'name': '',
            'quantity': '1',
            'expirationDate': None,
            'addedDate': datetime.now().isoformat()
        }
    
    if isinstance(item, dict):
        # Create a copy to avoid modifying the original
        normalized_item = item.copy()
        # Normalize expiration date format
        if 'expirationDate' in normalized_item and normalized_item['expirationDate']:
            normalized_date = normalize_expiration_date(normalized_item['expirationDate'])
            normalized_item['expirationDate'] = normalized_date
        # Ensure all required fields exist
        if 'id' not in normalized_item:
            normalized_item['id'] = str(uuid.uuid4())
        if 'quantity' not in normalized_item:
            normalized_item['quantity'] = '1'
        if 'addedDate' not in normalized_item:
            normalized_item['addedDate'] = datetime.now().isoformat()
        # Ensure name exists and is a string
        if 'name' not in normalized_item or normalized_item['name'] is None:
            normalized_item['name'] = ''
        else:
            normalized_item['name'] = str(normalized_item['name']).strip()
        
        # Ensure quantity is always a valid string
        if 'quantity' not in normalized_item or normalized_item['quantity'] is None:
            normalized_item['quantity'] = '1'
        else:
            normalized_item['quantity'] = str(normalized_item['quantity']).strip() or '1'
        
        # Ensure expirationDate is None or valid string
        if 'expirationDate' in normalized_item:
            if normalized_item['expirationDate'] == '' or normalized_item['expirationDate'] is None:
                normalized_item['expirationDate'] = None
            else:
                normalized_item['expirationDate'] = str(normalized_item['expirationDate']).strip()
        
        # Validate that name is not empty after normalization
        if not normalized_item.get('name'):
            normalized_item['name'] = 'Unnamed Item'
        return normalized_item
    else:
        # Convert string to dict format
        item_str = str(item).strip() if item else ''
        return {
            'id': str(uuid.uuid4()),
            'name': item_str,
            'quantity': '1',
            'expirationDate': None,
            'addedDate': datetime.now().isoformat()
        }

def get_user_pantry(user_id, use_cache=True):
    """Get user's pantry items with caching"""
    global _pantry_cache, _pantry_cache_timestamp
    
    # Check pantry cache first
    if use_cache:
        import time
        current_time = time.time()
        if user_id in _pantry_cache:
            cache_age = current_time - _pantry_cache_timestamp.get(user_id, 0)
            if cache_age < _users_cache_ttl:
                if VERBOSE_LOGGING:
                    print(f"Using cached pantry for user {user_id} (age: {cache_age:.2f}s)")
                return _pantry_cache[user_id].copy()
    
    users = load_users(use_cache=use_cache)
    if user_id in users:
        pantry = users[user_id].get('pantry', [])
        # Ensure pantry is a list
        if not isinstance(pantry, list):
            print(f"Warning: Pantry for user {user_id} is not a list, resetting to empty list")
            pantry = []
        # Normalize all items to ensure consistent format
        normalized_pantry = []
        for item in pantry:
            try:
                normalized_item = None
                if isinstance(item, dict):
                    normalized_item = normalize_pantry_item(item.copy())
                elif item is not None:
                    normalized_item = normalize_pantry_item(item)
                
                # Only add items with valid names (after normalization, name should never be empty)
                if normalized_item and normalized_item.get('name') and normalized_item.get('name').strip():
                    name_str = str(normalized_item.get('name', '')).strip()
                    if name_str and name_str != 'Unnamed Item':  # Skip placeholder names
                        normalized_pantry.append(normalized_item)
            except Exception as e:
                print(f"Warning: Failed to normalize item {item}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Cache the normalized pantry
        import time
        _pantry_cache[user_id] = normalized_pantry.copy()
        _pantry_cache_timestamp[user_id] = time.time()
        
        if VERBOSE_LOGGING:
            print(f"Retrieved pantry for user {user_id}: {len(normalized_pantry)} items")
        return normalized_pantry
    if VERBOSE_LOGGING:
        print(f"Warning: User {user_id} not found in users database")
    return []

def update_user_pantry(user_id, pantry_items):
    """Update user's pantry items"""
    if VERBOSE_LOGGING:
        print(f"\n{'='*60}")
        print(f"🔄 UPDATE USER PANTRY")
        print(f"{'='*60}")
        print(f"User ID: {user_id}")
        print(f"Items to save: {len(pantry_items)}")
    
    users = load_users(use_cache=False)  # Don't use cache when updating
    if VERBOSE_LOGGING:
        print(f"Total users in database: {len(users)}")
    
    if user_id not in users:
        users[user_id] = {'pantry': []}
    
    if VERBOSE_LOGGING:
        print(f"✅ User {user_id} found in database")
        print(f"   Username: {users[user_id].get('username', 'unknown')}")
    
    # Ensure pantry_items is a list
    if not isinstance(pantry_items, list):
        print(f"Warning: pantry_items is not a list, converting to list")
        pantry_items = []
    
    # Normalize all items before saving
    normalized_items = []
    for item in pantry_items:
        try:
            if isinstance(item, dict):
                normalized_items.append(normalize_pantry_item(item.copy()))
            elif item is not None:
                normalized_items.append(normalize_pantry_item(item))
        except Exception as e:
            print(f"Warning: Failed to normalize item {item}: {e}")
            continue
    
    users[user_id]['pantry'] = normalized_items
    if VERBOSE_LOGGING:
        print(f"💾 Saving {len(users)} users to {USERS_FILE}...")
    save_users(users)
    if VERBOSE_LOGGING:
        print(f"✅ Updated pantry for user {user_id}: {len(normalized_items)} items saved to {USERS_FILE}")
    
    # Update pantry cache directly (no need to verify by loading again)
    import time
    _pantry_cache[user_id] = normalized_items.copy()
    _pantry_cache_timestamp[user_id] = time.time()
    
    # Skip verification in production for performance (only verify in verbose mode)
    if VERBOSE_LOGGING:
        print(f"🔍 Verifying save...")
        verify_users = load_users(use_cache=False)
        if user_id in verify_users:
            verify_pantry = verify_users[user_id].get('pantry', [])
            if len(verify_pantry) == len(normalized_items):
                print(f"✅ Verified: Pantry update saved correctly ({len(verify_pantry)} items)")
            else:
                print(f"⚠️ Warning: Saved {len(normalized_items)} items but file contains {len(verify_pantry)} items")
                print(f"   Expected items: {[item.get('name', 'unknown') if isinstance(item, dict) else str(item) for item in normalized_items[:5]]}")
                print(f"   Saved items: {[item.get('name', 'unknown') if isinstance(item, dict) else str(item) for item in verify_pantry[:5]]}")
        else:
            print(f"❌ Error: User {user_id} not found after save!")
        print(f"{'='*60}\n")

 

# Authentication routes for web
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        print(f"Login attempt - Username: {username}, Password provided: {'Yes' if password else 'No'}")
        
        if not username or not password:
            flash("Please provide both username and password", "danger")
            return render_template("login.html")
        
        user_data, error = authenticate_user(username, password, 'web')
        if user_data:
            session.permanent = True  # Make session permanent
            session['user_id'] = user_data['id']
            session['username'] = user_data['username']
            print(f"Login successful - User ID: {user_data['id']}, Username: {user_data['username']}")
            print(f"Session after login: user_id={session.get('user_id')}, username={session.get('username')}")
            flash(f"Welcome back, {user_data['username']}!", "success")
            return redirect(url_for("index"))
        else:
            print(f"Login failed - Error: {error}")
            flash(error, "danger")
    
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        
        print(f"Signup attempt - Username: {username}, Email: {email}")
        
        if password != confirm_password:
            flash("Passwords do not match", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters", "danger")
        else:
            user_id, error = create_user(username, email, password, 'web')
            if user_id:
                session.permanent = True  # Make session permanent
                session['user_id'] = user_id
                session['username'] = username
                print(f"Signup successful - User ID: {user_id}, Username: {username}")
                print(f"Session after signup: user_id={session.get('user_id')}, username={session.get('username')}")
                flash(f"Account created successfully! Welcome, {username}!", "success")
                return redirect(url_for("index"))
            else:
                print(f"Signup failed - Error: {error}")
                flash(error, "danger")
    
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out", "info")
    return redirect(url_for("login"))

def update_user_password(user_id, new_password):
    """Update user's password"""
    users = load_users()
    if user_id in users:
        users[user_id]['password_hash'] = hash_password(new_password)
        save_users(users)
        return True, None
    return False, "User not found"

@app.route("/profile", methods=["GET", "POST"])
def profile():
    """User profile page with password change functionality"""
    # Check if user is logged in
    if 'user_id' not in session:
        flash("Please log in to access your profile", "warning")
        return redirect(url_for("login"))
    
    user_id = session['user_id']
    users = load_users()
    
    if user_id not in users:
        flash("User not found", "danger")
        session.clear()
        return redirect(url_for("login"))
    
    user_data = users[user_id]
    
    # Handle POST request (password change)
    if request.method == "POST":
        new_password = request.form.get("new_password")
        confirm_password = request.form.get("confirm_password")
        
        if not new_password or len(new_password) < 6:
            flash("Password must be at least 6 characters long", "danger")
        elif new_password != confirm_password:
            flash("Passwords do not match", "danger")
        else:
            success, error = update_user_password(user_id, new_password)
            if success:
                flash("Password updated successfully!", "success")
            else:
                flash(error or "Failed to update password", "danger")
    
    # Render profile page (GET request or after POST)
    return render_template("profile.html", 
                         username=user_data.get('username'),
                         email=user_data.get('email'))

# Home page (pantry list)
@app.route("/")
def index():
    # Check if user is logged in
    if 'user_id' in session:
        # get_user_pantry already normalizes items, so no need to normalize again
        user_pantry = get_user_pantry(session['user_id'])
        # Ensure user_pantry is a list
        if not isinstance(user_pantry, list):
            user_pantry = []
        
        if VERBOSE_LOGGING:
            print(f"DEBUG: Rendering index with {len(user_pantry)} items for user {session.get('username')}")
            print(f"DEBUG: Items: {[item.get('name', 'NO_NAME') for item in user_pantry[:3]]}")
        
        # Ensure items is always a list, never None
        items_to_render = user_pantry if user_pantry else []
        return render_template("index.html", items=items_to_render, username=session.get('username'))
    else:
        # Use session-based pantry for anonymous users (consistent with add_items and delete_item)
        if 'web_pantry' not in session:
            session['web_pantry'] = []
        # Ensure web_pantry is a list
        web_pantry = session.get('web_pantry', [])
        if not isinstance(web_pantry, list):
            web_pantry = []
        
        # Normalize anonymous pantry items (only once)
        normalized_web_pantry = []
        for item in web_pantry:
            try:
                normalized_item = None
                if isinstance(item, dict):
                    normalized_item = normalize_pantry_item(item.copy())
                elif item is not None:
                    normalized_item = normalize_pantry_item(item)
                
                # Only add items with valid names
                if normalized_item and normalized_item.get('name'):
                    name_str = str(normalized_item.get('name', '')).strip()
                    if name_str:
                        normalized_item['name'] = name_str
                        normalized_web_pantry.append(normalized_item)
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"Warning: Failed to normalize item {item}: {e}")
                    import traceback
                    traceback.print_exc()
                continue
        
        if VERBOSE_LOGGING:
            print(f"DEBUG: Rendering index with {len(normalized_web_pantry)} items for anonymous user")
            print(f"DEBUG: Items: {[item.get('name', 'NO_NAME') for item in normalized_web_pantry[:3]]}")
        
        # Ensure items is always a list, never None
        items_to_render = normalized_web_pantry if normalized_web_pantry else []
        return render_template("index.html", items=items_to_render, username=None)

# Add items
@app.route("/add", methods=["POST"])
def add_items():
    item = request.form.get("item")
    # Get quantity, default to "1" if empty or None
    quantity_raw = request.form.get("quantity", "").strip()
    quantity = quantity_raw if quantity_raw else "1"
    # Get expiration date, return None if empty
    expiration_date_raw = request.form.get("expiration_date", "").strip()
    expiration_date = expiration_date_raw if expiration_date_raw else None
    
    # Check if this is an AJAX request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    if not item or not item.strip():
        flash("Please enter an item name", "danger")
        if is_ajax:
            return jsonify({'success': False, 'error': 'Item name is required'}), 400
        return redirect(url_for("index"))
    
    # Sanitize and validate input
    item = item.strip()
    if len(item) > 100:  # Prevent extremely long items
        flash("Item name too long. Please keep it under 100 characters.", "danger")
        if is_ajax:
            return jsonify({'success': False, 'error': 'Item name too long'}), 400
        return redirect(url_for("index"))
    
    # Validate and normalize expiration date
    normalized_expiration = None
    if expiration_date:
        normalized_expiration = normalize_expiration_date(expiration_date)
        if normalized_expiration is None:
            flash(f"Warning: Invalid expiration date format '{expiration_date}'. Item added without expiration date.", "warning")
    
    # Ensure quantity is a string
    if not quantity or quantity == "":
        quantity = "1"
    
    if 'user_id' in session:
        # Add to user's pantry
        user_id = session['user_id']
        print(f"Adding item '{item}' to pantry for user {user_id}")
        user_pantry = get_user_pantry(user_id)
        print(f"Current pantry before add: {user_pantry} (type: {type(user_pantry)})")
        
        # Ensure pantry is a list
        if not isinstance(user_pantry, list):
            print(f"WARNING: Pantry is not a list, resetting to empty list")
            user_pantry = []
        
        # Convert to list of dicts if needed, check for duplicates
        pantry_list = []
        item_exists = False
        item_normalized = item.strip().lower()  # Normalize for comparison
        for pantry_item in user_pantry:
            if isinstance(pantry_item, dict):
                pantry_list.append(pantry_item)
                pantry_name = pantry_item.get('name', '').strip().lower() if pantry_item.get('name') else ''
                if pantry_name == item_normalized:
                    item_exists = True
            else:
                pantry_str = str(pantry_item).strip().lower() if pantry_item else ''
                pantry_list.append({
                    'id': str(uuid.uuid4()),
                    'name': str(pantry_item).strip() if pantry_item else '',
                    'quantity': '1',
                    'expirationDate': None,
                    'addedDate': datetime.now().isoformat()
                })
                if pantry_str == item_normalized:
                    item_exists = True
        
        if not item_exists:
            # Add new item with quantity and expiration date
            new_item = {
                'id': str(uuid.uuid4()),
                'name': item,
                'quantity': quantity,
                'expirationDate': normalized_expiration,
                'addedDate': datetime.now().isoformat()
            }
            pantry_list.append(new_item)
            update_user_pantry(user_id, pantry_list)
            if normalized_expiration:
                flash(f"{item} added to pantry.", "success")
            else:
                flash(f"{item} added to pantry.", "success")
        else:
            flash(f"{item} is already in your pantry.", "warning")
    else:
        # Add to anonymous web pantry (stored in session for persistence)
        # CRITICAL: Mark session as permanent for anonymous users to persist pantry items
        session.permanent = True
        
        if 'web_pantry' not in session:
            session['web_pantry'] = []
        
        # Check for duplicates (case-insensitive, normalized)
        item_exists = False
        item_normalized = item.strip().lower()  # Normalize for comparison
        for pantry_item in session['web_pantry']:
            if isinstance(pantry_item, dict):
                pantry_name = pantry_item.get('name', '').strip().lower() if pantry_item.get('name') else ''
                if pantry_name == item_normalized:
                    item_exists = True
                    break
            else:
                pantry_str = str(pantry_item).strip().lower() if pantry_item else ''
                if pantry_str == item_normalized:
                    item_exists = True
                    break
        
        if not item_exists:
            # Add new item as dictionary with quantity and expiration date
            new_item = {
                'id': str(uuid.uuid4()),
                'name': item,
                'quantity': quantity,
                'expirationDate': normalized_expiration,
                'addedDate': datetime.now().isoformat()
            }
            session['web_pantry'].append(new_item)
            # Mark session as modified to ensure it's saved
            session.modified = True
            if normalized_expiration:
                flash(f"{item} added to pantry.", "success")
            else:
                flash(f"{item} added to pantry.", "success")
        else:
            flash(f"{item} is already in your pantry.", "warning")
    
    # Return JSON response for AJAX requests, otherwise redirect
    # IMPORTANT: Ensure the item is saved before responding
    if is_ajax:
        return jsonify({'success': True, 'message': 'Item added successfully'}), 200
    return redirect(url_for("index"))

# Delete item
@app.route("/delete/<item_name>")
def delete_item(item_name):
    from urllib.parse import unquote
    # Decode URL-encoded item name and strip whitespace
    item_name = unquote(item_name).strip()
    
    if VERBOSE_LOGGING:
        print(f"DEBUG: Attempting to delete item: '{item_name}'")
    
    if 'user_id' in session:
        # Remove from user's pantry
        user_pantry = get_user_pantry(session['user_id'])
        if VERBOSE_LOGGING:
            print(f"DEBUG: User pantry has {len(user_pantry) if isinstance(user_pantry, list) else 0} items")
        
        # Ensure pantry is a list
        if not isinstance(user_pantry, list):
            user_pantry = []
        
        # Convert to list of dicts if needed
        pantry_list = []
        item_found = False
        for pantry_item in user_pantry:
            if isinstance(pantry_item, dict):
                pantry_name = pantry_item.get('name', '').strip() if pantry_item.get('name') else ''
                # Compare with stripped names (case-insensitive)
                if pantry_name and pantry_name.lower() == item_name.lower():
                    item_found = True
                    print(f"DEBUG: Found matching item: '{pantry_name}' == '{item_name}'")
                else:
                    pantry_list.append(pantry_item)
            else:
                pantry_str = str(pantry_item).strip() if pantry_item else ''
                if pantry_str and pantry_str.lower() == item_name.lower():
                    item_found = True
                    print(f"DEBUG: Found matching item (string): '{pantry_str}' == '{item_name}'")
                elif pantry_str:
                    # Convert old string format to dict format
                    pantry_list.append({
                        'id': str(uuid.uuid4()),
                        'name': pantry_str,
                        'quantity': '1',
                        'expirationDate': None,
                        'addedDate': datetime.now().isoformat()
                    })
        
        if item_found:
            update_user_pantry(session['user_id'], pantry_list)
            # Check if AJAX request
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': True, 'message': f'{item_name} removed from pantry.'}), 200
            flash(f"{item_name} removed from pantry.", "info")
        else:
            # Debug: print available item names
            if VERBOSE_LOGGING:
                available_names = []
                for item in user_pantry:
                    if isinstance(item, dict):
                        name = item.get('name', '').strip()
                        if name:
                            available_names.append(name)
                    else:
                        name = str(item).strip()
                        if name:
                            available_names.append(name)
                print(f"DEBUG: Item '{item_name}' not found. Available items: {available_names}")
            # Check if AJAX request
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'error': f"Item '{item_name}' not found in pantry."}), 404
            flash(f"Item '{item_name}' not found in pantry.", "warning")
    else:
        # Remove from anonymous web pantry (stored in session)
        # CRITICAL: Mark session as permanent for anonymous users
        session.permanent = True
        
        if 'web_pantry' not in session:
            session['web_pantry'] = []
        
        if VERBOSE_LOGGING:
            print(f"DEBUG: Anonymous pantry has {len(session['web_pantry'])} items")
        
        # Convert to list of dicts if needed
        pantry_list = []
        item_found = False
        item_name_normalized = item_name.strip().lower()  # Normalize once for comparison
        for pantry_item in session['web_pantry']:
            if isinstance(pantry_item, dict):
                pantry_name = pantry_item.get('name', '').strip() if pantry_item.get('name') else ''
                # Compare with stripped names (case-insensitive)
                if pantry_name and pantry_name.lower() == item_name_normalized:
                    item_found = True
                    print(f"DEBUG: Found matching item: '{pantry_name}' == '{item_name}'")
                else:
                    pantry_list.append(pantry_item)
            else:
                pantry_str = str(pantry_item).strip() if pantry_item else ''
                if pantry_str and pantry_str.lower() == item_name_normalized:
                    item_found = True
                    print(f"DEBUG: Found matching item (string): '{pantry_str}' == '{item_name}'")
                elif pantry_str:
                    # Convert old string format to dict format
                    pantry_list.append({
                        'id': str(uuid.uuid4()),
                        'name': pantry_str,
                        'quantity': '1',
                        'expirationDate': None,
                        'addedDate': datetime.now().isoformat()
                    })
        
        if item_found:
            session['web_pantry'] = pantry_list
            # Mark session as modified to ensure it's saved
            session.modified = True
            # Check if AJAX request
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': True, 'message': f'{item_name} removed from pantry.'}), 200
            flash(f"{item_name} removed from pantry.", "info")
        else:
            # Debug: print available item names
            if VERBOSE_LOGGING:
                available_names = []
                for item in session['web_pantry']:
                    if isinstance(item, dict):
                        name = item.get('name', '').strip()
                        if name:
                            available_names.append(name)
                    else:
                        name = str(item).strip()
                        if name:
                            available_names.append(name)
                print(f"DEBUG: Item '{item_name}' not found. Available items: {available_names}")
            # Check if AJAX request
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'error': f"Item '{item_name}' not found in pantry."}), 404
            flash(f"Item '{item_name}' not found in pantry.", "warning")
    
    # Check if AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'success': True, 'message': 'Item deleted successfully.'}), 200
    return redirect(url_for("index"))

def get_expiring_items(pantry_items, expiring_days=None):
    """Filter pantry items that are expiring within specified days"""
    if not pantry_items:
        return []
    
    if expiring_days is None:
        return pantry_items  # Return all items if no filter specified
    
    today = datetime.now().date()
    expiring_items = []
    
    for item in pantry_items:
        if isinstance(item, dict):
            exp_date_str = item.get('expirationDate')
            if exp_date_str:
                try:
                    # Normalize date format
                    normalized_date = normalize_expiration_date(exp_date_str)
                    if normalized_date:
                        exp_date = datetime.strptime(normalized_date, "%Y-%m-%d").date()
                        days_left = (exp_date - today).days
                        if 0 <= days_left <= expiring_days:
                            expiring_items.append(item)
                except (ValueError, TypeError):
                    pass
        # If item has no expiration date and we're filtering, skip it
        # (only include items with expiration dates when filtering)
    
    return expiring_items if expiring_items else pantry_items  # Fallback to all if none match

# Suggest recipes based on pantry
@app.route("/suggest")
def suggest_recipe():
    # CRITICAL: Check for existing recipes FIRST before any processing
    # This prevents regeneration when user navigates back from recipe detail page
    existing_recipes = session.get('current_recipes', [])
    if existing_recipes and isinstance(existing_recipes, list) and len(existing_recipes) > 0:
        # Use existing recipes - get pantry items for display only
        if VERBOSE_LOGGING:
            print(f"DEBUG: Using existing {len(existing_recipes)} recipes from session")
        
        # Refresh session to prevent expiration
        session['current_recipes'] = existing_recipes
        
        # Get pantry items for display (but don't regenerate recipes)
        if 'user_id' in session:
            current_pantry = get_user_pantry(session['user_id'])
        else:
            if 'web_pantry' not in session:
                session['web_pantry'] = []
            current_pantry = session['web_pantry']
        
        # Convert pantry to string list for template compatibility
        pantry_items_list = []
        # Also keep full pantry data for expiration checking
        pantry_items_full = []
        for pantry_item in current_pantry:
            if isinstance(pantry_item, dict):
                name = pantry_item.get('name', '')
                if name:
                    pantry_items_list.append(name)
                    pantry_items_full.append(pantry_item)  # Keep full item data
            else:
                if pantry_item:
                    pantry_items_list.append(str(pantry_item))
                    pantry_items_full.append({'name': str(pantry_item), 'expirationDate': None})
        
        flash("Showing your current recipes. Click 'Generate New Recipes' for fresh ideas!", "info")
        return render_template("suggest_recipe.html", recipes=existing_recipes, pantry_items=pantry_items_list, pantry_items_full=pantry_items_full)
    
    # Get expiring_days parameter from query string (only if generating new recipes)
    expiring_days = request.args.get('expiring_days', type=int)
    
    # Get appropriate pantry based on login status
    if 'user_id' in session:
        current_pantry = get_user_pantry(session['user_id'])
    else:
        # Use session-based pantry for anonymous users
        if 'web_pantry' not in session:
            session['web_pantry'] = []
        current_pantry = session['web_pantry']
    
    # Filter by expiration if requested
    if expiring_days is not None:
        current_pantry = get_expiring_items(current_pantry, expiring_days)
        if not current_pantry:
            flash(f"No items expiring within {expiring_days} days. Showing all items.", "info")
            # Fallback to all items
            if 'user_id' in session:
                current_pantry = get_user_pantry(session['user_id'])
            else:
                current_pantry = session.get('web_pantry', [])
    
    # Convert pantry to string list for template compatibility
    pantry_items_list = []
    # Also keep full pantry data for expiration checking
    pantry_items_full = []
    for pantry_item in current_pantry:
        if isinstance(pantry_item, dict):
            name = pantry_item.get('name', '')
            if name:
                pantry_items_list.append(name)
                pantry_items_full.append(pantry_item)  # Keep full item data
        else:
            if pantry_item:
                pantry_items_list.append(str(pantry_item))
                pantry_items_full.append({'name': str(pantry_item), 'expirationDate': None})
    
    # Check if pantry is empty (handle both None and empty list)
    if not pantry_items_list or len(pantry_items_list) == 0:
        flash("Your pantry is empty. Add items first.", "warning")
        return redirect(url_for("index"))

    # Generate AI-powered recipes based on pantry items (only if no existing recipes)
    pantry_items = ", ".join(pantry_items_list)
    pantry = pantry_items_list  # Use string list for compatibility with existing code
    prompt = f"""Based on the following pantry items: {pantry_items}

Generate 3 creative and practical recipes that use AT LEAST 50% of these pantry ingredients. For each recipe, provide:
1. Recipe name
2. List of ingredients (prioritizing pantry items - at least half must be from pantry)
3. Step-by-step cooking instructions
4. Estimated cooking time
5. Health assessment (Healthy/Moderately Healthy/Unhealthy)
6. Health explanation (brief reason for the health rating)

CRITICAL REQUIREMENT: Each recipe MUST use at least 50% of ingredients from the pantry list above.

Format as JSON:
{{
    "recipes": [
        {{
            "name": "Recipe Name",
            "ingredients": ["pantry_item1", "pantry_item2", "additional_item"],
            "instructions": ["step1", "step2", "step3"],
            "cooking_time": "X minutes",
            "difficulty": "Easy",
            "health_rating": "Healthy",
            "health_explanation": "This dish is healthy because it contains fresh vegetables, lean proteins, and minimal processed ingredients."
        }},
        {{
            "name": "Recipe Name 2",
            "ingredients": ["pantry_item1", "pantry_item3", "additional_item"],
            "instructions": ["step1", "step2", "step3"],
            "cooking_time": "X minutes",
            "difficulty": "Medium",
            "health_rating": "Moderately Healthy",
            "health_explanation": "This dish is moderately healthy with some nutritious ingredients but may contain higher sodium or fat content."
        }},
        {{
            "name": "Recipe Name 3",
            "ingredients": ["pantry_item2", "pantry_item3", "additional_item"],
            "instructions": ["step1", "step2", "step3"],
            "cooking_time": "X minutes",
            "difficulty": "Hard",
            "health_rating": "Healthy",
            "health_explanation": "This dish is healthy as it focuses on whole foods and balanced nutrition."
        }}
    ]
}}"""

    try:
        # Check if client is properly initialized
        if not client:
            raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
            
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a creative chef and recipe developer. Create practical, delicious recipes using available ingredients. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        import json
        recipe_text = response.choices[0].message.content
        
        # Try to parse JSON response
        try:
            # Clean the response text - remove any markdown formatting
            cleaned_text = recipe_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text.replace("```", "").strip()
            
            recipe_data = json.loads(cleaned_text)
            recipes = recipe_data.get("recipes", [])
            
            # Validate that we got recipes
            if not recipes:
                raise ValueError("No recipes found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            # If JSON parsing fails, create a fallback recipe
            flash(f"Using fallback recipe generation: {str(e)}", "info")
            # Ensure pantry is a list of strings for fallback
            fallback_pantry = pantry_items_list[:5] if pantry_items_list else ["ingredients"]
            recipes = [{
                "name": "Pantry Surprise",
                "ingredients": fallback_pantry,
                "instructions": [
                    "Combine your pantry items creatively",
                    "Season to taste",
                    "Cook until done"
                ],
                "cooking_time": "20 minutes",
                "difficulty": "Easy",
                "health_rating": "Healthy",
                "health_explanation": "This recipe uses fresh ingredients from your pantry, making it a healthy choice."
            }]
            
        # Validate that recipes use 50%+ pantry ingredients
        validated_recipes = []
        for recipe in recipes:
            ingredients = recipe.get('ingredients', [])
            if not ingredients:
                continue  # Skip recipes without ingredients
            # pantry_items_list is already a list of strings, use that for comparison
            pantry_count = sum(1 for ingredient in ingredients if any(pantry_item.lower() in ingredient.lower() for pantry_item in pantry_items_list))
            total_ingredients = len(ingredients)
            pantry_percentage = (pantry_count / total_ingredients) * 100 if total_ingredients > 0 else 0
            
            if pantry_percentage >= 50:
                validated_recipes.append(recipe)
            else:
                # If recipe doesn't meet 50% requirement, try to adjust it
                recipe_name = recipe.get('name', 'Unknown Recipe')
                flash(f"Recipe '{recipe_name}' only uses {pantry_percentage:.0f}% pantry ingredients. Adjusting...", "warning")
                # Add more pantry ingredients to meet requirement
                needed_pantry = max(1, (total_ingredients // 2) - pantry_count)
                if pantry_items_list and len(pantry_items_list) > 0:
                    additional_pantry = pantry_items_list[:min(needed_pantry, len(pantry_items_list))]
                    if 'ingredients' not in recipe:
                        recipe['ingredients'] = []
                    recipe['ingredients'].extend(additional_pantry)
                validated_recipes.append(recipe)
        
        recipes = validated_recipes
        
        flash(f"Generated {len(recipes)} recipe(s) using at least 50% pantry ingredients!", "success")
        
        # Store recipes in session for nutrition info lookup
        session['current_recipes'] = recipes
        
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR generating recipes: {error_msg}")
        import traceback
        traceback.print_exc()
        
        flash(f"AI unavailable: {error_msg}. Using fallback recipes.", "warning")
        # Fallback to simple recipe suggestions
        recipes = []
        
        # Use pantry_items_list if it has items, otherwise use current_pantry
        # pantry_items_list is always defined (set before try block), but might be empty
        pantry_for_fallback = pantry_items_list if (pantry_items_list and len(pantry_items_list) > 0) else (current_pantry if current_pantry else [])
        
        if pantry_for_fallback and len(pantry_for_fallback) > 0:
            # Convert pantry to string list if needed
            pantry_strings = []
            for p in pantry_for_fallback:
                if isinstance(p, dict):
                    name = p.get('name', '')
                    if name:
                        pantry_strings.append(name)
                else:
                    pantry_str = str(p).strip()
                    if pantry_str:
                        pantry_strings.append(pantry_str)
            pantry_strings = [p for p in pantry_strings if p]  # Remove empty strings
            
            # Create recipes based on available pantry items
            if len(pantry_strings) >= 1:
                recipes.append({
                    "name": f"Simple {pantry_strings[0].title()} Dish",
                    "ingredients": pantry_strings[:3] + ["salt", "pepper", "oil"],
                    "instructions": [
                        "Prepare your main ingredients",
                        "Season with salt and pepper",
                        "Cook in oil until tender",
                        "Serve hot"
                    ],
                    "cooking_time": "15 minutes",
                    "difficulty": "Easy",
                    "health_rating": "Healthy",
                    "health_explanation": "Simple cooking with fresh ingredients."
                })
            
            if len(pantry_strings) >= 2:
                recipes.append({
                    "name": f"Quick {pantry_strings[1].title()} Recipe",
                    "ingredients": pantry_strings[1:4] + ["garlic", "herbs"],
                    "instructions": [
                        "Chop ingredients finely",
                        "Sauté with garlic",
                        "Add herbs for flavor",
                        "Cook until golden"
                    ],
                    "cooking_time": "10 minutes",
                    "difficulty": "Easy",
                    "health_rating": "Healthy",
                    "health_explanation": "Quick and nutritious meal."
                })
            
            if len(pantry_strings) >= 3:
                recipes.append({
                    "name": "Pantry Fusion",
                    "ingredients": pantry_strings[:5] + ["spices"],
                    "instructions": [
                        "Mix all ingredients together",
                        "Add your favorite spices",
                        "Cook until well combined",
                        "Let flavors meld"
                    ],
                    "cooking_time": "25 minutes",
                    "difficulty": "Medium",
                    "health_rating": "Moderately Healthy",
                    "health_explanation": "Creative combination of available ingredients."
                })
        else:
            # If no pantry items, create a generic recipe
            recipes.append({
                "name": "Basic Pantry Meal",
                "ingredients": pantry_items_list[:3] + ["salt", "pepper", "oil"] if pantry_items_list else ["ingredients", "salt", "pepper"],
                "instructions": [
                    "Gather your pantry items",
                    "Season to taste",
                    "Cook until done",
                    "Serve hot"
                ],
                "cooking_time": "20 minutes",
                "difficulty": "Easy",
                "health_rating": "Healthy",
                "health_explanation": "A simple meal using available ingredients."
            })
        
        # Ensure we have at least one recipe
        if not recipes:
            recipes = [{
                "name": "Pantry Surprise",
                "ingredients": pantry_items_list[:5] if pantry_items_list else ["ingredients"],
                "instructions": [
                    "Combine your pantry items creatively",
                    "Season to taste",
                    "Cook until done"
                ],
                "cooking_time": "20 minutes",
                "difficulty": "Easy",
                "health_rating": "Healthy",
                "health_explanation": "This recipe uses fresh ingredients from your pantry."
            }]
        
        # Store fallback recipes in session
        session['current_recipes'] = recipes

    return render_template("suggest_recipe.html", recipes=recipes, pantry_items=pantry_items_list)

# Generate new recipes (force refresh)
@app.route("/generate_new_recipes")
def generate_new_recipes():
    # Get appropriate pantry based on user authentication
    if 'user_id' in session:
        pantry_to_check = get_user_pantry(session['user_id'])
    else:
        # Use session-based pantry for anonymous users
        if 'web_pantry' not in session:
            session['web_pantry'] = []
        pantry_to_check = session['web_pantry']
    
    # Check if pantry is empty (handle both None and empty list)
    if not pantry_to_check or len(pantry_to_check) == 0:
        flash("Your pantry is empty. Add items first.", "warning")
        return redirect(url_for("index"))

    # Clear existing recipes and generate new ones
    session.pop('current_recipes', None)
    return redirect(url_for("suggest_recipe"))

# Upload photo route (for food analysis)
@app.route("/upload_photo", methods=["POST"])
def upload_photo():
    photo = request.files.get("photo")
    if not photo or photo.filename == '':
        flash("No photo uploaded.", "danger")
        return redirect(url_for("index"))
    
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    file_ext = os.path.splitext(photo.filename)[1].lower() if photo.filename else ''
    if file_ext not in allowed_extensions:
        flash(f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}", "danger")
        return redirect(url_for("index"))
    
    # Read file content once and keep it in memory
    photo.seek(0)
    img_bytes = photo.read()
    
    # Validate file size (max 10MB)
    file_size = len(img_bytes)
    if file_size > 10 * 1024 * 1024:  # 10MB
        flash("File too large. Maximum size is 10MB.", "danger")
        return redirect(url_for("index"))
    
    if file_size == 0:
        flash("File is empty or could not be read.", "danger")
        return redirect(url_for("index"))
    
    # Save photo to uploads folder (optional, for debugging)
    # In serverless, use /tmp directory which is writable
    if IS_VERCEL:
        upload_folder = '/tmp/uploads'
    else:
        upload_folder = os.path.join(os.path.dirname(__file__), "uploads")
    
    # Only save if folder exists and is writable (skip on Vercel if /tmp is not available)
    try:
        os.makedirs(upload_folder, exist_ok=True)
        safe_filename = photo.filename or 'upload.jpg'
        safe_filename = os.path.basename(safe_filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(safe_filename)
        safe_filename = f"{name}_{timestamp}{ext}"
        photo_path = os.path.join(upload_folder, safe_filename)
        with open(photo_path, 'wb') as f:
            f.write(img_bytes)
    except Exception as e:
        # Log but don't fail - saving is optional
        if VERBOSE_LOGGING:
            print(f"Warning: Could not save photo to disk: {str(e)}")

    # Send image to OpenAI vision API
    import base64
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    prompt = """Analyze this photo and identify all food items with their quantities. 
For each food item, detect:
1. The quantity (e.g., "2 bottles", "three cans", "1 loaf", "5 slices")
2. The food item name (generic name only, no brand names)

Return the results in this exact JSON format:
{
  "items": [
    {"name": "milk", "quantity": "2 bottles"},
    {"name": "soup", "quantity": "3 cans"},
    {"name": "bread", "quantity": "1 loaf"}
  ]
}

If you cannot determine a quantity, use "1" as the default. Return ONLY valid JSON, no other text."""
    
    try:
        if not client:
            raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a food recognition assistant. Analyze images and identify food items with their quantities. Return results in JSON format with item names and quantities."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]}
            ],
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        food_response = response.choices[0].message.content
        
        # Parse JSON response
        import json
        try:
            food_data = json.loads(food_response)
            detected_items_data = food_data.get('items', [])
            
            # Convert to list of PantryItem objects
            pantry_items = []
            for item in detected_items_data:
                name = item.get('name', '').strip()
                quantity = item.get('quantity', '1').strip()
                if name:
                    pantry_items.append({
                        'id': str(uuid.uuid4()),
                        'name': name,
                        'quantity': quantity,
                        'expirationDate': None,
                        'addedDate': datetime.now().isoformat()
                    })
        except json.JSONDecodeError:
            # Fallback: try to parse as comma-separated list (old format)
            detected_items = [f.strip() for f in food_response.split(",") if f.strip()]
            pantry_items = []
            for item in detected_items:
                pantry_items.append({
                    'id': str(uuid.uuid4()),
                    'name': item,
                    'quantity': '1',
                    'expirationDate': None,
                    'addedDate': datetime.now().isoformat()
                })
        
        # Add to appropriate pantry based on user authentication
        if 'user_id' in session:
            # Add to user's pantry
            user_pantry = get_user_pantry(session['user_id'])
            # Convert to list of dicts if needed
            pantry_list = []
            for item in user_pantry:
                if isinstance(item, dict):
                    pantry_list.append(item)
                else:
                    pantry_list.append({
                        'id': str(uuid.uuid4()),
                        'name': item,
                        'quantity': '1',
                        'expirationDate': None,
                        'addedDate': datetime.now().isoformat()
                    })
            
            # Add new items (check for duplicates first)
            existing_names = {item.get('name', '').strip().lower() for item in pantry_list if item.get('name')}
            new_items = []
            for item in pantry_items:
                item_name = item.get('name', '').strip().lower() if item.get('name') else ''
                if item_name and item_name not in existing_names:
                    pantry_list.append(item)
                    existing_names.add(item_name)
                    new_items.append(item)
            
            if new_items:
                update_user_pantry(session['user_id'], pantry_list)
        else:
            # Add to anonymous web pantry (stored in session for persistence)
            # CRITICAL: Mark session as permanent for anonymous users
            session.permanent = True
            
            if 'web_pantry' not in session:
                session['web_pantry'] = []
            
            # Convert to list of dicts if needed
            pantry_list = []
            for item in session['web_pantry']:
                if isinstance(item, dict):
                    pantry_list.append(item)
                else:
                    pantry_list.append({
                        'id': str(uuid.uuid4()),
                        'name': item,
                        'quantity': '1',
                        'expirationDate': None,
                        'addedDate': datetime.now().isoformat()
                    })
            
            # Add new items (check for duplicates first)
            existing_names = {item.get('name', '').strip().lower() for item in pantry_list if item.get('name')}
            new_items = []
            for item in pantry_items:
                item_name = item.get('name', '').strip().lower() if item.get('name') else ''
                if item_name and item_name not in existing_names:
                    pantry_list.append(item)
                    existing_names.add(item_name)
                    new_items.append(item)
            
            if new_items:
                session['web_pantry'] = pantry_list
                # Mark session as modified to ensure it's saved
                session.modified = True
        
        item_names = [item['name'] for item in pantry_items]
        flash(f"Successfully analyzed photo! Added {len(pantry_items)} items: {', '.join(item_names)}", "success")
        
    except ValueError as e:
        # Specific error for missing API key
        flash(f"⚠️ {str(e)} Please configure OPENAI_API_KEY in your Render environment variables.", "danger")
    except Exception as e:
        # Generic error handler
        error_msg = str(e)
        print(f"Error analyzing photo: {error_msg}")
        flash(f"Error analyzing photo: {error_msg}", "danger")
    
    return redirect(url_for("index"))

# Get Nutrition Info (AI powered)
@app.route("/nutrition/<recipe_name>")
def nutrition_info(recipe_name):
    # Get the recipe from the session-stored recipes
    import urllib.parse
    decoded_recipe_name = urllib.parse.unquote(recipe_name)
    
    # Get recipes from session
    recipes = session.get('current_recipes', [])
    
    if not recipes:
        flash("No recipes found in session. Please generate recipes first.", "warning")
        return redirect(url_for("suggest_recipe"))
    
    # Find the specific recipe with flexible matching
    recipe = None
    
    # Try exact match first
    recipe = next((r for r in recipes if r["name"] == decoded_recipe_name), None)
    
    # If not found, try case-insensitive match
    if not recipe:
        recipe = next((r for r in recipes if r["name"].lower() == decoded_recipe_name.lower()), None)
    
    # If still not found, try partial match
    if not recipe:
        recipe = next((r for r in recipes if decoded_recipe_name.lower() in r["name"].lower()), None)
    
    if not recipe:
        available_names = [r["name"] for r in recipes]
        flash(f"Recipe '{decoded_recipe_name}' not found. Available recipes: {', '.join(available_names)}", "danger")
        return redirect(url_for("suggest_recipe"))

    # ✅ Call OpenAI to estimate nutrition based on pantry items
    # Get appropriate pantry based on user authentication
    if 'user_id' in session:
        current_pantry = get_user_pantry(session['user_id'])
    else:
        # Use session-based pantry for anonymous users
        if 'web_pantry' not in session:
            session['web_pantry'] = []
        current_pantry = session['web_pantry']
    
    if not current_pantry or len(current_pantry) == 0:
        flash("Your pantry is empty. Cannot generate nutrition recipes.", "warning")
        return redirect(url_for("index"))
    
    pantry_items = ", ".join(current_pantry)
    prompt = f"""Based on the pantry items: {pantry_items}

Generate 3 creative recipes that use AT LEAST 50% of the pantry ingredients. For each recipe, provide:
1. Recipe name
2. List of ingredients (prioritizing pantry items - at least half must be from pantry)
3. Step-by-step cooking instructions
4. Estimated cooking time
5. Detailed nutrition facts (calories, carbs, protein, fat, fiber per serving)

Format as JSON:
{{
    "recipes": [
        {{
            "name": "Recipe Name",
            "ingredients": ["ingredient1", "ingredient2"],
            "instructions": ["step1", "step2", "step3"],
            "cooking_time": "X minutes",
            "difficulty": "Easy",
            "nutrition": {{
                "calories": "X kcal",
                "carbs": "X g",
                "protein": "X g", 
                "fat": "X g",
                "fiber": "X g"
            }}
        }}
    ]
}}"""

    try:
        if not client:
            raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a nutritionist and chef. Create recipes that use at least 50% pantry ingredients and provide accurate nutrition information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        
        import json
        recipe_text = response.choices[0].message.content
        
        # Clean the response text
        cleaned_text = recipe_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text.replace("```", "").strip()
        
        nutrition_recipes = json.loads(cleaned_text)
        recipes_with_nutrition = nutrition_recipes.get("recipes", [])
        
        # Store the nutrition-enhanced recipes in session
        session['current_recipes'] = recipes_with_nutrition
        
        flash(f"Generated {len(recipes_with_nutrition)} recipe(s) with nutrition info based on your pantry!", "success")
        
    except Exception as e:
        flash(f"Error generating nutrition recipes: {str(e)}", "danger")
        # Use existing recipes from session
        recipes_with_nutrition = session.get('current_recipes', [])
    
    # Convert pantry to string list for template
    pantry_items_list = []
    # Also keep full pantry data for expiration checking
    pantry_items_full = []
    for pantry_item in current_pantry:
        if isinstance(pantry_item, dict):
            name = pantry_item.get('name', '')
            if name:  # Only add non-empty names
                pantry_items_list.append(name)
                pantry_items_full.append(pantry_item)  # Keep full item data
        else:
            pantry_str = str(pantry_item) if pantry_item else ''
            if pantry_str:  # Only add non-empty strings
                pantry_items_list.append(pantry_str)
                pantry_items_full.append({'name': pantry_str, 'expirationDate': None})

    return render_template("suggest_recipe.html", recipes=recipes_with_nutrition, pantry_items=pantry_items_list, pantry_items_full=pantry_items_full)

# Detailed recipe page with timers and full instructions
@app.route("/recipe/<recipe_name>")
def recipe_detail(recipe_name):
    # Get the recipe from the session-stored recipes
    import urllib.parse
    decoded_recipe_name = urllib.parse.unquote(recipe_name)
    
    # Get recipes from session - preserve them
    recipes = session.get('current_recipes', [])
    
    if not recipes:
        flash("No recipes found in session. Please generate recipes first.", "warning")
        return redirect(url_for("suggest_recipe"))
    
    # Ensure recipes are preserved in session (don't modify them)
    # This prevents regeneration when user goes back
    if 'current_recipes' not in session or not session.get('current_recipes'):
        session['current_recipes'] = recipes
    
    # Find the specific recipe with flexible matching
    recipe = None
    
    # Try exact match first
    recipe = next((r for r in recipes if r.get("name") == decoded_recipe_name), None)
    
    # If not found, try case-insensitive match
    if not recipe:
        recipe = next((r for r in recipes if r.get("name", "").lower() == decoded_recipe_name.lower()), None)
    
    # If still not found, try partial match
    if not recipe:
        recipe = next((r for r in recipes if decoded_recipe_name.lower() in r.get("name", "").lower()), None)
    
    if not recipe:
        available_names = [r.get("name", "Unnamed Recipe") for r in recipes if r.get("name")]
        flash(f"Recipe '{decoded_recipe_name}' not found. Available recipes: {', '.join(available_names) if available_names else 'None'}", "danger")
        return redirect(url_for("suggest_recipe"))

    # Check if detailed recipe and nutrition info are already cached in session
    # Use recipe name as key to cache detailed recipes per recipe
    recipe_cache_key = f"detailed_recipe_{decoded_recipe_name}"
    nutrition_cache_key = f"nutrition_{decoded_recipe_name}"
    
    detailed_recipe = session.get(recipe_cache_key)
    nutrition_info = session.get(nutrition_cache_key)
    
    # Only generate if not cached
    if not detailed_recipe:
        # Generate detailed cooking instructions with timers using AI
        ingredients = recipe.get('ingredients', [])
        instructions = recipe.get('instructions', [])
        recipe_name = recipe.get('name', 'Unknown Recipe')
        
        if not ingredients or not instructions:
            flash("Recipe data is incomplete. Please generate new recipes.", "warning")
            return redirect(url_for("suggest_recipe"))
        
        ingredients_text = ", ".join(ingredients)
        instructions_text = " | ".join(instructions)
        
        prompt = f"""For this recipe: {recipe_name}
Ingredients: {ingredients_text}
Basic Instructions: {instructions_text}

Create a detailed cooking guide with:
1. Preparation time
2. Cooking time  
3. Total time
4. Detailed step-by-step instructions with specific timing for each step
5. Cooking tips and techniques

Format as JSON:
{{
    "prep_time": "X minutes",
    "cook_time": "X minutes", 
    "total_time": "X minutes",
    "detailed_steps": [
        {{
            "step": 1,
            "instruction": "Detailed instruction",
            "timer": "X minutes",
            "tips": "Helpful tip"
        }}
    ],
    "cooking_tips": ["tip1", "tip2", "tip3"]
}}"""

        try:
            if not client:
                raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional chef creating detailed cooking instructions with precise timing and helpful tips."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800
            )
            
            import json
            recipe_text = response.choices[0].message.content
            
            # Clean the response text
            cleaned_text = recipe_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text.replace("```", "").strip()
            
            detailed_recipe = json.loads(cleaned_text)
            # Cache the detailed recipe in session
            session[recipe_cache_key] = detailed_recipe
            session.modified = True
            
        except Exception as e:
            print(f"Error generating detailed recipe: {e}")
            # Fallback detailed recipe structure
            detailed_recipe = {
                "prep_time": "15 minutes",
                "cook_time": recipe.get('cooking_time', '20 minutes'),
                "total_time": "35 minutes",
                "detailed_steps": [
                    {
                        "step": 1,
                        "instruction": "Prepare all ingredients",
                        "timer": "5 minutes",
                        "tips": "Read through all instructions first"
                    },
                    {
                        "step": 2,
                        "instruction": "Start cooking process",
                        "timer": recipe.get('cooking_time', '20 minutes'),
                        "tips": "Follow basic recipe instructions"
                    },
                    {
                        "step": 3,
                        "instruction": "Season and serve",
                        "timer": "5 minutes",
                        "tips": "Taste and adjust seasoning"
                    }
                ],
                "cooking_tips": [
                    "Use fresh ingredients when possible",
                    "Don't rush the cooking process",
                    "Taste as you go and adjust seasoning"
                ]
            }
            # Cache the fallback recipe too
            session[recipe_cache_key] = detailed_recipe
            session.modified = True

    # Only generate nutrition info if not cached
    if not nutrition_info:
        try:
            nutrition_prompt = f"""Calculate detailed nutrition facts for this recipe:

Recipe: {recipe['name']}
Ingredients: {', '.join(recipe['ingredients'])}

Provide nutrition facts per serving in JSON format:
{{
    "calories": "X kcal",
    "carbs": "X g",
    "protein": "X g",
    "fat": "X g",
    "fiber": "X g",
    "sugar": "X g",
    "sodium": "X mg"
}}"""

            if not client:
                raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
            nutrition_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional nutritionist. Provide accurate nutrition facts for recipes."},
                    {"role": "user", "content": nutrition_prompt}
                ],
                max_tokens=200
            )
            
            nutrition_text = nutrition_response.choices[0].message.content
            cleaned_nutrition = nutrition_text.strip()
            if cleaned_nutrition.startswith("```json"):
                cleaned_nutrition = cleaned_nutrition.replace("```json", "").replace("```", "").strip()
            elif cleaned_nutrition.startswith("```"):
                cleaned_nutrition = cleaned_nutrition.replace("```", "").strip()
            
            nutrition_info = json.loads(cleaned_nutrition)
            # Cache the nutrition info in session
            session[nutrition_cache_key] = nutrition_info
            session.modified = True

        except Exception as e:
            print(f"Error generating nutrition info: {e}")
            # Fallback nutrition info
            nutrition_info = {
                "calories": "250 kcal",
                "carbs": "30 g",
                "protein": "12 g",
                "fat": "8 g",
                "fiber": "5 g",
                "sugar": "6 g",
                "sodium": "400 mg"
            }
            # Cache the fallback nutrition info too
            session[nutrition_cache_key] = nutrition_info
            session.modified = True

    # CRITICAL: Preserve recipes in session so they don't regenerate when user goes back
    # Only update session if it's missing or empty, don't overwrite existing recipes
    if 'current_recipes' not in session or not session.get('current_recipes') or len(session.get('current_recipes', [])) == 0:
        session['current_recipes'] = recipes
    else:
        # Ensure recipes are still in session (refresh to prevent expiration)
        session['current_recipes'] = session.get('current_recipes', recipes)
    
    # Get pantry items for highlighting ingredients
    if 'user_id' in session:
        current_pantry = get_user_pantry(session['user_id'])
    else:
        if 'web_pantry' not in session:
            session['web_pantry'] = []
        current_pantry = session['web_pantry']
    
    # Convert to full pantry data for expiration checking
    pantry_items_full = []
    for pantry_item in current_pantry:
        if isinstance(pantry_item, dict):
            pantry_items_full.append(pantry_item)
        else:
            pantry_items_full.append({'name': str(pantry_item), 'expirationDate': None})
    
    return render_template("recipe_detail.html", recipe=recipe, detailed_recipe=detailed_recipe, nutrition=nutrition_info, pantry_items_full=pantry_items_full)

# =============================================================================
# API ENDPOINTS FOR MOBILE/FRONTEND APP
# =============================================================================

# Authentication API endpoints
@app.route('/api/auth/signup', methods=['POST'])
def api_signup():
    """Sign up a new user via API"""
    print(f"\n{'='*60}")
    print(f"📥 API SIGNUP REQUEST")
    print(f"{'='*60}")
    print(f"Method: {request.method}")
    print(f"Content-Type: {request.content_type}")
    print(f"Headers: {dict(request.headers)}")
    print(f"X-Client-Type: {request.headers.get('X-Client-Type', 'NOT PROVIDED')}")
    print(f"Raw request data (first 500 chars): {request.get_data()[:500]}")
    
    # Check if request has JSON data
    if not request.is_json:
        print(f"❌ ERROR: Request is not JSON. Content-Type: {request.content_type}")
        return jsonify({'success': False, 'error': 'Request must be JSON. Content-Type should be application/json'}), 400
    
    try:
        data = request.get_json(force=True)  # Force JSON parsing
        print(f"✅ Received JSON data: {data}")
        print(f"   Data type: {type(data)}")
        print(f"   Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
    except Exception as e:
        print(f"❌ Error parsing JSON: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Invalid JSON data: {str(e)}'}), 400
    
    if not data:
        print("❌ ERROR: No data received after parsing")
        return jsonify({'success': False, 'error': 'Invalid request data'}), 400
    
    username = data.get('username', '').strip() if data.get('username') else ''
    email = data.get('email', '').strip() if data.get('email') else ''
    password = data.get('password', '').strip() if data.get('password') else ''
    client_type = request.headers.get('X-Client-Type', 'mobile')
    
    print(f"📋 Extracted values:")
    print(f"   Username: '{username}' (length: {len(username)})")
    print(f"   Email: '{email}' (length: {len(email)})")
    print(f"   Password: {'***' if password else 'EMPTY'} (length: {len(password)})")
    print(f"   Client Type: {client_type}")
    
    if not username or not email or not password:
        print(f"❌ Validation failed: missing required fields")
        print(f"   Username empty: {not username}")
        print(f"   Email empty: {not email}")
        print(f"   Password empty: {not password}")
        return jsonify({'success': False, 'error': 'Username, email, and password are required'}), 400
    
    if len(password) < 6:
        print(f"❌ Validation failed: password too short ({len(password)} chars)")
        return jsonify({'success': False, 'error': 'Password must be at least 6 characters'}), 400
    
    print(f"💾 Calling create_user()...")
    user_id, error = create_user(username, email, password, client_type)
    if user_id:
        print(f"✅ User created successfully: {user_id}")
        print(f"{'='*60}\n")
        return jsonify({
            'success': True,
            'message': 'Account created successfully',
            'user_id': user_id,
            'username': username
        }), 200
    else:
        print(f"❌ User creation failed: {error}")
        print(f"{'='*60}\n")
        return jsonify({'success': False, 'error': error}), 409

@app.route('/api/admin/delete-all-users', methods=['POST'])
def api_delete_all_users():
    """Delete all users via API - ADMIN ONLY (for testing/cleanup)"""
    try:
        # Clear in-memory storage
        global _in_memory_users
        _in_memory_users = {}
        
        # Clear file storage
        empty_users = {}
        save_users(empty_users)
        
        print("All users deleted successfully via API")
        return jsonify({'success': True, 'message': 'All users deleted successfully'}), 200
    except Exception as e:
        print(f"Error deleting users: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Login user via API"""
    try:
        data = request.get_json()
        if not data:
            print("ERROR: No JSON data received in login request")
            return jsonify({'success': False, 'error': 'Invalid request data'}), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        client_type = request.headers.get('X-Client-Type', 'mobile')
        
        print(f"API Login attempt - Username: '{username}', Password provided: {'Yes' if password else 'No'}, Client: {client_type}")
        
        if not username or not password:
            print("ERROR: Username or password is empty")
            return jsonify({'success': False, 'error': 'Username and password are required'}), 400
        
        # Authenticate user - this function returns specific error messages
        user_data, error = authenticate_user(username, password, client_type)
        if user_data:
            print(f"✅ API Login successful for user '{username}' (ID: {user_data['id']})")
            # Convert pantry to new format if needed
            pantry = user_data.get('pantry', [])
            pantry_items = []
            for item in pantry:
                if isinstance(item, dict):
                    pantry_items.append(item)
                else:
                    pantry_items.append({
                        'id': str(uuid.uuid4()),
                        'name': item,
                        'quantity': '1',
                        'expirationDate': None,
                        'addedDate': datetime.now().isoformat()
                    })
            
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'user_id': user_data['id'],
                'username': user_data['username'],
                'email': user_data['email'],
                'pantry': pantry_items
            }), 200
        else:
            print(f"❌ API Login failed: {error}")
            return jsonify({'success': False, 'error': error}), 401
    except Exception as e:
        print(f"ERROR: Exception in api_login: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/pantry', methods=['GET'])
def api_get_pantry():
    """Get all pantry items as JSON"""
    
    try:
        client_type = request.headers.get('X-Client-Type', 'web')
        user_id = request.headers.get('X-User-ID')
        
        # Check if user is authenticated
        if user_id:
            pantry_to_use = get_user_pantry(user_id)
        else:
            # Use anonymous pantry
            pantry_to_use = mobile_pantry if client_type == 'mobile' else web_pantry
        
        # Ensure pantry_to_use is a list
        if not isinstance(pantry_to_use, list):
            pantry_to_use = []
        
        # Convert pantry items to new format if needed (backward compatibility)
        items = []
        for item in pantry_to_use:
            try:
                if isinstance(item, dict):
                    # Normalize item to ensure all fields are present and valid
                    normalized_item = normalize_pantry_item(item.copy())
                    # Only include items with valid names
                    if normalized_item.get('name') and normalized_item.get('name').strip() and normalized_item.get('name') != 'Unnamed Item':
                        items.append(normalized_item)
                elif item is not None:
                    # Old string format - convert to new format
                    item_str = str(item).strip() if item else ''
                    if item_str:
                        normalized_item = normalize_pantry_item(item_str)
                        if normalized_item.get('name') and normalized_item.get('name').strip() and normalized_item.get('name') != 'Unnamed Item':
                            items.append(normalized_item)
            except Exception as e:
                print(f"Warning: Failed to process item {item}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return jsonify({
            'success': True,
            'items': items,
            'count': len(items)
        })
    except Exception as e:
        print(f"Error in api_get_pantry: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'items': [],
            'count': 0
        }), 500

@app.route('/api/pantry', methods=['POST'])
def api_add_item():
    """Add an item to pantry via API"""
    global mobile_pantry, web_pantry  # Declare globals at function start
    
    # Log request details for debugging
    print(f"\n{'='*60}")
    print(f"📥 API ADD ITEM REQUEST")
    print(f"{'='*60}")
    print(f"Method: {request.method}")
    print(f"Content-Type: {request.content_type}")
    print(f"Headers: {dict(request.headers)}")
    print(f"X-User-ID: {request.headers.get('X-User-ID', 'NOT PROVIDED')}")
    print(f"X-Client-Type: {request.headers.get('X-Client-Type', 'NOT PROVIDED')}")
    print(f"Raw request data (first 500 chars): {request.get_data()[:500]}")
    print(f"Raw request data (hex): {request.get_data()[:100].hex()}")
    
    # Also check form data (in case it's being sent as form-data)
    print(f"Form data: {dict(request.form)}")
    print(f"Form files: {list(request.files.keys())}")
    
    # Check if request has JSON data
    if not request.is_json:
        print(f"❌ ERROR: Request is not JSON. Content-Type: {request.content_type}")
        print(f"   Raw data: {request.get_data()[:200]}")  # First 200 chars
        # Try to parse as form data
        if request.form:
            print(f"   Found form data: {dict(request.form)}")
            return jsonify({'success': False, 'error': 'Request must be JSON. Content-Type should be application/json. Received form data instead.'}), 400
        return jsonify({'success': False, 'error': 'Request must be JSON. Content-Type should be application/json'}), 400
    
    try:
        data = request.get_json(force=True)  # Force JSON parsing
        print(f"✅ Received data: {data}")
        print(f"   Data type: {type(data)}")
        print(f"   Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
    except Exception as e:
        print(f"❌ Error parsing JSON: {e}")
        print(f"   Raw request data: {request.get_data()[:500]}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Invalid JSON data: {str(e)}'}), 400
    
    if not data:
        print("❌ ERROR: No data received after parsing")
        print(f"   Raw request data: {request.get_data()[:500]}")
        return jsonify({'success': False, 'error': 'Invalid request data'}), 400
    
    # Support both old format (item: string) and new format (PantryItem object)
    if 'item' in data:
        # Old format - convert to new format
        item_name = data['item'].strip()
        if not item_name:
            return jsonify({'success': False, 'error': 'Item name cannot be empty'}), 400
        
        quantity = data.get('quantity', '1')
        expiration_date = data.get('expirationDate')
        
        pantry_item = {
            'id': str(uuid.uuid4()),
            'name': item_name,
            'quantity': quantity,
            'expirationDate': expiration_date,
            'addedDate': datetime.now().isoformat()
        }
    elif 'name' in data:
        # New format - PantryItem object
        item_name = data.get('name', '').strip() if data.get('name') else ''
        if not item_name:
            print(f"ERROR: Item name is empty or missing. Data received: {data}")
            return jsonify({'success': False, 'error': 'Item name cannot be empty'}), 400
        
        # Handle expirationDate - can be None, empty string, or a valid date string
        expiration_date = data.get('expirationDate')
        if expiration_date == '' or expiration_date is None:
            expiration_date = None
        
        # Handle quantity - ensure it's a valid string, default to '1' if None or empty
        quantity = data.get('quantity', '1')
        if not quantity or (isinstance(quantity, str) and quantity.strip() == ''):
            quantity = '1'
        elif not isinstance(quantity, str):
            quantity = str(quantity)  # Convert to string if it's a number
        
        pantry_item = {
            'id': data.get('id', str(uuid.uuid4())),
            'name': item_name,
            'quantity': quantity.strip() if isinstance(quantity, str) else str(quantity),
            'expirationDate': expiration_date,
            'addedDate': data.get('addedDate', datetime.now().isoformat())
        }
        
        print(f"✅ Created pantry item: name='{pantry_item['name']}', quantity='{pantry_item['quantity']}', expirationDate={pantry_item['expirationDate']}")
    else:
        print(f"ERROR: Neither 'item' nor 'name' found in data. Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        print(f"Full data: {data}")
        return jsonify({'success': False, 'error': 'Item name required. Please provide either "item" or "name" field.'}), 400
    
    client_type = request.headers.get('X-Client-Type', 'web')
    user_id = request.headers.get('X-User-ID')
    
    # Check if user is authenticated
    if user_id:
        pantry_to_use = get_user_pantry(user_id)
        # Convert to list of dicts if needed
        pantry_list = []
        for item in pantry_to_use:
            if isinstance(item, dict):
                pantry_list.append(item)
            else:
                pantry_list.append({
                    'id': str(uuid.uuid4()),
                    'name': item,
                    'quantity': '1',
                    'expirationDate': None,
                    'addedDate': datetime.now().isoformat()
                })
        
        # Check for duplicates (case-insensitive name match)
        # Safely handle None or missing name values
        item_name = pantry_item.get('name', '').strip() if pantry_item.get('name') else ''
        if item_name:
            for i in pantry_list:
                existing_name = i.get('name', '').strip() if i.get('name') else ''
                if existing_name and existing_name.lower() == item_name.lower():
                    return jsonify({'success': False, 'error': f'"{item_name}" is already in pantry'}), 409
        
        pantry_list.append(pantry_item)
        print(f"💾 Updating pantry for user {user_id} with {len(pantry_list)} items")
        update_user_pantry(user_id, pantry_list)
        print(f"✅ Successfully added item '{pantry_item['name']}' to user {user_id}'s pantry")
        return jsonify({
            'success': True,
            'message': f'Added "{pantry_item["name"]}" to pantry',
            'item': pantry_item,
            'total_items': len(pantry_list)
        }), 200
    else:
        # Use anonymous pantry
        global mobile_pantry, web_pantry  # Declare globals BEFORE using them
        pantry_to_use = mobile_pantry if client_type == 'mobile' else web_pantry
        # Ensure pantry_to_use is a list
        if not isinstance(pantry_to_use, list):
            pantry_to_use = []
        # Convert to list of dicts if needed
        pantry_list = []
        for item in pantry_to_use:
            if isinstance(item, dict):
                pantry_list.append(item)
            else:
                pantry_list.append({
                    'id': str(uuid.uuid4()),
                    'name': item,
                    'quantity': '1',
                    'expirationDate': None,
                    'addedDate': datetime.now().isoformat()
                })
        
        # Check for duplicates (case-insensitive name match) - FIX: Missing duplicate check
        item_name = pantry_item.get('name', '').strip() if pantry_item.get('name') else ''
        if item_name:
            for i in pantry_list:
                existing_name = i.get('name', '').strip() if i.get('name') else ''
                if existing_name and existing_name.lower() == item_name.lower():
                    return jsonify({'success': False, 'error': f'"{item_name}" is already in pantry'}), 409
        
        pantry_list.append(pantry_item)
        if client_type == 'mobile':
            mobile_pantry = pantry_list
        else:
            web_pantry = pantry_list
        
        print(f"✅ Successfully added item '{pantry_item['name']}' to anonymous {client_type} pantry")
        return jsonify({
            'success': True,
            'message': f'Added "{pantry_item["name"]}" to pantry',
            'item': pantry_item,
            'total_items': len(pantry_list)
        }), 200

@app.route('/api/pantry/<item_id>', methods=['DELETE'])
def api_delete_item(item_id):
    """Delete an item from pantry via API (by ID or name for backward compatibility)"""
    global mobile_pantry, web_pantry  # Declare globals at function start
    
    print(f"\n{'='*60}")
    print(f"🗑️ API DELETE ITEM REQUEST")
    print(f"{'='*60}")
    print(f"Item ID: {item_id}")
    print(f"X-User-ID: {request.headers.get('X-User-ID', 'NOT PROVIDED')}")
    print(f"X-Client-Type: {request.headers.get('X-Client-Type', 'NOT PROVIDED')}")
    
    client_type = request.headers.get('X-Client-Type', 'web')
    user_id = request.headers.get('X-User-ID')
    
    # URL decode the item_id
    from urllib.parse import unquote
    item_id = unquote(item_id)
    print(f"Decoded item ID: {item_id}")
    
    # Check if user is authenticated
    if user_id:
        pantry_to_use = get_user_pantry(user_id)
        # Convert to list of dicts if needed
        pantry_list = []
        for item in pantry_to_use:
            if isinstance(item, dict):
                pantry_list.append(item)
            else:
                pantry_list.append({
                    'id': str(uuid.uuid4()),
                    'name': item,
                    'quantity': '1',
                    'expirationDate': None,
                    'addedDate': datetime.now().isoformat()
                })
        
        # Find item by ID (exact match) or name (case-insensitive, trimmed) for backward compatibility
        item_to_delete = None
        item_id_clean = item_id.strip()
        item_id_clean_lower = item_id_clean.lower()
        for i, pantry_item in enumerate(pantry_list):
            if isinstance(pantry_item, dict):
                item_id_from_dict = pantry_item.get('id', '').strip() if pantry_item.get('id') else ''
                item_name = pantry_item.get('name', '').strip().lower() if pantry_item.get('name') else ''
                # Try exact ID match first (case-sensitive), then fallback to name match (case-insensitive)
                if item_id_from_dict == item_id_clean or item_name == item_id_clean_lower:
                    item_to_delete = pantry_list.pop(i)
                    break
            else:
                # Handle old string format - compare case-insensitively
                pantry_str = str(pantry_item).strip().lower() if pantry_item else ''
                if pantry_str == item_id_clean_lower:
                    item_to_delete = pantry_list.pop(i)
                    break
        
        if item_to_delete:
            print(f"✅ Found item to delete: {item_to_delete}")
            print(f"   Pantry before delete: {len(pantry_list)} items")
            update_user_pantry(user_id, pantry_list)
            item_name = item_to_delete.get('name', item_id) if isinstance(item_to_delete, dict) else item_to_delete
            print(f"✅ Successfully deleted item '{item_name}' from user {user_id}'s pantry")
            print(f"   Pantry after delete: {len(pantry_list)} items")
            print(f"{'='*60}\n")
            return jsonify({
                'success': True,
                'message': f'Removed "{item_name}" from pantry',
                'total_items': len(pantry_list)
            }), 200
        else:
            print(f"❌ Item not found in pantry. Item ID: {item_id}")
            print(f"   User ID: {user_id}")
            print(f"   Pantry has {len(pantry_list)} items")
            if pantry_list:
                print(f"   Available item IDs: {[item.get('id', 'unknown') if isinstance(item, dict) else str(item) for item in pantry_list[:5]]}")
            print(f"{'='*60}\n")
            return jsonify({'success': False, 'error': f'Item not found in pantry'}), 404
    else:
        # Use anonymous pantry
        pantry_to_use = mobile_pantry if client_type == 'mobile' else web_pantry
        # Ensure pantry_to_use is a list
        if not isinstance(pantry_to_use, list):
            pantry_to_use = []
        # Convert to list of dicts if needed
        pantry_list = []
        for item in pantry_to_use:
            if isinstance(item, dict):
                pantry_list.append(item)
            else:
                pantry_list.append({
                    'id': str(uuid.uuid4()),
                    'name': item,
                    'quantity': '1',
                    'expirationDate': None,
                    'addedDate': datetime.now().isoformat()
                })
        
        # Find item by ID (exact match) or name (case-insensitive, trimmed) for backward compatibility
        item_to_delete = None
        item_id_clean = item_id.strip()
        item_id_clean_lower = item_id_clean.lower()
        for i, pantry_item in enumerate(pantry_list):
            if isinstance(pantry_item, dict):
                item_id_from_dict = pantry_item.get('id', '').strip() if pantry_item.get('id') else ''
                item_name = pantry_item.get('name', '').strip().lower() if pantry_item.get('name') else ''
                # Try exact ID match first (case-sensitive), then fallback to name match (case-insensitive)
                if item_id_from_dict == item_id_clean or item_name == item_id_clean_lower:
                    item_to_delete = pantry_list.pop(i)
                    break
            else:
                # Handle old string format - compare case-insensitively
                pantry_str = str(pantry_item).strip().lower() if pantry_item else ''
                if pantry_str == item_id_clean_lower:
                    item_to_delete = pantry_list.pop(i)
                    break
        
        if item_to_delete:
            if client_type == 'mobile':
                mobile_pantry = pantry_list
            else:
                web_pantry = pantry_list
            item_name = item_to_delete.get('name', item_id) if isinstance(item_to_delete, dict) else item_to_delete
            return jsonify({
                'success': True,
                'message': f'Removed "{item_name}" from pantry',
                'total_items': len(pantry_list)
            })
        else:
            return jsonify({'success': False, 'error': f'Item not found in pantry'}), 404

@app.route('/api/pantry/<item_id>', methods=['PUT'])
def api_update_item(item_id):
    """Update an item in pantry via API"""
    from urllib.parse import unquote
    global mobile_pantry, web_pantry  # Declare global at function start
    
    item_id = unquote(item_id)
    
    print(f"\n{'='*60}")
    print(f"🔄 API UPDATE ITEM REQUEST")
    print(f"{'='*60}")
    print(f"Item ID: {item_id}")
    print(f"X-User-ID: {request.headers.get('X-User-ID', 'NOT PROVIDED')}")
    print(f"X-Client-Type: {request.headers.get('X-Client-Type', 'NOT PROVIDED')}")
    print(f"Session keys: {list(session.keys())}")
    print(f"Session user_id: {session.get('user_id', 'NOT IN SESSION')}")
    print(f"Session username: {session.get('username', 'NOT IN SESSION')}")
    
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    client_type = request.headers.get('X-Client-Type', 'web')
    # Try to get user_id from headers first (for mobile), then from session (for web)
    user_id = request.headers.get('X-User-ID')
    if not user_id and 'user_id' in session:
        user_id = session['user_id']
        print(f"✅ Using user_id from session: {user_id}")
    elif not user_id:
        print(f"⚠️ No user_id in headers or session - treating as anonymous user")
    
    # Validate required fields
    item_name = data.get('name', '').strip() if data.get('name') else ''
    if not item_name:
        return jsonify({'success': False, 'error': 'Item name cannot be empty'}), 400
    
    # Handle expirationDate
    expiration_date = data.get('expirationDate')
    if expiration_date == '' or expiration_date is None:
        expiration_date = None
    
    # Handle quantity - check if it's 0 or less (should delete item)
    quantity = data.get('quantity', '1')
    quantity_num = 0
    try:
        if isinstance(quantity, str):
            quantity_num = int(quantity.strip()) if quantity.strip() else 0
        else:
            quantity_num = int(quantity) if quantity else 0
    except (ValueError, TypeError):
        quantity_num = 0
    
    # Initialize pantry_list early to avoid UnboundLocalError
    # This must be done before any conditional blocks that might use it
    pantry_list = []
    item_found = False
    
    # If quantity is 0 or less, delete the item instead of updating
    if quantity_num <= 0:
        # Delete the item
        if user_id:
            pantry_to_use = get_user_pantry(user_id)
            pantry_list.clear()  # Use clear() instead of reassignment
            item_found = False
            item_id_clean = item_id.strip() if item_id else ''
            # Treat 'unknown' as empty ID (frontend sends 'unknown' when item has no ID)
            if item_id_clean == 'unknown':
                item_id_clean = ''
            item_name_lower = item_name.lower().strip()
            
            for item in pantry_to_use:
                if isinstance(item, dict):
                    item_id_from_dict = item.get('id', '').strip() if item.get('id') else ''
                    item_name_from_dict = item.get('name', '').strip().lower() if item.get('name') else ''
                    
                    # Skip the item to delete (match by ID or name)
                    # If ID is empty or 'unknown', match by name only
                    if (item_id_clean and item_id_from_dict and item_id_from_dict == item_id_clean) or item_name_from_dict == item_name_lower:
                        item_found = True
                        continue  # Don't add to pantry_list (effectively deletes it)
                    pantry_list.append(item)
                else:
                    pantry_list.append({
                        'id': str(uuid.uuid4()),
                        'name': item,
                        'quantity': '1',
                        'expirationDate': None,
                        'addedDate': datetime.now().isoformat()
                    })
            
            if item_found:
                update_user_pantry(user_id, pantry_list)
                return jsonify({
                    'success': True,
                    'message': f'Item "{item_name}" removed (quantity reached 0)',
                    'deleted': True,
                    'total_items': len(pantry_list)
                }), 200
            else:
                return jsonify({'success': False, 'error': 'Item not found in pantry'}), 404
        else:
            # Anonymous user - delete from session pantry
            client_type = request.headers.get('X-Client-Type', 'web')
            pantry_to_use = mobile_pantry if client_type == 'mobile' else web_pantry
            if not isinstance(pantry_to_use, list):
                pantry_to_use = []
            
            pantry_list.clear()  # Use clear() instead of reassignment
            item_found = False
            item_id_clean = item_id.strip() if item_id else ''
            # Treat 'unknown' as empty ID (frontend sends 'unknown' when item has no ID)
            if item_id_clean == 'unknown':
                item_id_clean = ''
            item_name_lower = item_name.lower().strip()
            
            for item in pantry_to_use:
                if isinstance(item, dict):
                    item_id_from_dict = item.get('id', '').strip() if item.get('id') else ''
                    item_name_from_dict = item.get('name', '').strip().lower() if item.get('name') else ''
                    
                    # Match by ID or name (if ID is empty or 'unknown', match by name only)
                    if (item_id_clean and item_id_from_dict and item_id_from_dict == item_id_clean) or item_name_from_dict == item_name_lower:
                        item_found = True
                        continue
                    pantry_list.append(item)
                else:
                    pantry_list.append({
                        'id': str(uuid.uuid4()),
                        'name': item,
                        'quantity': '1',
                        'expirationDate': None,
                        'addedDate': datetime.now().isoformat()
                    })
            
            if item_found:
                if client_type == 'mobile':
                    mobile_pantry = pantry_list
                else:
                    web_pantry = pantry_list
                    session['web_pantry'] = pantry_list
                    session.modified = True
                
                return jsonify({
                    'success': True,
                    'message': f'Item "{item_name}" removed (quantity reached 0)',
                    'deleted': True,
                    'total_items': len(pantry_list)
                }), 200
            else:
                return jsonify({'success': False, 'error': 'Item not found in pantry'}), 404
    
    # Normal quantity update (quantity > 0)
    if not isinstance(quantity, str):
        quantity = str(quantity)
    
    # Don't set ID in updated_item yet - we'll set it when we find the matching item
    # This prevents setting 'unknown' as the ID
    updated_item = {
        'name': item_name,
        'quantity': quantity.strip() if isinstance(quantity, str) else str(quantity),
        'expirationDate': expiration_date,
        'addedDate': data.get('addedDate', datetime.now().isoformat())
    }
    
    # Only set ID if we have a valid one from the request
    if item_id and item_id.strip() and item_id.strip() != 'unknown':
        updated_item['id'] = item_id.strip()
    
    # pantry_list and item_found are already initialized above (before quantity check)
    # Check if user is authenticated
    if user_id:
        print(f"✅ User authenticated: {user_id}")
        pantry_to_use = get_user_pantry(user_id)
        print(f"📦 User pantry has {len(pantry_to_use) if isinstance(pantry_to_use, list) else 0} items")
        # Convert to list of dicts if needed (reuse initialized pantry_list)
        pantry_list.clear()  # Clear instead of reassigning to avoid UnboundLocalError
        item_found = False
        
        for item in pantry_to_use:
            if isinstance(item, dict):
                pantry_list.append(item)
            else:
                pantry_list.append({
                    'id': str(uuid.uuid4()),
                    'name': item,
                    'quantity': '1',
                    'expirationDate': None,
                    'addedDate': datetime.now().isoformat()
                })
        print(f"📦 Converted pantry list has {len(pantry_list)} items")
        
        # Find and update item by ID (exact match, case-sensitive for IDs)
        # If ID match fails, fallback to name match (case-insensitive) for backward compatibility
        item_id_clean = item_id.strip() if item_id else ''
        # Treat 'unknown' as empty ID (frontend sends 'unknown' when item has no ID)
        if item_id_clean == 'unknown':
            item_id_clean = ''
        item_name_lower = item_name.lower().strip()
        
        for i, pantry_item in enumerate(pantry_list):
            if isinstance(pantry_item, dict):
                item_id_from_dict = pantry_item.get('id', '').strip() if pantry_item.get('id') else ''
                item_name_from_dict = pantry_item.get('name', '').strip().lower() if pantry_item.get('name') else ''
                
                # Try exact ID match first (only if we have a valid ID to match)
                if item_id_clean and item_id_from_dict and item_id_from_dict == item_id_clean:
                    # Ensure the updated item retains its original ID
                    if not updated_item.get('id'):
                        updated_item['id'] = item_id_from_dict
                    pantry_list[i] = updated_item
                    item_found = True
                    break
                
                # Fallback to name match (case-insensitive) if ID doesn't match or is empty
                # This handles cases where item has no ID or ID is 'unknown'
                if item_name_from_dict == item_name_lower:
                    # Preserve existing ID if item has one, otherwise generate new ID
                    if not updated_item.get('id'):
                        updated_item['id'] = item_id_from_dict if item_id_from_dict else str(uuid.uuid4())
                    pantry_list[i] = updated_item
                    item_found = True
                    break
        
        if item_found:
            print(f"💾 Updating pantry for user {user_id} with {len(pantry_list)} items")
            update_user_pantry(user_id, pantry_list)
            print(f"✅ Successfully updated item '{updated_item['name']}' in user {user_id}'s pantry")
            return jsonify({
                'success': True,
                'message': f'Updated "{updated_item["name"]}" in pantry',
                'item': updated_item,
                'total_items': len(pantry_list)
            }), 200
        else:
            print(f"❌ Item not found in pantry. Item ID: {item_id}, Item Name: {item_name}")
            print(f"   Cleaned Item ID: '{item_id_clean}' (was '{item_id}')")
            print(f"   User ID: {user_id}")
            print(f"   Pantry has {len(pantry_list)} items")
            if pantry_list:
                print(f"   Available items: {[item.get('name', 'NO_NAME') + ' (ID: ' + (item.get('id', 'NO_ID') or 'NO_ID') + ')' for item in pantry_list[:5]]}")
                # Also check if name matches exist (case-insensitive)
                matching_names = [item.get('name', '') for item in pantry_list if item.get('name', '').lower().strip() == item_name_lower]
                if matching_names:
                    print(f"   ⚠️ Found items with matching name (case-insensitive): {matching_names}")
            else:
                print(f"   Pantry is empty")
            print(f"{'='*60}\n")
            return jsonify({'success': False, 'error': f'Item not found in pantry'}), 404
    else:
        # Use anonymous pantry
        print(f"⚠️ No user_id found - using anonymous pantry (client_type: {client_type})")
        pantry_to_use = mobile_pantry if client_type == 'mobile' else web_pantry
        # Ensure pantry_to_use is a list
        if not isinstance(pantry_to_use, list):
            pantry_to_use = []
        print(f"📦 Anonymous pantry has {len(pantry_to_use)} items")
        # Convert to list of dicts if needed (reuse initialized pantry_list)
        pantry_list.clear()  # Clear instead of reassigning to avoid UnboundLocalError
        item_found = False
        
        for item in pantry_to_use:
            if isinstance(item, dict):
                pantry_list.append(item)
            else:
                pantry_list.append({
                    'id': str(uuid.uuid4()),
                    'name': item,
                    'quantity': '1',
                    'expirationDate': None,
                    'addedDate': datetime.now().isoformat()
                })
        
        # Find and update item by ID (exact match, case-sensitive for IDs)
        # If ID match fails, fallback to name match (case-insensitive) for backward compatibility
        item_id_clean = item_id.strip() if item_id else ''
        # Treat 'unknown' as empty ID (frontend sends 'unknown' when item has no ID)
        if item_id_clean == 'unknown':
            item_id_clean = ''
        item_name_lower = item_name.lower().strip()
        
        for i, pantry_item in enumerate(pantry_list):
            if isinstance(pantry_item, dict):
                item_id_from_dict = pantry_item.get('id', '').strip() if pantry_item.get('id') else ''
                item_name_from_dict = pantry_item.get('name', '').strip().lower() if pantry_item.get('name') else ''
                
                # Try exact ID match first (only if we have a valid ID to match)
                if item_id_clean and item_id_from_dict and item_id_from_dict == item_id_clean:
                    # Ensure the updated item retains its original ID
                    if not updated_item.get('id'):
                        updated_item['id'] = item_id_from_dict
                    pantry_list[i] = updated_item
                    item_found = True
                    break
                
                # Fallback to name match (case-insensitive) if ID doesn't match or is empty
                # This handles cases where item has no ID or ID is 'unknown'
                if item_name_from_dict == item_name_lower:
                    # Preserve existing ID if item has one, otherwise generate new ID
                    if not updated_item.get('id'):
                        updated_item['id'] = item_id_from_dict if item_id_from_dict else str(uuid.uuid4())
                    pantry_list[i] = updated_item
                    item_found = True
                    break
        
        if item_found:
            if client_type == 'mobile':
                mobile_pantry = pantry_list
            else:
                web_pantry = pantry_list
            return jsonify({
                'success': True,
                'message': f'Updated "{updated_item["name"]}" in pantry',
                'item': updated_item,
                'total_items': len(pantry_list)
            }), 200
        else:
            print(f"❌ Item not found in anonymous pantry. Item ID: {item_id}, Item Name: {item_name}")
            print(f"   Cleaned Item ID: '{item_id_clean}' (was '{item_id}')")
            print(f"   Client Type: {client_type}")
            print(f"   Pantry has {len(pantry_list)} items")
            if pantry_list:
                print(f"   Available items: {[item.get('name', 'NO_NAME') + ' (ID: ' + (item.get('id', 'NO_ID') or 'NO_ID') + ')' for item in pantry_list[:5]]}")
                # Also check if name matches exist (case-insensitive)
                matching_names = [item.get('name', '') for item in pantry_list if item.get('name', '').lower().strip() == item_name_lower]
                if matching_names:
                    print(f"   ⚠️ Found items with matching name (case-insensitive): {matching_names}")
            else:
                print(f"   Pantry is empty")
            print(f"{'='*60}\n")
            return jsonify({'success': False, 'error': f'Item not found in pantry'}), 404

@app.route('/api/recipes/suggest', methods=['POST'])
def api_suggest_recipe():
    """Get AI recipe suggestions via API"""
    data = request.get_json()
    client_type = request.headers.get('X-Client-Type', 'web')
    user_id = request.headers.get('X-User-ID')
    
    # Get appropriate pantry
    if user_id:
        default_pantry = get_user_pantry(user_id)
    else:
        default_pantry = mobile_pantry if client_type == 'mobile' else web_pantry
    
    pantry_items = data.get('pantry_items', default_pantry) if data else default_pantry
    
    if not pantry_items:
        return jsonify({'success': False, 'error': 'No items in pantry'}), 400
    
    # Process pantry items with quantities
    pantry_with_quantities = []
    item_names = []
    expiring_items = []
    
    if pantry_items and isinstance(pantry_items[0], dict):
        # Sort by expiration: expiring soon first
        sorted_items = sorted(pantry_items, key=lambda x: (
            x.get('expirationDate') is None,  # Items without dates go last
            x.get('expirationDate', '')  # Then sort by date
        ))
        
        from datetime import datetime
        today = datetime.now().date()
        
        for item in sorted_items:
            name = item.get('name', '')
            quantity = item.get('quantity', '1')
            if name:
                pantry_with_quantities.append(f"{name} ({quantity})")
                item_names.append(name)
                
                # Check if expiring soon
                exp_date_str = item.get('expirationDate')
                if exp_date_str:
                    try:
                        exp_date = datetime.fromisoformat(exp_date_str.replace('Z', '+00:00')).date()
                        days_left = (exp_date - today).days
                        if 0 <= days_left <= 7:
                            expiring_items.append(f"{name} ({quantity})")
                    except:
                        pass
    else:
        # Old format - just strings
        item_names = pantry_items if isinstance(pantry_items, list) else []
        pantry_with_quantities = [f"{item} (1)" for item in item_names]
    
    if not item_names:
        return jsonify({'success': False, 'error': 'No items in pantry'}), 400
    
    try:
        # Generate AI recipes - prioritize items that are expiring soon and use quantities
        pantry_list = ", ".join(pantry_with_quantities)
        priority_note = ""
        if expiring_items:
            priority_note = f"\n\nIMPORTANT: Prioritize using these items that are expiring soon (within 7 days): {', '.join(expiring_items)}. Try to include at least one of these in each recipe."
        
        prompt = f"""
        Create 3 delicious and diverse recipes using these available ingredients WITH QUANTITIES: {pantry_list}
        {priority_note}
        
        CRITICAL REQUIREMENTS:
        - Each recipe MUST use the EXACT quantities available from the pantry items listed above
        - Calculate serving sizes based on the available quantities (e.g., if you have "2 bottles of milk", create a recipe that uses 2 bottles and adjust servings accordingly)
        - Scale all other ingredients proportionally to match the serving size
        - If a recipe normally serves 4 but you have "2 bottles of milk" (which might be 1 liter each), adjust the recipe to use both bottles and calculate appropriate servings (e.g., 6-8 servings)
        - Each recipe must use at least 2-3 ingredients from the pantry list above
        - Use the full quantity of pantry items when possible to minimize waste
        - Include basic pantry staples (salt, pepper, oil, butter) as needed, scaled appropriately
        - Make recipes practical and easy to follow
        - Include realistic cooking times and difficulty levels
        - Make each recipe different (different cuisine, cooking method, etc.)
        - Assess the overall healthiness of each recipe and assign a health rating
        - Include dietary information (vegan, vegetarian, halal, etc.) if applicable
        - Add timer steps for cooking steps that require specific timing (frying, baking, simmering, etc.)
        
        QUANTITY AND SERVING CALCULATION EXAMPLES:
        - If pantry has "2 bottles of milk (500ml each)" → Recipe should use 1 liter total, calculate servings based on typical milk usage (e.g., 4-6 servings for a milk-based dish)
        - If pantry has "3 cans of soup (400g each)" → Recipe should use all 3 cans (1200g total), adjust servings to 6-8 people
        - If pantry has "5 slices of pizza" → Recipe should use all 5 slices, serving size is 5 servings
        - Always scale non-pantry ingredients (spices, oil, etc.) proportionally to match the serving size
        
        Health Rating Guidelines:
        - "Healthy": Recipes with mostly vegetables, lean proteins, whole grains, minimal processed ingredients
        - "Moderately Healthy": Recipes with some healthy ingredients but also some less healthy elements
        - "Unhealthy": Recipes with high amounts of processed foods, sugars, unhealthy fats, or fried items
        
        Dietary Information Guidelines:
        - Only include if clearly applicable: "Vegan", "Vegetarian", "Halal", "Kosher", "Gluten-Free", "Dairy-Free"
        - Leave empty array if no specific dietary restrictions apply
        
        Timer Steps Guidelines:
        - Only include for steps that require specific timing (frying, baking, simmering, etc.)
        - Include stepNumber (1-based), instruction, duration in minutes, and description
        - Example: {{"stepNumber": 2, "instruction": "Fry the onions until golden brown", "duration": 5, "description": "Fry onions"}}
        
        Return a JSON response with this exact structure:
        {{
            "recipes": [
                {{
                    "name": "Recipe Name 1",
                    "description": "Brief description of the dish",
                    "ingredients": [
                        "2 cups ingredient from pantry",
                        "1 tsp salt",
                        "2 tbsp oil"
                    ],
                    "instructions": [
                        "Step 1: Prepare ingredients",
                        "Step 2: Cook the dish",
                        "Step 3: Serve hot"
                    ],
                    "prepTime": "15 minutes",
                    "cookTime": "30 minutes",
                    "difficulty": "Easy",
                    "servings": 4,
                    "healthRating": "Healthy",
                    "dietaryInfo": ["Vegetarian"],
                    "timerSteps": [
                        {{"stepNumber": 2, "instruction": "Fry the vegetables until golden", "duration": 8, "description": "Fry vegetables"}}
                    ],
                    "nutrition": {{
                        "calories": "350 kcal",
                        "carbs": "45g",
                        "protein": "20g",
                        "fat": "12g"
                    }}
                }},
                {{
                    "name": "Recipe Name 2",
                    "description": "Brief description of the dish",
                    "ingredients": [
                        "1 cup ingredient from pantry",
                        "1/2 tsp pepper",
                        "1 tbsp butter"
                    ],
                    "instructions": [
                        "Step 1: Heat pan",
                        "Step 2: Add ingredients",
                        "Step 3: Cook until done"
                    ],
                    "prepTime": "10 minutes",
                    "cookTime": "20 minutes",
                    "difficulty": "Medium",
                    "servings": 2,
                    "healthRating": "Moderately Healthy",
                    "dietaryInfo": [],
                    "timerSteps": [
                        {{"stepNumber": 2, "instruction": "Simmer the sauce for 15 minutes", "duration": 15, "description": "Simmer sauce"}}
                    ],
                    "nutrition": {{
                        "calories": "280 kcal",
                        "carbs": "35g",
                        "protein": "15g",
                        "fat": "10g"
                    }}
                }},
                {{
                    "name": "Recipe Name 3",
                    "description": "Brief description of the dish",
                    "ingredients": [
                        "3 cups ingredient from pantry",
                        "1 tsp herbs",
                        "2 tbsp olive oil"
                    ],
                    "instructions": [
                        "Step 1: Mix ingredients",
                        "Step 2: Bake in oven",
                        "Step 3: Let cool and serve"
                    ],
                    "prepTime": "20 minutes",
                    "cookTime": "45 minutes",
                    "difficulty": "Hard",
                    "servings": 6,
                    "servingNote": "Serving size calculated based on available pantry quantities",
                    "healthRating": "Unhealthy",
                    "dietaryInfo": ["Vegan"],
                    "timerSteps": [
                        {{"stepNumber": 2, "instruction": "Bake in preheated oven", "duration": 45, "description": "Bake dish"}}
                    ],
                    "nutrition": {{
                        "calories": "420 kcal",
                        "carbs": "55g",
                        "protein": "25g",
                        "fat": "15g"
                    }}
                }}
            ]
        }}
        """
        
        if not client:
            raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a creative chef. Return only valid JSON with 3 different recipes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        recipe_text = response.choices[0].message.content.strip()
        
        # Clean up the response
        if recipe_text.startswith("```json"):
            recipe_text = recipe_text.replace("```json", "").replace("```", "").strip()
        elif recipe_text.startswith("```"):
            recipe_text = recipe_text.replace("```", "").strip()
        
        import json
        recipe_data = json.loads(recipe_text)
        
        return jsonify({
            'success': True,
            'recipes': recipe_data['recipes'],
            'pantry_items_used': pantry_items
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to generate recipes: {str(e)}'
        }), 500

@app.route('/api/recipes/fallback', methods=['GET'])
def api_fallback_recipes():
    """Get fallback recipes as JSON"""
    fallback_recipes = [
        {
            "name": "Pasta with Tomato Sauce",
            "description": "A simple and classic pasta dish",
            "ingredients": [
                {"name": "pasta", "amount": "8", "unit": "oz"},
                {"name": "tomato", "amount": "2", "unit": "medium"},
                {"name": "garlic", "amount": "2", "unit": "cloves"},
                {"name": "olive oil", "amount": "2", "unit": "tbsp"}
            ],
            "instructions": [
                "Boil water and cook pasta according to package directions",
                "Heat olive oil in a pan, add minced garlic",
                "Add chopped tomatoes and cook until softened",
                "Season with salt and pepper",
                "Toss cooked pasta with sauce and serve"
            ],
            "prep_time": "10 minutes",
            "cook_time": "15 minutes",
            "difficulty": "Easy",
            "servings": 2
        },
        {
            "name": "Grilled Cheese Sandwich",
            "description": "A comforting classic sandwich",
            "ingredients": [
                {"name": "bread", "amount": "4", "unit": "slices"},
                {"name": "cheese", "amount": "4", "unit": "slices"},
                {"name": "butter", "amount": "2", "unit": "tbsp"}
            ],
            "instructions": [
                "Butter one side of each bread slice",
                "Place cheese between bread slices",
                "Heat a pan over medium heat",
                "Cook sandwich until golden brown on both sides",
                "Serve hot"
            ],
            "prep_time": "5 minutes",
            "cook_time": "8 minutes",
            "difficulty": "Easy",
            "servings": 2
        }
    ]
    
    return jsonify({
        'success': True,
        'recipes': fallback_recipes
    })

@app.route('/api/upload_photo', methods=['POST'])
def api_upload_photo():
    """Upload photo via API (for mobile app)"""
    
    try:
        # Check if request has files
        if 'photo' not in request.files:
            # Also check if data was sent as raw body (for debugging)
            if request.content_type and 'multipart' in request.content_type:
                return jsonify({'success': False, 'error': 'No photo field found in multipart form data'}), 400
            return jsonify({'success': False, 'error': 'No photo uploaded. Content-Type: ' + str(request.content_type)}), 400
        
        photo = request.files['photo']
        if photo.filename == '':
            return jsonify({'success': False, 'error': 'No photo selected'}), 400
        
        # Read file content once
        photo.seek(0)
        img_bytes = photo.read()
        
        # Validate file size
        if len(img_bytes) == 0:
            return jsonify({'success': False, 'error': 'File is empty or could not be read'}), 400
        
        # Validate file size (max 10MB)
        if len(img_bytes) > 10 * 1024 * 1024:
            return jsonify({'success': False, 'error': 'File too large. Maximum size is 10MB'}), 400
        
        # Basic validation: check if it looks like an image (JPEG/PNG magic bytes)
        if len(img_bytes) < 4:
            return jsonify({'success': False, 'error': 'Invalid image file'}), 400
        
        # Check for JPEG magic bytes (FF D8 FF) or PNG magic bytes (89 50 4E 47)
        is_jpeg = img_bytes[:3] == b'\xff\xd8\xff'
        is_png = img_bytes[:4] == b'\x89PNG'
        if not (is_jpeg or is_png):
            return jsonify({'success': False, 'error': 'Invalid image format. Only JPEG and PNG are supported'}), 400
        
        # Send image to OpenAI vision API
        import base64
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        prompt = """Analyze this photo and identify all food items with their quantities. 
For each food item, detect:
1. The quantity (e.g., "2 bottles", "three cans", "1 loaf", "5 slices")
2. The food item name (generic name only, no brand names)

Return the results in this exact JSON format:
{
  "items": [
    {"name": "milk", "quantity": "2 bottles"},
    {"name": "soup", "quantity": "3 cans"},
    {"name": "bread", "quantity": "1 loaf"}
  ]
}

If you cannot determine a quantity, use "1" as the default. Return ONLY valid JSON, no other text."""
        
        if not client:
            return jsonify({
                'success': False,
                'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.'
            }), 500
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a food recognition assistant. Analyze images and identify food items with their quantities. Return results in JSON format with item names and quantities."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]}
            ],
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        food_response = response.choices[0].message.content
        
        # Parse JSON response
        import json
        try:
            food_data = json.loads(food_response)
            detected_items = food_data.get('items', [])
            
            # Convert to list of PantryItem objects
            pantry_items = []
            for item in detected_items:
                name = item.get('name', '').strip()
                quantity = item.get('quantity', '1').strip()
                if name:
                    pantry_items.append({
                        'id': str(uuid.uuid4()),
                        'name': name,
                        'quantity': quantity,
                        'expirationDate': None,
                        'addedDate': datetime.now().isoformat()
                    })
        except json.JSONDecodeError:
            # Fallback: try to parse as comma-separated list (old format)
            detected_items = [f.strip() for f in food_response.split(",") if f.strip()]
            pantry_items = []
            for item in detected_items:
                pantry_items.append({
                    'id': str(uuid.uuid4()),
                    'name': item,
                    'quantity': '1',
                    'expirationDate': None,
                    'addedDate': datetime.now().isoformat()
                })
        
        # Add to appropriate pantry based on client type and user authentication
        client_type = request.headers.get('X-Client-Type', 'web')
        user_id = request.headers.get('X-User-ID')
        
        total_items = 0
        if user_id:
            # Add to user's pantry
            user_pantry = get_user_pantry(user_id)
            # Convert detected_items to proper format (list of dicts)
            for item in pantry_items:
                user_pantry.append(item)
            update_user_pantry(user_id, user_pantry)
            total_items = len(user_pantry)
        else:
            # Add to anonymous pantry
            global mobile_pantry, web_pantry  # Declare globals at function start
            if client_type == 'mobile':
                if not isinstance(mobile_pantry, list):
                    mobile_pantry = []
                for item in pantry_items:
                    mobile_pantry.append(item)
                total_items = len(mobile_pantry)
            else:
                # Add to anonymous web pantry (stored in session)
                # CRITICAL: Mark session as permanent for anonymous users
                session.permanent = True
                
                if 'web_pantry' not in session:
                    session['web_pantry'] = []
                for item in pantry_items:
                    session['web_pantry'].append(item)
                # Mark session as modified to ensure it's saved
                session.modified = True
                total_items = len(session.get('web_pantry', []))
        
        # Return item names as strings for backward compatibility with iOS app
        item_names = [item['name'] for item in pantry_items]
        
        return jsonify({
            'success': True,
            'message': f'Successfully analyzed photo! Added {len(pantry_items)} items',
            'items': item_names,  # Return as list of strings for iOS compatibility
            'total_items': total_items
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        if VERBOSE_LOGGING:
            print(f"❌ Error in api_upload_photo: {str(e)}")
            print(f"   Traceback: {error_trace}")
        return jsonify({
            'success': False,
            'error': f'Error analyzing photo: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint"""
    client_type = request.headers.get('X-Client-Type', 'web')
    user_id = request.headers.get('X-User-ID')
    
    # Get appropriate pantry
    if user_id:
        pantry_to_use = get_user_pantry(user_id)
    else:
        pantry_to_use = mobile_pantry if client_type == 'mobile' else web_pantry
    
    # Ensure pantry_to_use is a list
    if not isinstance(pantry_to_use, list):
        pantry_to_use = []
    
    return jsonify({
        'success': True,
        'status': 'healthy',
        'pantry_items': len(pantry_to_use),
        'ai_available': bool(client and api_key)
    })

# Note: handler is exported in api/index.py for Vercel serverless functions
# Do not export handler here to avoid conflicts with Vercel's handler detection

if __name__ == "__main__":
    # On startup, check for old users and migrate them
    print("=" * 60)
    print("🚀 Starting Smart Pantry Server")
    print("=" * 60)
    print(f"📁 USERS_FILE location: {USERS_FILE}")
    print(f"   Absolute path: {os.path.abspath(USERS_FILE)}")
    print(f"   IS_VERCEL: {IS_VERCEL}, IS_RENDER: {IS_RENDER}")
    
    # Load users to trigger migration if needed
    initial_users = load_users()
    print(f"📊 Initial user count: {len(initial_users)}")
    if initial_users:
        print(f"   User IDs: {list(initial_users.keys())[:5]}...")  # Show first 5
    print("=" * 60)
    
    # Get port from environment variable (Render sets this) or use default
    port = int(os.getenv('PORT', 5050))
    # host='0.0.0.0' allows connections from other devices on the network
    app.run(debug=False, host='0.0.0.0', port=port)
