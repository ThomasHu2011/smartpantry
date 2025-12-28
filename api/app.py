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

# Use environment variable for secret key if available, otherwise use default
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')  # Needed for flash messages

# Configure Flask for serverless environment
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching in serverless (better for debugging)
if IS_VERCEL:
    app.config['TESTING'] = False
    app.config['DEBUG'] = False  # Disable debug in production
else:
    app.config['DEBUG'] = True

# Add global error handler for better debugging in serverless
@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler - logs errors for debugging in Vercel"""
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
        is_api_request = (hasattr(request, 'path') and request.path.startswith('/api/')) or \
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
    # Local development: use current directory
    USERS_FILE = 'users.json'
    _in_memory_users = {}

def load_users():
    """Load users from JSON file or in-memory storage"""
    if IS_VERCEL or IS_RENDER:
        # Try to load from /tmp, fallback to in-memory
        users = {}
        
        # First, try to load from new location (/tmp/users.json)
        try:
            if os.path.exists(USERS_FILE):
                with open(USERS_FILE, 'r') as f:
                    users = json.load(f)
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
                        global _in_memory_users
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
        global _in_memory_users
        _in_memory_users = users.copy()
        
        if not users:
            print(f"No users found, using in-memory storage")
            return _in_memory_users.copy() if _in_memory_users else {}
        
        return users
    else:
        # Local development: use file system
        try:
            if os.path.exists(USERS_FILE):
                with open(USERS_FILE, 'r') as f:
                    loaded = json.load(f)
                    print(f"Loaded {len(loaded)} users from {USERS_FILE}")
                    return loaded
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load users file: {e}")
        return {}

def save_users(users):
    """Save users to JSON file or in-memory storage"""
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
            print(f"Saved {len(users)} users to {USERS_FILE}")
        except (IOError, OSError) as e:
            # Fallback to in-memory storage if file write fails
            print(f"Warning: Could not save users file: {e}. Using in-memory storage.")
    else:
        # Local development: use file system with atomic write
        try:
            # Use atomic write: write to temp file first, then rename
            temp_file = USERS_FILE + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(users, f, indent=2)
                f.flush()  # Force write to disk
                os.fsync(f.fileno())  # Ensure data is written to disk
            
            # Atomic rename
            os.replace(temp_file, USERS_FILE)
            print(f"Saved {len(users)} users to {USERS_FILE}")
        except (IOError, OSError) as e:
            print(f"Error: Could not save users file: {e}")

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
    save_users(users)
    
    # Verify the save was successful by reloading
    # This ensures the file is actually written before returning
    verify_users = load_users()
    if user_id not in verify_users:
        print(f"Warning: User {user_id} not found after save, retrying...")
        save_users(users)  # Retry once
    
    print(f"User created successfully: {username} (ID: {user_id})")
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
    
    # Debug: print all usernames (without passwords)
    usernames = [user_data.get('username', 'N/A') for user_data in users.values()]
    print(f"Available usernames: {usernames}")
    
    password_hash = hash_password(password)
    print(f"Password hash for provided password: {password_hash[:20]}...")
    
    # Track if we found a matching username but wrong password
    found_username = False
    
    for user_id, user_data in users.items():
        stored_username = user_data.get('username', '').strip()
        stored_email = user_data.get('email', '').strip()
        stored_password_hash = user_data.get('password_hash', '')
        
        # Case-insensitive matching for username and email
        username_match = (
            stored_username.lower() == username_normalized or 
            stored_email.lower() == username_normalized
        )
        password_match = stored_password_hash == password_hash
        
        print(f"Checking user {user_id}: username='{stored_username}', email='{stored_email}', username_match={username_match}, password_match={password_match}")
        
        if username_match and password_match:
            # Update last login
            user_data['last_login'] = datetime.now().isoformat()
            save_users(users)
            
            print(f"✅ Authentication successful for user '{username}' (ID: {user_id})")
            return user_data, None
        elif username_match and not password_match:
            found_username = True
            print(f"❌ Password mismatch for user '{username}' - stored hash: {stored_password_hash[:20]}..., provided hash: {password_hash[:20]}...")
    
    # Provide specific error messages
    if found_username:
        print(f"❌ Authentication failed: Incorrect password for user '{username}'")
        return None, "Incorrect password. Please try again."
    else:
        print(f"❌ Authentication failed: Username or email '{username}' not found")
        return None, "Incorrect username or email. Please check and try again."

def get_user_pantry(user_id):
    """Get user's pantry items"""
    users = load_users()
    if user_id in users:
        pantry = users[user_id].get('pantry', [])
        # Return a copy of the list to avoid reference issues
        pantry_copy = list(pantry) if pantry else []
        print(f"Retrieved pantry for user {user_id}: {len(pantry_copy)} items")
        return pantry_copy
    print(f"Warning: User {user_id} not found in users database")
    return []

def update_user_pantry(user_id, pantry_items):
    """Update user's pantry items"""
    users = load_users()
    if user_id in users:
        users[user_id]['pantry'] = pantry_items
        save_users(users)
        print(f"Updated pantry for user {user_id}: {len(pantry_items)} items saved")
    else:
        print(f"Error: Cannot update pantry - user {user_id} not found in users database")

 

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

# Home page (pantry list)
@app.route("/")
def index():
    # Check if user is logged in
    if 'user_id' in session:
        user_id = session['user_id']
        user_pantry = get_user_pantry(user_id)
        # Convert to string list for web template (backward compatibility)
        items_list = []
        for item in user_pantry:
            if isinstance(item, dict):
                items_list.append(item.get('name', ''))
            else:
                items_list.append(item)
        print(f"Loading index page for user {user_id}, pantry has {len(items_list)} items: {items_list}")
        return render_template("index.html", items=items_list, username=session.get('username'))
    else:
        # For anonymous users, use session to persist pantry across server restarts
        if 'web_pantry' not in session:
            session['web_pantry'] = []
        # Convert to string list for web template
        items_list = []
        for item in session.get('web_pantry', []):
            if isinstance(item, dict):
                items_list.append(item.get('name', ''))
            else:
                items_list.append(item)
        return render_template("index.html", items=items_list, username=None)

# Add items
@app.route("/add", methods=["POST"])
def add_items():
    item = request.form.get("item")
    if item:
        # Sanitize and validate input
        item = item.strip()
        if len(item) > 100:  # Prevent extremely long items
            flash("Item name too long. Please keep it under 100 characters.", "danger")
            return redirect(url_for("index"))
        
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
            for pantry_item in user_pantry:
                if isinstance(pantry_item, dict):
                    pantry_list.append(pantry_item)
                    if pantry_item.get('name', '').lower() == item.lower():
                        item_exists = True
                else:
                    pantry_list.append({
                        'id': str(uuid.uuid4()),
                        'name': pantry_item,
                        'quantity': '1',
                        'expirationDate': None,
                        'addedDate': datetime.now().isoformat()
                    })
                    if pantry_item.lower() == item.lower():
                        item_exists = True
            
            if not item_exists:
                # Add new item
                pantry_list.append({
                    'id': str(uuid.uuid4()),
                    'name': item,
                    'quantity': '1',
                    'expirationDate': None,
                    'addedDate': datetime.now().isoformat()
                })
                update_user_pantry(user_id, pantry_list)
                flash(f"{item} added to pantry.", "success")
            else:
                flash(f"{item} is already in your pantry.", "warning")
        else:
            # Add to anonymous web pantry (stored in session for persistence)
            if 'web_pantry' not in session:
                session['web_pantry'] = []
            if item not in session['web_pantry']:  # Prevent duplicates
                session['web_pantry'].append(item)
                flash(f"{item} added to pantry.", "success")
            else:
                flash(f"{item} is already in your pantry.", "warning")
    return redirect(url_for("index"))

# Delete item
@app.route("/delete/<item_name>")
def delete_item(item_name):
    from urllib.parse import unquote
    item_name = unquote(item_name)
    
    if 'user_id' in session:
        # Remove from user's pantry
        user_pantry = get_user_pantry(session['user_id'])
        # Convert to list of dicts if needed
        pantry_list = []
        item_found = False
        for pantry_item in user_pantry:
            if isinstance(pantry_item, dict):
                if pantry_item.get('name', '').lower() != item_name.lower():
                    pantry_list.append(pantry_item)
                else:
                    item_found = True
            else:
                if pantry_item.lower() != item_name.lower():
                    pantry_list.append({
                        'id': str(uuid.uuid4()),
                        'name': pantry_item,
                        'quantity': '1',
                        'expirationDate': None,
                        'addedDate': datetime.now().isoformat()
                    })
                else:
                    item_found = True
        
        if item_found:
            update_user_pantry(session['user_id'], pantry_list)
            flash(f"{item_name} removed from pantry.", "info")
    else:
        # Remove from anonymous web pantry (stored in session)
        if 'web_pantry' not in session:
            session['web_pantry'] = []
        # Convert to list of dicts if needed
        pantry_list = []
        item_found = False
        for pantry_item in session['web_pantry']:
            if isinstance(pantry_item, dict):
                if pantry_item.get('name', '').lower() != item_name.lower():
                    pantry_list.append(pantry_item)
                else:
                    item_found = True
            else:
                if pantry_item.lower() != item_name.lower():
                    pantry_list.append(pantry_item)
                else:
                    item_found = True
        
        if item_found:
            session['web_pantry'] = pantry_list
            flash(f"{item_name} removed from pantry.", "info")
    return redirect(url_for("index"))

# Suggest recipes based on pantry
@app.route("/suggest")
def suggest_recipe():
    # Get appropriate pantry based on login status
    if 'user_id' in session:
        current_pantry = get_user_pantry(session['user_id'])
    else:
        # Use session-based pantry for anonymous users
        if 'web_pantry' not in session:
            session['web_pantry'] = []
        current_pantry = session['web_pantry']
    
    # Convert pantry to string list for template compatibility
    pantry_items_list = []
    for pantry_item in current_pantry:
        if isinstance(pantry_item, dict):
            pantry_items_list.append(pantry_item.get('name', ''))
        else:
            pantry_items_list.append(pantry_item)
    
    # Check if pantry is empty (handle both None and empty list)
    if not pantry_items_list or len(pantry_items_list) == 0:
        flash("Your pantry is empty. Add items first.", "warning")
        return redirect(url_for("index"))

    # Check if we already have recipes in session
    existing_recipes = session.get('current_recipes', [])
    if existing_recipes:
        # Use existing recipes instead of generating new ones
        flash("Showing your current recipes. Click 'Generate New Recipes' for fresh ideas!", "info")
        return render_template("suggest_recipe.html", recipes=existing_recipes, pantry_items=pantry_items_list)

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
        if not client or not api_key:
            raise ValueError("OpenAI client not properly initialized")
            
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a creative chef and recipe developer. Create practical, delicious recipes using available ingredients."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
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
            recipes = [{
                "name": "Pantry Surprise",
                "ingredients": pantry[:5],  # Use first 5 pantry items
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
            pantry_count = sum(1 for ingredient in recipe['ingredients'] if any(pantry_item.lower() in ingredient.lower() for pantry_item in pantry))
            total_ingredients = len(recipe['ingredients'])
            pantry_percentage = (pantry_count / total_ingredients) * 100 if total_ingredients > 0 else 0
            
            if pantry_percentage >= 50:
                validated_recipes.append(recipe)
            else:
                # If recipe doesn't meet 50% requirement, try to adjust it
                flash(f"Recipe '{recipe['name']}' only uses {pantry_percentage:.0f}% pantry ingredients. Adjusting...", "warning")
                # Add more pantry ingredients to meet requirement
                needed_pantry = max(1, (total_ingredients // 2) - pantry_count)
                additional_pantry = pantry[:needed_pantry]
                recipe['ingredients'].extend(additional_pantry)
                validated_recipes.append(recipe)
        
        recipes = validated_recipes
        
        flash(f"Generated {len(recipes)} recipe(s) using at least 50% pantry ingredients!", "success")
        
        # Store recipes in session for nutrition info lookup
        session['current_recipes'] = recipes
        
    except Exception as e:
        flash(f"AI unavailable: {str(e)}. Using fallback recipes.", "warning")
        # Fallback to simple recipe suggestions
        recipes = [
            {
                "name": f"Simple {pantry[0].title()} Dish" if pantry else "Basic Recipe",
                "ingredients": pantry[:3] + ["salt", "pepper", "oil"],
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
            },
            {
                "name": f"Quick {pantry[1].title()} Recipe" if len(pantry) > 1 else "Quick Pantry Meal",
                "ingredients": pantry[1:4] + ["garlic", "herbs"],
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
            },
            {
                "name": "Pantry Fusion",
                "ingredients": pantry[:5] + ["spices"],
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
            }
        ]
        
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
    if not photo:
        flash("No photo uploaded.", "danger")
        return redirect(url_for("index"))
    
    # Save photo to uploads folder
    # In serverless, use /tmp directory which is writable
    if IS_VERCEL:
        upload_folder = '/tmp/uploads'
    else:
        upload_folder = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    photo_path = os.path.join(upload_folder, photo.filename)
    photo.save(photo_path)

    # Send image to OpenAI vision API
    import base64
    photo.seek(0)
    img_bytes = photo.read()
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
            
            # Add new items
            pantry_list.extend(pantry_items)
            update_user_pantry(session['user_id'], pantry_list)
        else:
            # Add to anonymous web pantry (stored in session for persistence)
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
            
            # Add new items
            pantry_list.extend(pantry_items)
            session['web_pantry'] = pantry_list
        
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
    for pantry_item in current_pantry:
        if isinstance(pantry_item, dict):
            pantry_items_list.append(pantry_item.get('name', ''))
        else:
            pantry_items_list.append(pantry_item)

    return render_template("suggest_recipe.html", recipes=recipes_with_nutrition, pantry_items=pantry_items_list)

# Detailed recipe page with timers and full instructions
@app.route("/recipe/<recipe_name>")
def recipe_detail(recipe_name):
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

    # Generate detailed cooking instructions with timers using AI
    ingredients_text = ", ".join(recipe['ingredients'])
    instructions_text = " | ".join(recipe['instructions'])
    
    prompt = f"""For this recipe: {recipe['name']}
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
        
    except Exception as e:
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

    # Generate nutrition facts for this recipe
    nutrition_info = None
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

    except Exception as e:
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

    return render_template("recipe_detail.html", recipe=recipe, detailed_recipe=detailed_recipe, nutrition=nutrition_info)

# =============================================================================
# API ENDPOINTS FOR MOBILE/FRONTEND APP
# =============================================================================

# Authentication API endpoints
@app.route('/api/auth/signup', methods=['POST'])
def api_signup():
    """Sign up a new user via API"""
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Invalid request data'}), 400
    
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()
    client_type = request.headers.get('X-Client-Type', 'mobile')
    
    if not username or not email or not password:
        return jsonify({'success': False, 'error': 'Username, email, and password are required'}), 400
    
    if len(password) < 6:
        return jsonify({'success': False, 'error': 'Password must be at least 6 characters'}), 400
    
    user_id, error = create_user(username, email, password, client_type)
    if user_id:
        return jsonify({
            'success': True,
            'message': 'Account created successfully',
            'user_id': user_id,
            'username': username
        })
    else:
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
    
    client_type = request.headers.get('X-Client-Type', 'web')
    user_id = request.headers.get('X-User-ID')
    
    # Check if user is authenticated
    if user_id:
        pantry_to_use = get_user_pantry(user_id)
    else:
        # Use anonymous pantry
        pantry_to_use = mobile_pantry if client_type == 'mobile' else web_pantry
    
    # Convert pantry items to new format if needed (backward compatibility)
    items = []
    for item in pantry_to_use:
        if isinstance(item, dict):
            # Already in new format
            items.append(item)
        else:
            # Old string format - convert to new format
            items.append({
                'id': str(uuid.uuid4()),
                'name': item,
                'quantity': '1',
                'expirationDate': None,
                'addedDate': datetime.now().isoformat()
            })
    
    return jsonify({
        'success': True,
        'items': items,
        'count': len(items)
    })

@app.route('/api/pantry', methods=['POST'])
def api_add_item():
    """Add an item to pantry via API"""
    
    data = request.get_json()
    if not data:
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
        pantry_item = {
            'id': data.get('id', str(uuid.uuid4())),
            'name': data['name'].strip(),
            'quantity': data.get('quantity', '1'),
            'expirationDate': data.get('expirationDate'),
            'addedDate': data.get('addedDate', datetime.now().isoformat())
        }
        
        if not pantry_item['name']:
            return jsonify({'success': False, 'error': 'Item name cannot be empty'}), 400
    else:
        return jsonify({'success': False, 'error': 'Item name required'}), 400
    
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
        if any(i.get('name', '').lower() == pantry_item['name'].lower() for i in pantry_list):
            return jsonify({'success': False, 'error': f'"{pantry_item["name"]}" is already in pantry'}), 409
        
        pantry_list.append(pantry_item)
        update_user_pantry(user_id, pantry_list)
        return jsonify({
            'success': True,
            'message': f'Added "{pantry_item["name"]}" to pantry',
            'item': pantry_item,
            'total_items': len(pantry_list)
        })
    else:
        # Use anonymous pantry
        pantry_to_use = mobile_pantry if client_type == 'mobile' else web_pantry
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
        
        # Check for duplicates
        if any(i.get('name', '').lower() == pantry_item['name'].lower() for i in pantry_list):
            return jsonify({'success': False, 'error': f'"{pantry_item["name"]}" is already in pantry'}), 409
        
        pantry_list.append(pantry_item)
        if client_type == 'mobile':
            global mobile_pantry
            mobile_pantry = pantry_list
        else:
            global web_pantry
            web_pantry = pantry_list
        
        return jsonify({
            'success': True,
            'message': f'Added "{pantry_item["name"]}" to pantry',
            'item': pantry_item,
            'total_items': len(pantry_list)
        })

@app.route('/api/pantry/<item_id>', methods=['DELETE'])
def api_delete_item(item_id):
    """Delete an item from pantry via API (by ID or name for backward compatibility)"""
    client_type = request.headers.get('X-Client-Type', 'web')
    user_id = request.headers.get('X-User-ID')
    
    # URL decode the item_id
    from urllib.parse import unquote
    item_id = unquote(item_id)
    
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
        
        # Find item by ID or name (for backward compatibility)
        item_to_delete = None
        for i, pantry_item in enumerate(pantry_list):
            if (isinstance(pantry_item, dict) and 
                (pantry_item.get('id') == item_id or pantry_item.get('name', '').lower() == item_id.lower())):
                item_to_delete = pantry_list.pop(i)
                break
            elif not isinstance(pantry_item, dict) and pantry_item.lower() == item_id.lower():
                item_to_delete = pantry_list.pop(i)
                break
        
        if item_to_delete:
            update_user_pantry(user_id, pantry_list)
            item_name = item_to_delete.get('name', item_id) if isinstance(item_to_delete, dict) else item_to_delete
            return jsonify({
                'success': True,
                'message': f'Removed "{item_name}" from pantry',
                'total_items': len(pantry_list)
            })
        else:
            return jsonify({'success': False, 'error': f'Item not found in pantry'}), 404
    else:
        # Use anonymous pantry
        pantry_to_use = mobile_pantry if client_type == 'mobile' else web_pantry
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
        
        # Find item by ID or name
        item_to_delete = None
        for i, pantry_item in enumerate(pantry_list):
            if (isinstance(pantry_item, dict) and 
                (pantry_item.get('id') == item_id or pantry_item.get('name', '').lower() == item_id.lower())):
                item_to_delete = pantry_list.pop(i)
                break
            elif not isinstance(pantry_item, dict) and pantry_item.lower() == item_id.lower():
                item_to_delete = pantry_list.pop(i)
                break
        
        if item_to_delete:
            if client_type == 'mobile':
                global mobile_pantry
                mobile_pantry = pantry_list
            else:
                global web_pantry
                web_pantry = pantry_list
            item_name = item_to_delete.get('name', item_id) if isinstance(item_to_delete, dict) else item_to_delete
            return jsonify({
                'success': True,
                'message': f'Removed "{item_name}" from pantry',
                'total_items': len(pantry_list)
            })
        else:
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
    
    if 'photo' not in request.files:
        return jsonify({'success': False, 'error': 'No photo uploaded'}), 400
    
    photo = request.files['photo']
    if photo.filename == '':
        return jsonify({'success': False, 'error': 'No photo selected'}), 400
    
    try:
        # Send image to OpenAI vision API
        import base64
        photo.seek(0)
        img_bytes = photo.read()
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
        
        if user_id:
            # Add to user's pantry
            user_pantry = get_user_pantry(user_id)
            user_pantry.extend(detected_items)
            update_user_pantry(user_id, user_pantry)
        else:
            # Add to anonymous pantry
            if client_type == 'mobile':
                global mobile_pantry
                mobile_pantry.extend(detected_items)
            else:
                # Add to anonymous web pantry (stored in session)
                if 'web_pantry' not in session:
                    session['web_pantry'] = []
                session['web_pantry'].extend(detected_items)
        
        return jsonify({
            'success': True,
            'message': f'Successfully analyzed photo! Added {len(detected_items)} items',
            'items': detected_items,
            'total_items': len(user_pantry) if user_id else len(mobile_pantry if client_type == 'mobile' else web_pantry)
        })
        
    except Exception as e:
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
    
    return jsonify({
        'success': True,
        'status': 'healthy',
        'pantry_items': len(pantry_to_use),
        'ai_available': bool(client and api_key)
    })

# Export handler for Vercel serverless functions
# This is required for Vercel to properly invoke the Flask app
handler = app

if __name__ == "__main__":
    # Get port from environment variable (Render sets this) or use default
    port = int(os.getenv('PORT', 5050))
    # host='0.0.0.0' allows connections from other devices on the network
    app.run(debug=False, host='0.0.0.0', port=port)
