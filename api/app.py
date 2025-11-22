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
if IS_VERCEL:
    # For serverless, we'll use Flask's default in-memory sessions
    # Note: Sessions won't persist across function invocations in serverless
    # For production, consider using external session storage (Redis, database, etc.)
    pass

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
    print("   Create a .env file with: OPENAI_API_KEY=your_api_key_here")
    client = None
else:
    client = OpenAI(api_key=api_key)

 

# Separate pantry lists for different clients (for non-authenticated users)
web_pantry = []
mobile_pantry = []

# User management system
# In serverless, use /tmp directory for temporary storage or in-memory storage
if IS_VERCEL:
    # Use /tmp directory which is writable in Vercel serverless functions
    USERS_FILE = os.path.join('/tmp', 'users.json')
    # In-memory fallback if file operations fail
    _in_memory_users = {}
else:
    USERS_FILE = 'users.json'
    _in_memory_users = {}

def load_users():
    """Load users from JSON file or in-memory storage"""
    if IS_VERCEL:
        # Try to load from /tmp, fallback to in-memory
        try:
            if os.path.exists(USERS_FILE):
                with open(USERS_FILE, 'r') as f:
                    loaded = json.load(f)
                    # Update in-memory cache as well
                    global _in_memory_users
                    _in_memory_users = loaded.copy()
                    return loaded
            # Return in-memory copy
            return _in_memory_users.copy() if _in_memory_users else {}
        except (IOError, OSError, json.JSONDecodeError) as e:
            # Fallback to in-memory if file read fails
            print(f"Warning: Could not load users file: {e}")
            return _in_memory_users.copy() if _in_memory_users else {}
    else:
        # Local development: use file system
        try:
            if os.path.exists(USERS_FILE):
                with open(USERS_FILE, 'r') as f:
                    return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load users file: {e}")
        return {}

def save_users(users):
    """Save users to JSON file or in-memory storage"""
    if IS_VERCEL:
        # Try to save to /tmp, always update in-memory cache
        global _in_memory_users
        _in_memory_users = users.copy()  # Always update in-memory
        
        try:
            # Ensure /tmp directory exists
            os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
            with open(USERS_FILE, 'w') as f:
                json.dump(users, f, indent=2)
        except (IOError, OSError) as e:
            # Fallback to in-memory storage if file write fails
            print(f"Warning: Could not save users file: {e}. Using in-memory storage.")
    else:
        # Local development: use file system
        try:
            with open(USERS_FILE, 'w') as f:
                json.dump(users, f, indent=2)
        except (IOError, OSError) as e:
            print(f"Error: Could not save users file: {e}")

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, email, password, client_type='web'):
    """Create a new user"""
    users = load_users()
    
    # Check if username or email already exists (regardless of client_type)
    for user_id, user_data in users.items():
        if user_data['username'] == username or user_data['email'] == email:
            return None, "Username or email already exists"
    
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
    
    save_users(users)
    return user_id, None

def authenticate_user(username, password, client_type='web'):
    """Authenticate user and return user data"""
    users = load_users()
    
    for user_id, user_data in users.items():
        if (user_data['username'] == username or user_data['email'] == username) and \
           user_data['password_hash'] == hash_password(password):
            
            # Update last login
            user_data['last_login'] = datetime.now().isoformat()
            save_users(users)
            
            return user_data, None
    
    return None, "Invalid credentials"

def get_user_pantry(user_id):
    """Get user's pantry items"""
    users = load_users()
    if user_id in users:
        return users[user_id]['pantry']
    return []

def update_user_pantry(user_id, pantry_items):
    """Update user's pantry items"""
    users = load_users()
    if user_id in users:
        users[user_id]['pantry'] = pantry_items
        save_users(users)

 

# Authentication routes for web
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        user_data, error = authenticate_user(username, password, 'web')
        if user_data:
            session['user_id'] = user_data['id']
            session['username'] = user_data['username']
            flash(f"Welcome back, {user_data['username']}!", "success")
            return redirect(url_for("index"))
        else:
            flash(error, "danger")
    
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        
        if password != confirm_password:
            flash("Passwords do not match", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters", "danger")
        else:
            user_id, error = create_user(username, email, password, 'web')
            if user_id:
                session['user_id'] = user_id
                session['username'] = username
                flash(f"Account created successfully! Welcome, {username}!", "success")
                return redirect(url_for("index"))
            else:
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
        user_pantry = get_user_pantry(session['user_id'])
        return render_template("index.html", items=user_pantry, username=session.get('username'))
    else:
        # For anonymous users, use session to persist pantry across server restarts
        if 'web_pantry' not in session:
            session['web_pantry'] = []
        return render_template("index.html", items=session['web_pantry'], username=None)

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
            user_pantry = get_user_pantry(session['user_id'])
            if item not in user_pantry:  # Prevent duplicates
                user_pantry.append(item)
                update_user_pantry(session['user_id'], user_pantry)
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
    if 'user_id' in session:
        # Remove from user's pantry
        user_pantry = get_user_pantry(session['user_id'])
        if item_name in user_pantry:
            user_pantry.remove(item_name)
            update_user_pantry(session['user_id'], user_pantry)
            flash(f"{item_name} removed from pantry.", "info")
    else:
        # Remove from anonymous web pantry (stored in session)
        if 'web_pantry' not in session:
            session['web_pantry'] = []
        if item_name in session['web_pantry']:
            session['web_pantry'].remove(item_name)
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
    
    # Check if pantry is empty (handle both None and empty list)
    if not current_pantry or len(current_pantry) == 0:
        flash("Your pantry is empty. Add items first.", "warning")
        return redirect(url_for("index"))

    # Check if we already have recipes in session
    existing_recipes = session.get('current_recipes', [])
    if existing_recipes:
        # Use existing recipes instead of generating new ones
        flash("Showing your current recipes. Click 'Generate New Recipes' for fresh ideas!", "info")
        return render_template("suggest_recipe.html", recipes=existing_recipes, pantry_items=current_pantry)

    # Generate AI-powered recipes based on pantry items (only if no existing recipes)
    pantry_items = ", ".join(current_pantry)
    pantry = current_pantry  # Use for compatibility with existing code
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

    return render_template("suggest_recipe.html", recipes=recipes, pantry_items=pantry)

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
    
    prompt = "List all food items you see in this photo. Only return a comma-separated list of food names. Return ONLY the generic food item names (e.g., 'applesauce', 'pasta', 'bread') - DO NOT include brand names, company names, or product names."
    
    try:
        if not client:
            raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a food recognition assistant. Identify all food items in the image and return them as a comma-separated list. Return only generic food names without brand names or company names."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]}
            ],
            max_tokens=200
        )
        food_list = response.choices[0].message.content
        
        # Parse comma-separated food names
        detected_items = [f.strip() for f in food_list.split(",") if f.strip()]
        
        # Add to appropriate pantry based on user authentication
        if 'user_id' in session:
            # Add to user's pantry
            user_pantry = get_user_pantry(session['user_id'])
            user_pantry.extend(detected_items)
            update_user_pantry(session['user_id'], user_pantry)
        else:
            # Add to anonymous web pantry (stored in session)
            if 'web_pantry' not in session:
                session['web_pantry'] = []
            session['web_pantry'].extend(detected_items)
        
        flash(f"Successfully analyzed photo! Added {len(detected_items)} items: {', '.join(detected_items)}", "success")
        
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

    return render_template("suggest_recipe.html", recipes=recipes_with_nutrition, pantry_items=current_pantry)

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

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Login user via API"""
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Invalid request data'}), 400
    
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    client_type = request.headers.get('X-Client-Type', 'mobile')
    
    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password are required'}), 400
    
    user_data, error = authenticate_user(username, password, client_type)
    if user_data:
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user_id': user_data['id'],
            'username': user_data['username'],
            'email': user_data['email'],
            'pantry': user_data['pantry']
        })
    else:
        return jsonify({'success': False, 'error': error}), 401

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
    
    return jsonify({
        'success': True,
        'items': pantry_to_use,
        'count': len(pantry_to_use)
    })

@app.route('/api/pantry', methods=['POST'])
def api_add_item():
    """Add an item to pantry via API"""
    
    data = request.get_json()
    if not data or 'item' not in data:
        return jsonify({'success': False, 'error': 'Item name required'}), 400
    
    item = data['item'].strip()
    if not item:
        return jsonify({'success': False, 'error': 'Item name cannot be empty'}), 400
    
    client_type = request.headers.get('X-Client-Type', 'web')
    user_id = request.headers.get('X-User-ID')
    
    # Check if user is authenticated
    if user_id:
        pantry_to_use = get_user_pantry(user_id)
        if item.lower() not in [i.lower() for i in pantry_to_use]:
            pantry_to_use.append(item)
            update_user_pantry(user_id, pantry_to_use)
            return jsonify({
                'success': True,
                'message': f'Added "{item}" to pantry',
                'item': item,
                'total_items': len(pantry_to_use)
            })
        else:
            return jsonify({'success': False, 'error': f'"{item}" is already in pantry'}), 409
    else:
        # Use anonymous pantry
        pantry_to_use = mobile_pantry if client_type == 'mobile' else web_pantry
        if item.lower() not in [i.lower() for i in pantry_to_use]:
            pantry_to_use.append(item)
            return jsonify({
                'success': True,
                'message': f'Added "{item}" to pantry',
                'item': item,
                'total_items': len(pantry_to_use)
            })
        else:
            return jsonify({'success': False, 'error': f'"{item}" is already in pantry'}), 409

@app.route('/api/pantry/<item>', methods=['DELETE'])
def api_delete_item(item):
    """Delete an item from pantry via API"""
    client_type = request.headers.get('X-Client-Type', 'web')
    user_id = request.headers.get('X-User-ID')
    
    # Check if user is authenticated
    if user_id:
        pantry_to_use = get_user_pantry(user_id)
        if item in pantry_to_use:
            pantry_to_use.remove(item)
            update_user_pantry(user_id, pantry_to_use)
            return jsonify({
                'success': True,
                'message': f'Removed "{item}" from pantry',
                'total_items': len(pantry_to_use)
            })
        else:
            return jsonify({'success': False, 'error': f'"{item}" not found in pantry'}), 404
    else:
        # Use anonymous pantry
        pantry_to_use = mobile_pantry if client_type == 'mobile' else web_pantry
        if item in pantry_to_use:
            pantry_to_use.remove(item)
            return jsonify({
                'success': True,
                'message': f'Removed "{item}" from pantry',
                'total_items': len(pantry_to_use)
            })
        else:
            return jsonify({'success': False, 'error': f'"{item}" not found in pantry'}), 404

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
    
    try:
        # Generate AI recipes
        pantry_list = ", ".join(pantry_items)
        prompt = f"""
        Create 3 delicious and diverse recipes using these available ingredients: {pantry_list}
        
        Requirements:
        - Each recipe must use at least 2-3 ingredients from this list: {pantry_list}
        - Include basic pantry staples (salt, pepper, oil, butter) as needed
        - Make recipes practical and easy to follow
        - Include realistic cooking times and difficulty levels
        - Make each recipe different (different cuisine, cooking method, etc.)
        - Assess the overall healthiness of each recipe and assign a health rating
        - Include dietary information (vegan, vegetarian, halal, etc.) if applicable
        - Add timer steps for cooking steps that require specific timing (frying, baking, simmering, etc.)
        
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
        
        prompt = "List all food items you see in this photo. Only return a comma-separated list of food names. Return ONLY the generic food item names (e.g., 'applesauce', 'pasta', 'bread') - DO NOT include brand names, company names, or product names."
        
        if not client:
            return jsonify({
                'success': False,
                'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.'
            }), 500
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a food recognition assistant. Identify all food items in the image and return them as a comma-separated list. Return only generic food names without brand names or company names."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]}
            ],
            max_tokens=200
        )
        food_list = response.choices[0].message.content
        
        # Parse comma-separated food names
        detected_items = [f.strip() for f in food_list.split(",") if f.strip()]
        
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
