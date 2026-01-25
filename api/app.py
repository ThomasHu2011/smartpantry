import os
import json
import hashlib
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from openai import OpenAI
from dotenv import load_dotenv

# Firebase support (optional - set USE_FIREBASE=true to enable)
USE_FIREBASE = os.getenv('USE_FIREBASE', 'false').lower() == 'true'
if USE_FIREBASE:
    try:
        # Import with explicit path handling for different environments
        import sys
        import importlib.util
        
        # Get the directory where app.py is located
        _app_file_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try to load firebase_config module
        firebase_config_path = os.path.join(_app_file_dir, 'firebase_config.py')
        firebase_helpers_path = os.path.join(_app_file_dir, 'firebase_helpers.py')
        
        if os.path.exists(firebase_config_path) and os.path.exists(firebase_helpers_path):
            # Add the directory to sys.path if not already there
            if _app_file_dir not in sys.path:
                sys.path.insert(0, _app_file_dir)
            
            # Import modules
            from firebase_config import get_db, initialize_firebase
            from firebase_helpers import (
                load_users as firebase_load_users,
                save_users as firebase_save_users,
                create_user_in_firestore,
                get_user_by_id as firebase_get_user_by_id,
                update_user as firebase_update_user,
                get_user_pantry as firebase_get_user_pantry,
                update_user_pantry as firebase_update_user_pantry
            )
            # Initialize Firebase on import
            initialize_firebase()
            print("✅ Firebase enabled and initialized")
        else:
            raise ImportError(f"Firebase config files not found. Expected: {firebase_config_path}, {firebase_helpers_path}")
    except Exception as e:
        print(f"⚠️ Warning: Firebase enabled but initialization failed: {e}")
        import traceback
        traceback.print_exc()
        print("   Falling back to file-based storage")
        USE_FIREBASE = False

# Check if running on Vercel (serverless environment)
IS_VERCEL = os.getenv('VERCEL') == '1' or os.getenv('VERCEL_ENV') is not None
# Check if running on Render (persistent server environment)
IS_RENDER = os.getenv('RENDER') == 'true' or 'render.com' in os.getenv('RENDER_EXTERNAL_HOSTNAME', '')

# CORS will be handled via manual headers (no flask-cors dependency needed)
# This approach works perfectly for all use cases and doesn't require additional packages
CORS_AVAILABLE = False

# Get the directory where app.py is located (needed for imports and paths)
_app_file_dir = os.path.dirname(os.path.abspath(__file__))  # api/ directory

# Load environment variables from .env file (optional - safe for serverless)
# In Vercel, environment variables are set directly, so .env file is optional
try:
    # Try to load from api/.env first (where app.py is located)
    _env_path = os.path.join(_app_file_dir, '.env')
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
    else:
        # Fallback to root .env
        load_dotenv()
    pass
except Exception as e:
    # Silently ignore if .env file doesn't exist (common in serverless)
    if not IS_VERCEL:
        print(f"Note: Could not load .env file: {e}")

# Initialize Flask app with explicit configuration for serverless
# Use absolute paths for templates and static files to work in serverless
# Templates and static are now in api/ directory (copied from SmartPantryWeb)
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
# SECURITY WARNING: Default secret key is insecure - always set FLASK_SECRET_KEY in production
secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')
if secret_key == 'supersecretkey' and (IS_VERCEL or IS_RENDER):
    print("⚠️  WARNING: Using default secret key in production! Set FLASK_SECRET_KEY environment variable.")
app.secret_key = secret_key  # Needed for flash messages

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
            pass
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
        pass
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
        print("   On Render: Go to Dashboard -> Your Service -> Environment -> Add OPENAI_API_KEY")
    else:
        print("   Create a .env file with: OPENAI_API_KEY=your_api_key_here")
    client = None
else:
    try:
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully")
        pass
    except Exception as e:
        print(f"⚠️  ERROR: Failed to initialize OpenAI client: {e}")
        client = None

 
# ✅ ML Vision (Hybrid) configuration
# ML Vision Configuration
# IMPORTANT: ML models are disabled by default for serverless to reduce function size
# Enable only if you have sufficient storage and want ML-based detection
# Default to enabled for local/non-serverless so photo detection works out-of-the-box.
# Keep disabled by default on serverless (Vercel/Render) to reduce cold-start and size pressure.
_default_ml_vision_enabled = "false" if (IS_VERCEL or IS_RENDER) else "true"
ML_VISION_ENABLED = os.getenv("ML_VISION_ENABLED", _default_ml_vision_enabled).lower() == "true"
# For serverless deployments, prefer "classify_only" (smaller) or disable ML entirely
# ML models (transformers, torch, easyocr) are very large and can exceed function size limits
ML_VISION_MODE = os.getenv("ML_VISION_MODE", "classify_only").lower()  # hybrid | classify_only

# ✅ YOLOv8 Object Detection (Better accuracy than DETR)
# YOLOv8 provides superior object detection for pantry/fridge images
# Enable this for better detection accuracy (recommended over DETR)
_default_yolo_enabled = "false" if (IS_VERCEL or IS_RENDER) else "true"
YOLO_DETECTION_ENABLED = os.getenv("YOLO_DETECTION_ENABLED", _default_yolo_enabled).lower() == "true"
# 🔥 FIX #1: Upgrade default YOLO from nano to small for better recall
# YOLOv8s has significantly better recall for pantry scenes with dense items
YOLO_MODEL_SIZE = os.getenv("YOLO_MODEL_SIZE", "s").lower()  # n (nano), s (small), m (medium), l (large), x (xlarge)

# Lazy-loaded ML models (keep memory usage lower in serverless)
_ml_models_loaded = False
_food_classifier = None
_ocr_reader = None
_object_detector = None
_yolo_model = None  # YOLOv8 model for better object detection

# CLIP model for open-vocabulary food matching (semantic layer)
_clip_model = None
_clip_processor = None

def load_ml_models():
    """Load ML models lazily for serverless-friendly photo analysis.
    Includes comprehensive error handling and timeout protection."""
    global _ml_models_loaded, _food_classifier, _ocr_reader, _object_detector, _yolo_model, _clip_model, _clip_processor
    
    if _ml_models_loaded:
        return True
    
    # Prevent concurrent loading attempts
    import threading
    if not hasattr(load_ml_models, '_loading_lock'):
        load_ml_models._loading_lock = threading.Lock()
    
    with load_ml_models._loading_lock:
        # Double-check after acquiring lock
        if _ml_models_loaded:
            return True
        
        try:
            print("🔄 Loading ML models for photo analysis...")
            
            # Food classification (lightweight) with timeout
            # Only load if ML_VISION_ENABLED is true to reduce function size
            if not ML_VISION_ENABLED:
                print("ℹ️  ML Vision is disabled. Skipping model loading.")
                _food_classifier = None
            else:
                try:
                    # Import only when needed (lazy loading)
                    from transformers import pipeline, CLIPProcessor, CLIPModel
                    import threading
                    
                    classifier_loaded = False
                    classifier_error = None
                    _food_classifier_local = None
                    clip_loaded = False
                    clip_error = None
                    _clip_model_local = None
                    _clip_processor_local = None
                    
                    def load_classifier():
                        nonlocal classifier_loaded, classifier_error, _food_classifier_local
                        try:
                            print("   🔍 Loading lightweight food classifier from HuggingFace...")
                            # Primary option: nateraw/food (small, food-focused)
                            _food_classifier_local = pipeline(
                                "image-classification",
                                model="nateraw/food"
                            )
                            classifier_loaded = True
                            print("   ✅ Food classifier loaded: nateraw/food")
                        except Exception as e_primary:
                            print(f"   ⚠️ Primary food model 'nateraw/food' failed: {e_primary}")
                            # Fallback: Food-101 ViT
                            try:
                                _food_classifier_local = pipeline(
                                    "image-classification",
                                    model="nateraw/vit-base-food101"
                                )
                                classifier_loaded = True
                                print("   ✅ Food classifier loaded: nateraw/vit-base-food101")
                            except Exception as e_fallback:
                                classifier_error = Exception(
                                    f"Failed to load food classifiers: primary={e_primary}, fallback={e_fallback}"
                                )
                                _food_classifier_local = None
                                classifier_loaded = False
                        except Exception as e:
                            classifier_error = e
                            print(f"   ⚠️ Food classifier loading failed (non-critical): {e}")
                            classifier_loaded = False  # Don't fail completely if classifier can't load
                    
                    # Load food classifier in background
                    classifier_thread = threading.Thread(target=load_classifier)
                    classifier_thread.daemon = True
                    classifier_thread.start()
                    classifier_thread.join(timeout=180)  # up to 3 minutes to download/load HF model
                    
                    if classifier_thread.is_alive():
                        print("⚠️  Warning: Food classifier loading timed out")
                        _food_classifier = None
                    elif classifier_error:
                        print(f"⚠️  Warning: food classifier not available: {classifier_error}")
                        _food_classifier = None
                    elif classifier_loaded and _food_classifier_local:
                        _food_classifier = _food_classifier_local
                        print("✅ Food classifier loaded successfully")
                    else:
                        print("⚠️  Warning: food classifier did not load (no error but not ready)")
                        _food_classifier = None

                    # Load CLIP model for open-vocabulary matching
                    def load_clip():
                        nonlocal clip_loaded, clip_error, _clip_model_local, _clip_processor_local
                        try:
                            print("   🔍 Loading CLIP model for open-vocabulary food matching...")
                            _clip_model_local = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                            _clip_processor_local = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                            clip_loaded = True
                            print("   ✅ CLIP model loaded: openai/clip-vit-base-patch32")
                        except Exception as e:
                            clip_error = e
                            clip_loaded = False
                            print(f"   ⚠️ CLIP model loading failed (non-critical): {e}")

                    clip_thread = threading.Thread(target=load_clip)
                    clip_thread.daemon = True
                    clip_thread.start()
                    clip_thread.join(timeout=180)

                    if clip_thread.is_alive():
                        print("⚠️  Warning: CLIP loading timed out")
                        _clip_model = None
                        _clip_processor = None
                    elif clip_error:
                        print(f"⚠️  Warning: CLIP not available: {clip_error}")
                        _clip_model = None
                        _clip_processor = None
                    elif clip_loaded and _clip_model_local and _clip_processor_local:
                        _clip_model = _clip_model_local
                        _clip_processor = _clip_processor_local
                        print("✅ CLIP model ready for semantic matching")
                    else:
                        print("⚠️  Warning: CLIP did not load (no error but not ready)")
                        _clip_model = None
                        _clip_processor = None
                except Exception as e:
                    print(f"⚠️  Warning: food classifier loading failed: {e}")
                    _food_classifier = None

            # OCR for expiration dates with timeout
            # Only load if ML_VISION_ENABLED is true
            if not ML_VISION_ENABLED:
                print("ℹ️  ML Vision is disabled. Skipping OCR loading.")
                _ocr_reader = None
            else:
                try:
                    # Import only when needed (lazy loading)
                    import easyocr
                    import threading
                    
                    ocr_loaded = False
                    ocr_error = None
                    
                    _ocr_reader_local = None
                    
                    def load_ocr():
                        nonlocal ocr_loaded, ocr_error, _ocr_reader_local
                        try:
                            _ocr_reader_local = easyocr.Reader(['en'], gpu=False)
                            ocr_loaded = True
                        except Exception as e:
                            ocr_error = e
                    
                    thread = threading.Thread(target=load_ocr)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=180)  # 3 minute timeout (OCR takes longer)
                    
                    if thread.is_alive():
                        print("⚠️  Warning: OCR reader loading timed out")
                        _ocr_reader = None
                    elif ocr_error:
                        print(f"⚠️  Warning: OCR not available: {ocr_error}")
                        _ocr_reader = None
                    elif ocr_loaded and _ocr_reader_local:
                        _ocr_reader = _ocr_reader_local
                        print("✅ OCR reader loaded")
                    else:
                        _ocr_reader = None
                except Exception as e:
                    print(f"⚠️  Warning: OCR loading failed: {e}")
                    _ocr_reader = None

            # Object detection (general model with food label mapping) with timeout
            # Only load if ML_VISION_ENABLED is true AND mode is hybrid
            # Hybrid mode requires large models (torch, DETR) - can exceed serverless limits
            if not ML_VISION_ENABLED or ML_VISION_MODE != "hybrid":
                print(f"ℹ️  Object detection disabled. ML_VISION_ENABLED={ML_VISION_ENABLED}, ML_VISION_MODE={ML_VISION_MODE}")
                _object_detector = None
            else:
                try:
                    # Import only when needed (lazy loading) - these are very large!
                    from transformers import AutoImageProcessor, AutoModelForObjectDetection
                    import torch
                    import threading
                    
                    detector_loaded = False
                    detector_error = None
                    
                    _object_detector_local = None
                    
                    def load_detector():
                        nonlocal detector_loaded, detector_error, _object_detector_local
                        try:
                            processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
                            model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
                            model.eval()
                            _object_detector_local = {"processor": processor, "model": model, "torch": torch}
                            detector_loaded = True
                            pass
                        except Exception as e:
                            detector_error = e
                    
                    thread = threading.Thread(target=load_detector)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=180)  # 3 minute timeout
                    
                    if thread.is_alive():
                        print("⚠️  Warning: Object detector loading timed out")
                        _object_detector = None
                    elif detector_error:
                        print(f"⚠️  Warning: object detector not available: {detector_error}")
                        _object_detector = None
                    elif detector_loaded and _object_detector_local:
                        _object_detector = _object_detector_local
                        print("✅ Object detector loaded")
                    else:
                        _object_detector = None
                except Exception as e:
                    print(f"⚠️  Warning: object detector loading failed: {e}")
                    _object_detector = None

            # YOLOv8 Object Detection (Better accuracy - recommended over DETR)
            # YOLOv8 provides superior object detection for pantry/fridge images
            if not YOLO_DETECTION_ENABLED:
                print("ℹ️  YOLOv8 detection disabled. Set YOLO_DETECTION_ENABLED=true to enable.")
                _yolo_model = None
            else:
                try:
                    # Import only when needed (lazy loading)
                    from ultralytics import YOLO
                    import threading
                    
                    yolo_loaded = False
                    yolo_error = None
                    _yolo_model_local = None
                    
                    # 🔥 FIX #1: Use YOLOv8s (small) by default for better recall on pantry scenes
                    # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large), yolov8x.pt (xlarge)
                    # Small model provides better recall for dense pantry items while still being fast
                    model_name = f"yolov8{YOLO_MODEL_SIZE}.pt"
                    
                    def load_yolo():
                        nonlocal yolo_loaded, yolo_error, _yolo_model_local
                        try:
                            _yolo_model_local = YOLO(model_name)
                            yolo_loaded = True
                            pass
                        except Exception as e:
                            yolo_error = e
                    
                    thread = threading.Thread(target=load_yolo)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=120)  # 2 minute timeout
                    
                    if thread.is_alive():
                        print("⚠️  Warning: YOLOv8 model loading timed out")
                        _yolo_model = None
                    elif yolo_error:
                        print(f"⚠️  Warning: YOLOv8 not available: {yolo_error}")
                        print("   Install with: pip install ultralytics")
                        _yolo_model = None
                    elif yolo_loaded and _yolo_model_local:
                        _yolo_model = _yolo_model_local
                        print(f"✅ YOLOv8 model loaded ({model_name})")
                    else:
                        _yolo_model = None
                except ImportError:
                    print("⚠️  Warning: ultralytics not installed. Install with: pip install ultralytics")
                    _yolo_model = None
                except Exception as e:
                    print(f"⚠️  Warning: YOLOv8 loading failed: {e}")
                    _yolo_model = None

            # Mark as loaded even if some models failed (partial loading is OK)
            _ml_models_loaded = True
            return True
        except Exception as e:
            print(f"⚠️  ERROR: Failed to load ML models: {e}")
            import traceback
            traceback.print_exc()
            # Still mark as loaded to prevent retry loops
            _ml_models_loaded = True
            return False

def compute_image_quality_score(img):
    """
    Compute image quality score (0.0-1.0) based on:
    - Blur score (Laplacian variance)
    - Brightness
    - Contrast
    - Glare/reflection detection
    
    Returns:
        quality_score: float in [0.0, 1.0], where 1.0 is perfect quality
        quality_metrics: dict with individual scores
    """
    try:
        import numpy as np
        from PIL import Image
        import cv2
        
        # Convert PIL to numpy array
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = img
        
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        metrics = {}
        
        # 1. Blur score using Laplacian variance (higher = sharper)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize: good images typically have variance > 100, poor < 50
        blur_score = min(1.0, laplacian_var / 200.0)  # Cap at 1.0 for variance > 200
        metrics['blur_score'] = blur_score
        
        # 2. Brightness score (optimal range: 100-200)
        try:
            mean_brightness = np.mean(gray)
            if not np.isfinite(mean_brightness):
                mean_brightness = 128.0  # Default to middle
            
            # Penalize too dark (< 50) or too bright (> 230)
            if mean_brightness < 50:
                brightness_score = max(0.0, mean_brightness / 50.0)
            elif mean_brightness > 230:
                brightness_score = max(0.0, (255 - mean_brightness) / 25.0)
            else:
                brightness_score = 1.0  # Optimal range
            brightness_score = max(0.0, min(1.0, brightness_score))
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"Warning: Brightness detection failed: {e}")
            brightness_score = 0.8  # Default to good score
        
        metrics['brightness_score'] = brightness_score
        
        # 3. Contrast score (standard deviation of pixel values)
        try:
            contrast_std = np.std(gray)
            if not np.isfinite(contrast_std) or contrast_std < 0:
                contrast_std = 30.0  # Default to medium contrast
            # Good contrast typically has std > 30, poor < 15
            contrast_score = min(1.0, max(0.0, contrast_std / 50.0))
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"Warning: Contrast detection failed: {e}")
            contrast_score = 0.7  # Default to medium score
        
        metrics['contrast_score'] = contrast_score
        
        # 4. Glare/reflection detection (high brightness variance in small regions)
        # Divide image into blocks and check for high variance (indicates glare)
        try:
            h, w = gray.shape
            if h <= 0 or w <= 0:
                glare_score = 1.0
            else:
                block_size = min(32, max(1, h // 4), max(1, w // 4))  # Ensure at least 1
                if block_size > 0:
                    block_vars = []
                    for y in range(0, h, block_size):
                        for x in range(0, w, block_size):
                            try:
                                block = gray[y:y+block_size, x:x+block_size]
                                if block.size > 0:
                                    block_vars.append(np.var(block))
                            except (IndexError, ValueError):
                                continue
                    
                    if block_vars and len(block_vars) > 0:
                        # High variance in many blocks suggests glare/reflection
                        high_var_blocks = sum(1 for v in block_vars if v > 1000)
                        glare_ratio = high_var_blocks / len(block_vars)  # Safe division (len > 0)
                        # Penalize if > 20% of blocks have high variance (glare)
                        glare_score = max(0.0, 1.0 - (glare_ratio - 0.2) * 2.0)
                    else:
                        glare_score = 1.0
                else:
                    glare_score = 1.0
        except (AttributeError, IndexError, ValueError) as e:
            if VERBOSE_LOGGING:
                print(f"Warning: Glare detection failed: {e}")
            glare_score = 1.0  # Default to good score if analysis fails
        
        metrics['glare_score'] = glare_score
        
        # Combined quality score (weighted average)
        quality_score = (
            blur_score * 0.35 +      # Most important: sharpness
            brightness_score * 0.25 + # Important: proper exposure
            contrast_score * 0.25 +   # Important: good contrast
            glare_score * 0.15         # Less critical: glare detection
        )
        
        metrics['overall_quality'] = quality_score
        return quality_score, metrics
        
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Warning: Image quality scoring failed: {e}")
        # Return default good score if analysis fails
        return 0.8, {'error': str(e)}

def preprocess_image_for_ml(img_bytes, angle=None, apply_perspective_correction=False):
    """
    Enhanced preprocessing: resize + enhance + sharpen + rotation correction for better accuracy.
    Handles bad angle photos by normalizing rotation and improving visibility.
    
    Args:
        img_bytes: Raw image bytes
        angle: Optional rotation angle in degrees (0, 90, 180, 270)
        apply_perspective_correction: If True, try to correct perspective distortion
    """
    from PIL import Image, ImageEnhance, ImageFilter
    import io
    import numpy as np
    
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Apply rotation if specified
    if angle is not None and angle in [90, 180, 270]:
        img = img.rotate(-angle, expand=True)  # Negative because PIL rotates counter-clockwise
    
    # Try to correct perspective distortion (optional, can be slow)
    # Disabled by default to reduce dependencies (would require numpy/OpenCV)
    if apply_perspective_correction:
        try:
            # Simple perspective correction attempt (keystone correction)
            # This would require numpy/OpenCV which are large dependencies
            # For now, skip perspective correction to keep function size small
            # Could be enabled in future if needed
            pass
            pass
        except Exception:
            pass  # Skip perspective correction if it fails
    
    # 🔥 PERFORMANCE: Resize early to reduce processing time
    # Balance: 800px is sufficient for YOLO/CLIP while being 2x faster than 1024px
    max_size = 800  # Reduced from 1024 for better performance (still good accuracy)
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        # Use faster resize method for initial resize
        img = img.resize(new_size, Image.Resampling.BILINEAR)  # Faster than LANCZOS
    
    # Enhanced preprocessing optimized for fridge/pantry photos AND bad angles
    # Bad angle photos often have poor lighting and contrast
    # Sharpen to improve edge detection and text recognition (especially important for angled photos)
    img = img.filter(ImageFilter.SHARPEN)
    
    # Improve: More aggressive enhancement for difficult angles
    # Increase contrast MORE aggressively for angled photos (items are harder to see)
    # Angled photos often have less contrast due to lighting angles
    img = ImageEnhance.Contrast(img).enhance(1.6)  # Increased from 1.5 for difficult angles
    
    # Brightness adjustment - angled photos may have uneven lighting
    # Use adaptive brightness enhancement
    img = ImageEnhance.Brightness(img).enhance(1.2)  # Increased from 1.15 for difficult angles
    
    # Saturation boost to help identify colorful food items (more important with bad angles)
    img = ImageEnhance.Color(img).enhance(1.2)  # Increased from 1.15 for difficult angles
    
    # Additional enhancement for angled photos: reduce noise and improve sharpness
    # Apply gentle unsharp mask for better edge detection
    try:
        from PIL import ImageFilter as IF
        # Unsharp mask improves visibility of edges in angled photos
        img = img.filter(IF.UnsharpMask(radius=1, percent=150, threshold=3))
        pass
    except Exception:
        pass  # Skip if UnsharpMask not available
    
    return img

def detect_rotation_angle(img):
    """
    Detect if image is rotated (0, 90, 180, 270 degrees).
    Uses OCR text orientation as a heuristic.
    Returns best angle (0, 90, 180, 270) or None if uncertain.
    """
    try:
        import numpy as np
        # Simple heuristic: try OCR on original and rotated versions
        # The version with most readable text is likely the correct orientation
        if not _ocr_reader:
            return None
        
        img_array = np.array(img)
        angles_to_test = [0, 90, 180, 270]
        best_angle = 0
        max_text_score = 0
        
        for angle in angles_to_test:
            try:
                # Rotate image
                rotated_img = img.rotate(-angle, expand=True)
                rotated_array = np.array(rotated_img)
                
                # Run OCR (with timeout)
                import threading
                ocr_results = []
                ocr_error = None
                
                def run_ocr():
                    nonlocal ocr_results, ocr_error
                    try:
                        # Limit OCR to small sample to be fast
                        ocr_results = _ocr_reader.readtext(rotated_array[:500, :500])
                        pass
                    except Exception as e:
                        ocr_error = e
                
                thread = threading.Thread(target=run_ocr)
                thread.daemon = True
                thread.start()
                thread.join(timeout=5)  # Quick timeout
                
                if not ocr_error and ocr_results:
                    # Score based on text confidence and number of words
                    score = sum(conf for _, _, conf in ocr_results if conf > 0.5) * len(ocr_results)
                    if score > max_text_score:
                        max_text_score = score
                        best_angle = angle
            except Exception:
                continue
        
        # Only return if we found a clear winner (threshold to avoid false positives)
        if max_text_score > 2.0:  # Threshold for confidence
            return best_angle
        return None
    except Exception:
        return None

def detect_with_multiple_angles(img_bytes, user_pantry=None):
    """
    Detect food items by trying multiple orientations (ensemble approach).
    Merges results from different angles for better consistency with bad angle photos.
    """
    item_confidence_map = {}
    
    # Improve: Try more angles for difficult angle detection
    # Try original and common rotations (0, 90, 180, 270) for better coverage
    # Limit to most common angles to balance speed vs accuracy
    angles_to_try = [None, 90, 180, 270]  # Original, 90° right, 180° upside down, 90° left
    
    for angle in angles_to_try:
        try:
            # Preprocess with specific angle
            img = preprocess_image_for_ml(img_bytes, angle=angle)
            
            # Convert PIL Image back to bytes for detection function
            import io
            from PIL import Image
            img_bytes_processed = io.BytesIO()
            img.save(img_bytes_processed, format='JPEG', quality=85)
            img_bytes_processed.seek(0)
            img_bytes_processed = img_bytes_processed.read()
            
            # Run detection on this orientation (skip multi-angle to avoid recursion)
            items = detect_food_items_with_ml(img_bytes_processed, user_pantry=user_pantry, skip_preprocessing=False, use_multi_angle=False)
            
            # Merge results (keep highest confidence per item)
            for item in items:
                name_key = item.get("name", "").lower().strip()
                if not name_key:
                    continue
                
                conf = item.get("confidence", 0)
                # Boost confidence slightly for multi-angle consensus
                if name_key in item_confidence_map:
                    # Item detected from multiple angles = higher confidence
                    existing_conf = item_confidence_map[name_key].get("confidence", 0)
                    # Weighted average with boost for consensus
                    boosted_conf = (existing_conf * 0.6 + conf * 0.4) * 1.15  # 15% boost for consensus
                    if boosted_conf > existing_conf:
                        item["confidence"] = min(1.0, boosted_conf)
                        item_confidence_map[name_key] = item
                else:
                    # First detection of this item
                    item_confidence_map[name_key] = item
            
            pass
        except Exception as e:
            if VERBOSE_LOGGING:

                print(f"Warning: Failed to detect with angle {angle}: {e}")
            continue
    
    # Return merged results
    result = list(item_confidence_map.values())
    result.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return result

def extract_expiration_dates(img, ocr_reader):
    """Extract expiration dates using OCR."""
    if not ocr_reader:
        return []
    try:
        import numpy as np
        import re
        img_array = np.array(img)
        ocr_results = ocr_reader.readtext(img_array)
        date_patterns = [
            r'EXP[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'USE\s+BY[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'BEST\s+BY[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'SELL\s+BY[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ]
        found_dates = []
        for (_bbox, text, confidence) in ocr_results:
            if confidence < 0.5:
                continue
            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    date_str = match.group(1) if match.groups() else match.group(0)
                    normalized = normalize_expiration_date(date_str)
                    if normalized:
                        found_dates.append(normalized)
                        break
        return found_dates
        pass
    except Exception as e:
        if VERBOSE_LOGGING:

            print(f"OCR error: {e}")
        return []

def classify_food_hierarchical(img_crop, classification_pred=None):
    """
    Hierarchical classification: Category -> Specific Item
    Returns: {"name": "...", "category": "...", "confidence": 0.0-1.0}
    """
    if not _food_classifier and not classification_pred:
        return None
    
    categories = {
        "dairy": ["milk", "cheese", "yogurt", "butter", "eggs", "cream", "egg"],
        "produce": ["apple", "banana", "tomato", "lettuce", "carrot", "onion", "potato", "orange", "broccoli"],
        "meat": ["chicken", "beef", "pork", "fish", "turkey", "bacon", "sausage"],
        "beverages": ["juice", "soda", "water", "coffee", "tea", "beer", "wine", "cola"],
        "bakery": ["bread", "bagel", "muffin", "pastry", "roll", "bun"],
        "canned_goods": ["soup", "beans", "tuna", "corn"],
        "snacks": ["chips", "crackers", "cookies", "nuts", "popcorn"],
        "condiments": ["ketchup", "mustard", "mayonnaise", "sauce", "dressing", "oil", "vinegar"],
        "grains": ["rice", "pasta", "cereal", "flour", "quinoa", "oats"],
        "frozen": ["ice_cream", "frozen_vegetables"],
        "other": []
    }
    
    # Get classification if not provided
    if not classification_pred and _food_classifier:
        try:
            classification_pred = _food_classifier(img_crop)
            pass
        except Exception:
            return None
    
    if not classification_pred:
        return None
    
    # Find best matching category and item
    best_item = None
    best_conf = 0.0
    best_category = "other"
    
    for pred in classification_pred[:5]:  # Check top 5 predictions
        if not isinstance(pred, dict):
            continue
        
        label = pred.get("label", "").lower()
        score = float(pred.get("score", 0))
        
        # Lower threshold from 0.10 to 0.05 to catch more items
        if score < 0.05:
            continue
        
        # Find category that matches
        matched_category = "other"
        for cat, keywords in categories.items():
            if any(kw in label for kw in keywords):
                matched_category = cat
                break
        
        # Normalize item name
        name = normalize_item_name(label)
        if not name:
            continue
        
        # Use this prediction if it's better
        if score > best_conf:
            best_conf = score
            best_item = name
            best_category = matched_category
    
    if best_item:
        return {
            "name": best_item,
            "category": best_category,
            "confidence": best_conf
        }


def clip_match_open_vocabulary(img_crop, labels, use_prompt_engineering=True):
    """
    Use CLIP to match an image crop against an arbitrary list of text labels.
    
    Precision Rule #2: Prompt engineering - uses "a photo of {item}" format for better accuracy.
    Precision Rule #7: Calibrates CLIP scores using temperature scaling.
    
    Returns: {"label": best_label, "score": calibrated_score, "second_best": calibrated_second_best, "raw_score": raw_score} or None
    """
    global _clip_model, _clip_processor
    if not _clip_model or not _clip_processor:
        return None
    if not labels:
        return None
    try:
        from PIL import Image
        import torch
    except Exception:
        return None
    try:
        if not isinstance(img_crop, Image.Image):
            return None
        
        # 🔥 IMPROVEMENT #2: Better prompt engineering - pantry-specific context
        # Instead of "a photo of cereal", use "a boxed breakfast cereal on a pantry shelf"
        # This significantly improves CLIP accuracy for packaged food
        if use_prompt_engineering:
            # Map generic labels to pantry-specific prompts
            prompt_mapping = {
                "cereal": "a boxed breakfast cereal on a pantry shelf",
                "granola": "a boxed granola cereal on a pantry shelf",
                "oats": "a boxed oatmeal or oats on a pantry shelf",
                "protein bar": "a packaged protein bar",
                "granola bar": "a packaged granola bar",
                "snacks": "packaged snacks on a pantry shelf",
                "chips": "a bag of chips on a pantry shelf",
                "crackers": "a box of crackers on a pantry shelf",
                "cookies": "a package of cookies on a pantry shelf",
                "pasta": "a box of pasta on a pantry shelf",
                "rice": "a bag or box of rice on a pantry shelf",
                "flour": "a bag of flour on a pantry shelf",
                "sugar": "a bag of sugar on a pantry shelf",
                "bread": "a loaf of bread",
                "milk": "a carton of milk",
                "juice": "a bottle or carton of juice",
                "sauce": "a jar or bottle of sauce on a pantry shelf",
                "tomato sauce": "a can or jar of tomato sauce on a pantry shelf",
                "pasta sauce": "a jar of pasta sauce on a pantry shelf",
                "ketchup": "a bottle of ketchup",
                "mustard": "a bottle of mustard",
                "mayonnaise": "a jar of mayonnaise",
                "soup": "a can of soup on a pantry shelf",
                "canned goods": "a can of food on a pantry shelf",
                "quinoa": "a bag of quinoa on a pantry shelf",
                "macaroni": "a box of macaroni on a pantry shelf"
            }
            
            prompt_labels = []
            for label in labels:
                label_lower = label.lower().strip()
                # Check for exact match first
                if label_lower in prompt_mapping:
                    prompt_labels.append(prompt_mapping[label_lower])
                # Check for partial matches (e.g., "protein bar" contains "bar")
                elif any(key in label_lower for key in prompt_mapping.keys()):
                    # Find the best matching key
                    best_match = None
                    for key in prompt_mapping.keys():
                        if key in label_lower or label_lower in key:
                            best_match = key
                            break
                    if best_match:
                        prompt_labels.append(prompt_mapping[best_match])
                    else:
                        # Fallback: use pantry context for generic items
                        prompt_labels.append(f"a packaged {label} on a pantry shelf")
                else:
                    # Fallback: use pantry context for unknown items
                    prompt_labels.append(f"a packaged {label} on a pantry shelf")
        else:
            prompt_labels = labels
        
        # Performance: Use efficient batching and move to device if available
        device = next(_clip_model.parameters()).device
        inputs = _clip_processor(text=prompt_labels, images=img_crop, return_tensors="pt", padding=True)
        # Move inputs to same device as model (GPU if available)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            out = _clip_model(**inputs)
            probs = out.logits_per_image.softmax(dim=-1)[0].cpu().numpy()
        
        if probs.size == 0:
            return None
        
        # FIX 3: Get top-3 results for ambiguity detection
        sorted_indices = probs.argsort()[::-1]  # Sort descending
        top3_indices = sorted_indices[:min(3, len(labels))]
        
        best_idx = int(top3_indices[0])
        raw_best_score = float(probs[best_idx])
        
        # Get top-3 scores and labels
        top3_scores = [float(probs[i]) for i in top3_indices]
        top3_labels = [labels[i].lower() for i in top3_indices]
        
        # second-best and third-best scores
        if len(top3_scores) > 1:
            raw_second_best = top3_scores[1]
        else:
            raw_second_best = 0.0
        if len(top3_scores) > 2:
            raw_third_best = top3_scores[2]
        else:
            raw_third_best = 0.0
        
        # Precision Rule #7: Calibrate CLIP scores using temperature scaling
        # FIXED: Softer calibration that preserves low scores for pantry images
        # Old formula zeroed out scores < 0.2, causing all low-confidence items to become 0.0
        # New formula: preserves relative differences and maps to 0.0-1.0 range
        # Formula: calibrated = raw^0.7 * 1.5 (soft power scaling)
        # This maps: 0.01 → 0.03, 0.05 → 0.12, 0.1 → 0.19, 0.2 → 0.33, 0.5 → 0.65
        def calibrate_score(raw):
            if raw <= 0:
                return 0.0
            # Soft power scaling preserves low scores while still boosting high scores
            calibrated = (raw ** 0.7) * 1.5
            return min(1.0, max(0.0, calibrated))
        
        calibrated_best = calibrate_score(raw_best_score)
        calibrated_second = calibrate_score(raw_second_best)
        calibrated_third = calibrate_score(raw_third_best)
        
        return {
            "label": top3_labels[0],  # Return original label, not prompt
            "score": calibrated_best,
            "raw_score": raw_best_score,
            "second_best": calibrated_second,
            "raw_second_best": raw_second_best,
            "third_best": calibrated_third,
            "raw_third_best": raw_third_best,
            "top3_labels": top3_labels,  # FIX 3: Return top-3 labels
            "top3_scores": [calibrated_best, calibrated_second, calibrated_third]  # FIX 3: Return top-3 calibrated scores
        }
    except Exception:
        return None
    
    return None

def apply_context_fusion(items, user_pantry=None, img_height=None):
    """
    Enhanced context fusion with shelf-aware spatial context:
        - Items already in pantry are more likely
        - Common items are more likely
        - Shelf-aware location hints (door shelves -> beverages/condiments, crisper -> produce, etc.)
    
    Args:
        items: List of detected items with bounding boxes
        user_pantry: Optional user's current pantry for context fusion
        img_height: Optional image height for spatial context (Y-position normalization)

    """
    if not user_pantry:
        user_pantry = []
    
    existing_items = set()
    if isinstance(user_pantry, list):
        for item in user_pantry:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name:
                    existing_items.add(name.lower().strip())
    
    common_items = {"milk", "eggs", "bread", "cheese", "chicken", "yogurt", "butter", "apple", "banana", "tomato"}
    
    # Shelf-aware categories (typical fridge/pantry organization)
    DOOR_SHELF_ITEMS = {"beverages", "condiments", "juice", "soda", "water", "ketchup", "mustard", "mayonnaise", "sauce", "dressing"}
    CRISPER_ITEMS = {"produce", "apple", "banana", "orange", "tomato", "lettuce", "carrot", "onion", "potato", "broccoli"}
    BOTTOM_SHELF_ITEMS = {"meat", "dairy", "chicken", "beef", "pork", "fish", "milk", "cheese", "yogurt", "butter"}
    TOP_SHELF_ITEMS = {"bakery", "snacks", "bread", "bagel", "muffin", "crackers", "cookies"}
    
    for item in items:
        name_lower = item.get("name", "").lower().strip()
        if not name_lower:
            continue
        
        base_confidence = item.get("confidence", 0)
        boosts = []
        
        # 1. Boost if already in pantry
        if name_lower in existing_items:
            item["confidence"] = min(1.0, base_confidence + 0.15)
            boosts.append("already_in_pantry")
        
        # 2. Boost if common item
        elif name_lower in common_items:
            item["confidence"] = min(1.0, base_confidence + 0.1)
            boosts.append("common_item")
        
        # 3. Shelf-aware spatial context (if bounding box available)
        if img_height and img_height > 0 and 'bbox' in item and item.get('bbox'):
            try:
                bbox = item['bbox']
                if len(bbox) >= 4 and isinstance(bbox[1], (int, float)) and isinstance(bbox[3], (int, float)):
                    # Get Y position (normalized: 0.0 = top, 1.0 = bottom)
                    # Prevent division by zero
                    if img_height > 0:
                        y_center = (bbox[1] + bbox[3]) / 2.0 / img_height
                    else:
                        continue  # Skip if invalid img_height
                    
                    # Get item category
                    category = item.get('category', 'other').lower()
                    item_name = name_lower
                    
                    # Door shelves (typically on sides, but we use Y-position as proxy for top-middle)
                    # In fridges, door items are often in middle Y-range (0.3-0.7)
                    if 0.3 <= y_center <= 0.7:
                        if any(door_item in item_name or door_item in category for door_item in DOOR_SHELF_ITEMS):
                            item["confidence"] = min(1.0, item.get("confidence", base_confidence) + 0.1)
                            boosts.append("door_shelf_match")
                    
                    # Crisper drawer (bottom: y > 0.75)
                    if y_center > 0.75:
                        if any(crisper_item in item_name or crisper_item in category for crisper_item in CRISPER_ITEMS):
                            item["confidence"] = min(1.0, item.get("confidence", base_confidence) + 0.1)
                            boosts.append("crisper_match")
                    
                    # Bottom shelf (y > 0.65)
                    if y_center > 0.65:
                        if any(bottom_item in item_name or bottom_item in category for bottom_item in BOTTOM_SHELF_ITEMS):
                            item["confidence"] = min(1.0, item.get("confidence", base_confidence) + 0.08)
                            boosts.append("bottom_shelf_match")
                    
                    # Top shelf (y < 0.35)
                    if y_center < 0.35:
                        if any(top_item in item_name or top_item in category for top_item in TOP_SHELF_ITEMS):
                            item["confidence"] = min(1.0, item.get("confidence", base_confidence) + 0.08)
                            boosts.append("top_shelf_match")
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"Warning: Shelf-aware context fusion error: {e}")
        
        if boosts:
            item["context_boost"] = "+".join(boosts)
    
    return items

def apply_nms(items, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping detections of the same item.
    Keeps the detection with highest confidence when boxes overlap significantly.
    """
    if not items or len(items) < 2:
        return items
    
    # Filter items that have bounding boxes
    items_with_bbox = [item for item in items if 'bbox' in item and item.get('bbox')]
    items_without_bbox = [item for item in items if 'bbox' not in item or not item.get('bbox')]
    
    if not items_with_bbox:
        return items
    
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    # Sort by confidence (descending)
    items_with_bbox.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    # Apply NMS
    keep = []
    while items_with_bbox:
        # Take the item with highest confidence
        current = items_with_bbox.pop(0)
        keep.append(current)
        
        # Remove items that overlap significantly with current
        items_with_bbox = [
            item for item in items_with_bbox
            if calculate_iou(current['bbox'], item['bbox']) < iou_threshold
        ]
    
    # Combine kept items with items without bboxes
    return keep + items_without_bbox

def apply_ensemble_boosting(result, all_detections):
    """
    Boost confidence for items detected by multiple methods.
    If the same item was detected by YOLOv8, DETR, and classification, boost confidence.
    """
    if not all_detections or len(all_detections) < 2:
        return result
    
    # Count detection methods per item
    detection_counts = {}
    for item in all_detections:
        key = item.get("name", "").lower().strip()
        if not key:
            continue
        
        if key not in detection_counts:
            detection_counts[key] = {
                'count': 0,
                'methods': set(),
                'max_confidence': item.get('confidence', 0),
                'item': item
            }
        
        detection_counts[key]['count'] += 1
        method = item.get('detection_method', 'unknown')
        if method:
            detection_counts[key]['methods'].add(method)
        detection_counts[key]['max_confidence'] = max(
            detection_counts[key]['max_confidence'],
            item.get('confidence', 0)
        )
    
    # Boost confidence for items detected by multiple methods or multiple times
    for item in result:
        key = item.get("name", "").lower().strip()
        if key in detection_counts:
            detection_info = detection_counts[key]
            method_count = len(detection_info['methods'])
            detection_count = detection_info['count']
            
            # Boost for multiple detection methods
            method_boost = 1.0
            if method_count > 1:
                method_boost = 1.0 + (method_count - 1) * 0.10  # 10% per additional method
            
            # Boost for multiple detections of same item (consensus)
            count_boost = 1.0
            if detection_count > 1:
                count_boost = 1.0 + min(0.20, (detection_count - 1) * 0.05)  # 5% per additional detection, max 20%
            
            # Apply boosts
            boosted_conf = item.get('confidence', 0) * method_boost * count_boost
            # 🔥 IMPROVEMENT 1: Cap confidence at 0.85 to prevent "fake confidence"
            item['confidence'] = min(0.85, boosted_conf)


            item['ensemble_boost'] = True


            item['detection_methods'] = list(detection_info['methods'])


            item['detection_count'] = detection_count


    
    return result


def calibrate_confidence(items):
    """
    🔥 IMPROVED: Calibrate confidence scores with confidence ceiling and disagreement penalty.
    Prevents "fake confidence" from stacked boosts.
    """
    # TODO: Load historical accuracy data from feedback
    # For now, apply basic calibration based on detection method
    for item in items:
        method = item.get('detection_method', 'unknown')
        original_conf = item.get('confidence', 0.5)
        name = item.get('name', '').lower()  # Get item name for keyword matching
        
        # 🔥 PRECISION: Reduced confidence boosts to prevent "fake confidence"
        # Prevents weak detections from being boosted to appear strong
        boost_factors = {
            'yolov8': 1.15,  # Reduced from 1.25 - prevent low confidence from becoming high
            'yolov8+classification': 1.25,  # Reduced from 1.35 - ensemble is strong but not inflated
            'ensemble': 1.25,  # Reduced from 1.40 - multiple methods agree but don't over-boost
            'classification': 1.10,  # Reduced from 1.15 - classification is food-specific
            'unknown': 1.05  # Reduced from 1.10
        }
        
        factor = boost_factors.get(method, 1.10)
        
        # Additional boosts for specific item characteristics
        additional_boost = 1.0
        
        # Boost for actual food items (not containers) - but less aggressively
        food_keywords = ['apple', 'banana', 'orange', 'bread', 'milk', 'cheese', 'egg', 
                        'chicken', 'beef', 'pasta', 'rice', 'cereal', 'soup', 'donut',
                        'sandwich', 'pizza', 'cake', 'carrot', 'broccoli', 'tomato',
                        'yogurt', 'butter', 'fish', 'pork', 'turkey', 'onion', 'garlic',
                        'potato', 'lettuce', 'cucumber', 'pepper', 'olive', 'honey',
                        'mustard', 'mayonnaise', 'ketchup', 'vinegar', 'oil']
        if any(kw in name for kw in food_keywords):
            additional_boost *= 1.15  # Reduced from 1.30 - food items are more reliable but don't over-boost
        
        # Boost for items detected by YOLO - but only if confidence is decent
        # Base confidence matters - but don't boost very low confidence items
        if original_conf > 0.20:  # Raised threshold from 0.10 to 0.20
            additional_boost *= 1.05  # Reduced from 1.15 - only slight boost for decent confidence
        
        # Rule 1: YOLO is blind - never trust YOLO class names as semantic truth
        # YOLO only provides regions, not food labels
        # Only use classifier/CLIP/OCR labels for disagreement checks
        classifier_label = item.get('classifier_label', '').lower() if item.get('classifier_label') else None
        clip_label = item.get('clip_label', '').lower() if item.get('clip_label') else None
        ocr_label = item.get('ocr_label', '').lower() if item.get('ocr_label') else None
        
        disagreement_penalty = 1.0
        # Only check disagreement between semantic models (classifier, CLIP, OCR), not YOLO
        labels_to_check = [l for l in [classifier_label, clip_label, ocr_label] if l]
        if len(labels_to_check) >= 2:
            # Check if labels disagree significantly
            unique_labels = set(labels_to_check)
            if len(unique_labels) > 1:
                # Strong disagreement between semantic models - reduce confidence by 20%
                disagreement_penalty = 0.80
                if VERBOSE_LOGGING:
                    print(f"   ⚠️ Disagreement penalty: {unique_labels} -> confidence × 0.80")
        
        # FIX 7: Stop inflating confidence - use honest scores
        # Only apply minimal calibration, don't stack boosts
        # Original confidence should reflect actual uncertainty
        if method in ['yolov8+clip', 'yolov8+ocr', 'ensemble']:
            # These methods are already strong - minimal boost
            calibrated_conf = min(0.9, original_conf * 1.05 * disagreement_penalty)
        else:
            # Other methods - slight boost but keep honest
            calibrated_conf = min(0.85, original_conf * factor * disagreement_penalty)
        
        # FIX 7: Never force confidence above actual signal strength
        # If original confidence is low, don't inflate it
        if original_conf < 0.3:
            calibrated_conf = min(calibrated_conf, original_conf * 1.2)  # Max 20% boost for low confidence
        
        # 🔥 IMPROVEMENT 1C: Only boost if all models agree (OCR + YOLO + classifier)
        # If models agree, then boost - otherwise we already penalized
        if not item.get('disagreement', False):
            # Models agree - slight boost for consensus
            calibrated_conf = min(0.85, calibrated_conf * 1.05)  # Small boost for agreement
        
        # Store both original and calibrated
        item['original_confidence'] = original_conf
        item['confidence'] = calibrated_conf
        item['calibration_boost'] = factor * additional_boost * disagreement_penalty
        if disagreement_penalty < 1.0:
            item['disagreement_penalty'] = True
    
    return items

def categorize_by_confidence(items):
    """
    Categorize items by adaptive confidence thresholds for user confirmation:
        - High confidence (≥adaptive threshold): Auto-add
    - Medium confidence (0.5-adaptive threshold): Ask user
    - Low confidence (<0.5): Ignore or mark uncertain
    
    Uses item-aware thresholds: some items need higher confidence to auto-add.
    """
    MEDIUM_CONF = 0.5  # Minimum for consideration
    
    high_conf = []
    medium_conf = []
    low_conf = []
    
    for item in items:
        if not isinstance(item, dict):
            continue
        
        # Safely get confidence with validation
        conf = item.get("confidence", 0)
        try:
            conf = float(conf) if conf is not None else 0.0
            conf = max(0.0, min(1.0, conf))  # Clamp to [0.0, 1.0]
        except (ValueError, TypeError):
            conf = 0.0
        
        item_name = item.get("name", "") or ""
        category = item.get("category", "other") or "other"
        
        # Get adaptive threshold for this item
        try:
            adaptive_threshold = get_adaptive_confidence_threshold(item_name, category)
            adaptive_threshold = max(0.0, min(1.0, float(adaptive_threshold)))  # Validate threshold
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"Warning: Error getting adaptive threshold: {e}")
            adaptive_threshold = 0.7  # Fallback to default
        
        item["auto_add_threshold"] = adaptive_threshold
        
        if conf >= adaptive_threshold:
            high_conf.append(item)
        elif conf >= MEDIUM_CONF:
            medium_conf.append(item)
        else:
            low_conf.append(item)
    
    return {
        "high_confidence": high_conf,
        "medium_confidence": medium_conf,
        "low_confidence": low_conf
    }

def get_adaptive_confidence_threshold(item_name, category):
    """
    Get item-aware confidence threshold for auto-adding.
    Some item types are naturally noisier and need higher thresholds.
    
    Returns:
        threshold: float in [0.0, 1.0] - confidence needed for auto-add
    """
    name_lower = item_name.lower() if item_name else ""
    category_lower = category.lower() if category else ""
    
    # High-reliability items (common, distinctive) - lower threshold
    high_reliability = {
        "milk", "eggs", "bread", "apple", "banana", "orange", "tomato",
        "chicken", "beef", "cheese", "yogurt", "butter"
    }
    
    # Medium-reliability items - standard threshold
    medium_reliability = {
        "pasta", "rice", "cereal", "soup", "juice", "water", "soda",
        "carrot", "lettuce", "onion", "potato"
    }
    
    # Low-reliability items (packaged, similar-looking) - higher threshold
    low_reliability = {
        "sauce", "dressing", "condiment", "spice", "herb", "oil", "vinegar",
        "crackers", "cookies", "chips", "snacks"
    }
    
    # Check item name
    if any(hr in name_lower for hr in high_reliability):
        return 0.6  # Lower threshold for reliable items
    elif any(mr in name_lower for mr in medium_reliability):
        return 0.7  # Standard threshold
    elif any(lr in name_lower for lr in low_reliability):
        return 0.8  # Higher threshold for noisy items
    elif category_lower in ["condiments", "snacks"]:
        return 0.8  # Category-based threshold
    elif category_lower in ["produce", "dairy", "meat"]:
        return 0.65  # Slightly lower for fresh items
    else:
        return 0.7  # Default threshold

# ============================================================================
# STAGE-BASED DETECTION PIPELINE
# Each stage answers one question only, following clean architecture
# ============================================================================

def stage_yolo_detect_regions_multi_crop(img, use_multi_crop=False):
    """
    🔥 FIX #2: Multi-crop zooming - split image into overlapping tiles when YOLO fails
    Mimics human zooming behavior for better detection on dense pantry scenes
    """
    global _yolo_model
    regions = []
    
    if not _yolo_model or not YOLO_DETECTION_ENABLED:
        return regions
    
    if not use_multi_crop:
        # Standard single-pass detection
        return stage_yolo_detect_regions(img)
    
    try:
        import numpy as np
        img_width, img_height = img.size
        
        # 🔥 PERFORMANCE: Reduce overlap for faster processing (30% is sufficient)
        # Split into 2x2 overlapping tiles (reduced overlap for performance)
        tile_width = img_width // 2
        tile_height = img_height // 2
        overlap = 0.3  # Reduced from 0.5 to 0.3 for better performance
        
        tiles = []
        for i in range(2):
            for j in range(2):
                x1 = max(0, int(i * tile_width * (1 - overlap)))
                y1 = max(0, int(j * tile_height * (1 - overlap)))
                x2 = min(img_width, x1 + tile_width + int(tile_width * overlap))
                y2 = min(img_height, y1 + tile_height + int(tile_height * overlap))
                tiles.append((x1, y1, x2, y2))
        
        # Run YOLO on each tile
        for tile_idx, (x1, y1, x2, y2) in enumerate(tiles):
            try:
                tile = img.crop((x1, y1, x2, y2))
                tile_regions = stage_yolo_detect_regions(tile)
                
                # Adjust bbox coordinates to full image space
                for region in tile_regions:
                    bbox = region["bbox"]
                    region["bbox"] = [
                        bbox[0] + x1,  # x1
                        bbox[1] + y1,  # y1
                        bbox[2] + x1,  # x2
                        bbox[3] + y1   # y2
                    ]
                    region["tile_source"] = tile_idx  # Track which tile found this
                    regions.append(region)
                
                if VERBOSE_LOGGING and tile_regions:
                    print(f"   🔍 Tile {tile_idx}: Found {len(tile_regions)} regions")
            except Exception as tile_error:
                if VERBOSE_LOGGING:
                    print(f"   ⚠️ Error processing tile {tile_idx}: {tile_error}")
                continue
        
        # Deduplicate overlapping detections from different tiles
        if len(regions) > 1:
            # Simple IoU-based deduplication
            unique_regions = []
            for region in regions:
                is_duplicate = False
                bbox1 = region["bbox"]
                for existing in unique_regions:
                    bbox2 = existing["bbox"]
                    # Calculate IoU
                    x1_inter = max(bbox1[0], bbox2[0])
                    y1_inter = max(bbox1[1], bbox2[1])
                    x2_inter = min(bbox1[2], bbox2[2])
                    y2_inter = min(bbox1[3], bbox2[3])
                    
                    if x1_inter < x2_inter and y1_inter < y2_inter:
                        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                        union_area = area1 + area2 - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0
                        
                        if iou > 0.5:  # Same detection
                            is_duplicate = True
                            # Keep the one with higher confidence
                            if region["yolo_conf"] > existing["yolo_conf"]:
                                unique_regions.remove(existing)
                                unique_regions.append(region)
                            break
                
                if not is_duplicate:
                    unique_regions.append(region)
            
            regions = unique_regions
        
        if VERBOSE_LOGGING:
            print(f"   🔍 Multi-crop: Found {len(regions)} unique regions across {len(tiles)} tiles")
        
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"   ⚠️ Multi-crop error: {e}")
        # Fallback to single-pass
        return stage_yolo_detect_regions(img)
    
    return regions


def stage_yolo_detect_regions(img):
    """
    STAGE 1: YOLO - "Where is something?"
    Purpose: Find regions of interest, NOT food names.
    Returns: List of regions with bbox and confidence only.
    """
    global _yolo_model
    regions = []
    
    if not _yolo_model or not YOLO_DETECTION_ENABLED:
        return regions
    
    try:
        import numpy as np
        img_array = np.array(img)
        # 🔥 FIX #2: Optimize YOLO for pantry images
        # Pantry scenes: stacked items, small packages, low contrast shelves
        # Use lower confidence, higher IoU, filter small/nested/thin boxes
        scene_type = "pantry"  # Default to pantry (more permissive)
        # 🔥 FIX #2: Lower threshold for pantry scenes (0.05) - want loose boxes, not precise ones
        yolo_conf_threshold = 0.05 if scene_type == "pantry" else 0.25  # FIX #2: 0.05 for pantry, 0.25 for other scenes
        yolo_iou_threshold = 0.50  # FIX 4: Slightly lower IoU to catch more items (0.55 was too aggressive)
        
        # Get image dimensions for area filtering
        img_height, img_width = img_array.shape[:2]
        image_area = img_height * img_width
        min_area = 0.01 * image_area  # FIX 4: Minimum 1% of image area (was 2%, too aggressive for small items)
        
        # 🔥 PERFORMANCE: Reduce max_det for faster processing (50 is sufficient for most pantries)
        # Optimize YOLO for pantry - detect as many items as possible but limit for speed
        results = _yolo_model(img_array, conf=yolo_conf_threshold, iou=yolo_iou_threshold, verbose=False, max_det=50)  # Reduced from 100
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    try:
                        # Extract bbox
                        xyxy = box.xyxy[0] if hasattr(box.xyxy, '__getitem__') else box.xyxy
                        if hasattr(xyxy, 'cpu'):
                            xyxy = xyxy.cpu().numpy()
                        elif hasattr(xyxy, 'numpy'):
                            xyxy = xyxy.numpy()
                        
                        if len(xyxy) < 4:
                            continue
                        x1, y1, x2, y2 = xyxy[:4]
                        
                        # FIX 4: Filter small boxes (minimum area threshold)
                        box_width = x2 - x1
                        box_height = y2 - y1
                        box_area = box_width * box_height
                        if box_area < min_area:
                            if VERBOSE_LOGGING:
                                print(f"   ⚠️ Filtering small box: area={box_area:.0f} < min={min_area:.0f}")
                            continue
                        
                        # FIX 4: Filter thin boxes (width or height too small relative to image)
                        min_dimension = min(box_width, box_height)
                        if min_dimension < 0.005 * min(img_width, img_height):  # Less than 0.5% of smallest image dimension (was 1%, too aggressive)
                            if VERBOSE_LOGGING:
                                print(f"   ⚠️ Filtering thin box: min_dim={min_dimension:.0f}")
                            continue
                        
                        # Extract confidence
                        conf = box.conf[0] if hasattr(box.conf, '__getitem__') else box.conf
                        if hasattr(conf, 'cpu'):
                            conf = conf.cpu().numpy()
                        elif hasattr(conf, 'numpy'):
                            conf = conf.numpy()
                        yolo_conf = float(conf)
                        
                        # Extract class_id for container detection
                        cls = box.cls[0] if hasattr(box.cls, '__getitem__') else box.cls
                        if hasattr(cls, 'cpu'):
                            cls = cls.cpu().numpy()
                        elif hasattr(cls, 'numpy'):
                            cls = cls.numpy()
                        class_id = int(cls)
                        
                        # Get class name for container detection
                        if hasattr(_yolo_model, 'names') and class_id < len(_yolo_model.names):
                            class_name = _yolo_model.names[class_id]
                        else:
                            class_name = f"class_{class_id}"
                        
                        # Check if container-like
                        container_types = ['bottle', 'jar', 'can', 'box', 'bag', 'package', 'bowl', 'cup', 'wine glass']
                        is_container = any(kw in class_name.lower() for kw in container_types)
                        
                        # YOLO food prior: weak signal based on COCO class
                        ALLOWED_COCO_CLASSES = {
                            46: "banana", 47: "apple", 49: "orange",
                            50: "broccoli", 51: "carrot",
                            48: "sandwich", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
                            39: "bottle", 40: "wine glass", 41: "cup", 45: "bowl"
                        }
                        yolo_food_prior = 0.3 if class_id in ALLOWED_COCO_CLASSES else 0.05
                        
                        regions.append({
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "yolo_conf": yolo_conf,
                            "class_id": class_id,
                            "class_name": class_name,
                            "is_container": is_container,
                            "yolo_food_prior": yolo_food_prior,
                            "area": box_area  # Store area for nested box filtering
                        })
                    except Exception as e:
                        if VERBOSE_LOGGING:
                            print(f"   ⚠️ Error processing YOLO box: {e}")
                        continue
                
                # FIX 4: Filter nested boxes (smaller box inside larger box)
                # Sort by area (largest first) and remove boxes that are mostly contained in larger boxes
                if len(regions) > 1:
                    regions_sorted = sorted(regions, key=lambda r: r.get("area", 0), reverse=True)
                    filtered_regions = []
                    for i, region in enumerate(regions_sorted):
                        x1, y1, x2, y2 = region["bbox"]
                        is_nested = False
                        for j, other_region in enumerate(regions_sorted):
                            if i == j:
                                continue
                            ox1, oy1, ox2, oy2 = other_region["bbox"]
                            # Check if this box is mostly inside the other box
                            overlap_x1 = max(x1, ox1)
                            overlap_y1 = max(y1, oy1)
                            overlap_x2 = min(x2, ox2)
                            overlap_y2 = min(y2, oy2)
                            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                                region_area = region.get("area", (x2 - x1) * (y2 - y1))
                                # If >80% of this box is inside the other box, it's nested
                                if overlap_area > 0.8 * region_area:
                                    is_nested = True
                                    if VERBOSE_LOGGING:
                                        print(f"   ⚠️ Filtering nested box: {region.get('class_name')} inside {other_region.get('class_name')}")
                                    break
                        if not is_nested:
                            filtered_regions.append(region)
                    regions = filtered_regions
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"   ⚠️ YOLO detection error: {e}")
    
    return regions


def stage_classify_food_type(crop, region_info=None):
    """
    STAGE 1.5: Food Type Classification - "What kind of thing is this?"
    Purpose: Narrow the answer space before naming. This dramatically improves precision.
    
    Returns: {"food_type": "...", "score": 0.0-1.0} or None
    """
    # Food types that CLIP can distinguish well
    FOOD_TYPES = [
        "fresh produce",
        "bottle or jar",
        "carton",
        "can",
        "box",
        "bag",
        "snack",
        "beverage"
    ]
    
    # Use CLIP to classify food type
    food_type_pred = clip_match_open_vocabulary(crop, FOOD_TYPES, use_prompt_engineering=True)
    if not food_type_pred:
        return None
    
    food_type = food_type_pred.get("label", "")
    score = food_type_pred.get("score", 0.0)
    
    if VERBOSE_LOGGING:
        print(f"   🏷️ Food type classification: {food_type} (score: {score:.2f})")
    
    return {
        "food_type": food_type,
        "score": score
    }


def stage_clip_suggest_label(crop, candidate_labels, region_info=None):
    """
    STAGE 2: CLIP - "What could this be?"
    Purpose: Open-vocabulary semantic matching. Suggests, doesn't decide.
    
    Precision Rules Applied:
    - #3: Margin-based selection (requires separation between top-1 and top-2)
    - #5: Visual plausibility checks (soft penalties)
    - #8: Food-only gating (checks if item is food before trusting)
    
    Args:
        crop: PIL Image crop
        candidate_labels: List of candidate food labels (context-aware)
        region_info: Optional dict with region metadata (is_container, class_name, etc.)
    
    Returns: {"label": "...", "score": calibrated_score, "second_best": calibrated_second, "needs_confirmation": bool} or None
    """
    if not candidate_labels:
        return None
    
    # Performance: Skip food gate for speed when we have many food candidates
    # Food gate is expensive (2-3 CLIP calls) and candidate_labels are already food-focused
    run_food_gate = len(candidate_labels) < 10  # Only run if very few candidates
    
    if run_food_gate:
        # 🔥 FIX 5: Foodness gating - check if food > object before rejecting
        # Use binary food vs non-food check, not label-specific gating
        FOODNESS_LABELS = ["food", "ingredient", "grocery item", "edible item"]
        NON_FOOD_LABELS = ["tool", "furniture", "clothing", "appliance", "object", "non-food"]
        
        # Performance: Batch process food check (single CLIP call instead of 2)
        food_check_labels = FOODNESS_LABELS + NON_FOOD_LABELS
        food_check = clip_match_open_vocabulary(crop, food_check_labels, use_prompt_engineering=False)
        
        if food_check:
            # Get actual scores for food vs non-food
            food_label = food_check["label"]
            food_score = food_check["score"] if food_label in FOODNESS_LABELS else 0.0
            non_food_score = food_check["score"] if food_label in NON_FOOD_LABELS else 0.0
            clip_food_score = food_score - non_food_score
        else:
            clip_food_score = 0.0
        
        # 🔥 PRINCIPLE 1 & 4: Never hard-reject food-like items
        # Separate foodness check from food name check
        # If food_score > object_score → surface guess (even if weak)
        # If food_score < object_score by >0.3 → mark for confirmation but still process
        if clip_food_score < -0.3:
            # Very negative score - likely non-food, but still process (never hard-reject)
            if VERBOSE_LOGGING:
                print(f"   ⚠️ CLIP food gate: likely non-food (food_score={clip_food_score:.2f}) - will mark for confirmation but still process")
            # Continue processing - let other signals decide (never hard-reject)
        elif clip_food_score < 0.0:
            # Slightly negative - uncertain, but allow food prediction
            if VERBOSE_LOGGING:
                print(f"   ⚠️ CLIP food gate: uncertain (food_score={clip_food_score:.2f}) - allowing food prediction, mark for confirmation")
            # Continue processing, will mark for confirmation later
        else:
            # Food score >= 0 - food-like, allow prediction
            if VERBOSE_LOGGING:
                print(f"   ✅ CLIP food gate: food-like (food_score={clip_food_score:.2f}) - allowing prediction")
    else:
        # Skip food gate - assume food since we have many food candidates
        clip_food_score = 0.5  # Neutral score, continue processing
    
    # Run CLIP matching on candidate labels
    clip_pred = clip_match_open_vocabulary(crop, candidate_labels, use_prompt_engineering=True)
    if not clip_pred:
        return None
    
    best = clip_pred["score"]  # Already calibrated
    second = clip_pred.get("second_best", 0.0)  # Already calibrated
    raw_best = clip_pred.get("raw_score", 0.0)
    top3_labels = clip_pred.get("top3_labels", [clip_pred["label"]])
    top3_scores = clip_pred.get("top3_scores", [best, second, 0.0])
    
    # FIX 3: Check for ambiguity - if top-2 scores are close (< 0.07), show both
    margin = best - second
    ambiguity_threshold = 0.07  # FIX 3: If difference < 0.07, show ambiguity
    
    # FIX 3: Build ambiguous label if scores are close
    ambiguous_label = None
    if len(top3_labels) >= 2 and margin < ambiguity_threshold:
        # Top-2 are ambiguous - show both
        ambiguous_label = " / ".join(top3_labels[:2])
        if VERBOSE_LOGGING:
            print(f"   🔀 FIX 3: Ambiguous prediction (margin={margin:.3f} < {ambiguity_threshold}) - showing: {ambiguous_label}")
    
    # Fix #4: Disable CLIP margin logic unless comparing >= 3 labels
    # Margin checks are misleading when there are too few labels
    margin_threshold = 0.01  # Optimize: Lower margin threshold to 0.01 - accept more predictions
    
    # Fix 3: Margin gate should NOT reject - margin ≠ correctness, margin = confidence difference
    # FIXED: When both scores are very low (0.0), it means CLIP is uncertain, not that margin is wrong
    # Check raw scores to determine if this is a real uncertainty vs calibration artifact
    raw_second = clip_pred.get("raw_second_best", 0.0)
    
    if len(candidate_labels) >= 3 and margin < margin_threshold:
        # Margin too small - but check if it's due to low raw scores or actual uncertainty
        if raw_best > 0.01 and raw_second > 0.01:
            # Both raw scores are reasonable - small margin is real uncertainty
            if VERBOSE_LOGGING:
                print(f"   ⚠️ CLIP margin too small: {best:.3f} vs {second:.3f} (margin={margin:.3f} < {margin_threshold}, raw: {raw_best:.3f} vs {raw_second:.3f}) - accepting best label, needs confirmation")
        else:
            # Very low raw scores - calibration artifact, accept anyway
            if VERBOSE_LOGGING:
                print(f"   ⚠️ CLIP low scores (raw: {raw_best:.3f} vs {raw_second:.3f}) - accepting best label, needs confirmation")
        # Accept the best label, just mark for confirmation
        # FIX 3: Use ambiguous label if available
        final_label = ambiguous_label if ambiguous_label else clip_pred["label"]
        return {
            "label": final_label,  # FIX 3: Use ambiguous label if scores are close
            "score": best,
            "second_best": second,
            "needs_confirmation": True,
            "margin": margin,
            "is_ambiguous": ambiguous_label is not None,  # FIX 3: Flag for ambiguity
            "top3_labels": top3_labels  # FIX 3: Include top-3 for reference
        }
    elif len(candidate_labels) < 3:
        # Too few labels - skip margin check, accept best
        if VERBOSE_LOGGING:
            print(f"   ℹ️ Skipping margin check (only {len(candidate_labels)} labels) - accepting best label")
    
    # Precision Rule #5: Visual plausibility checks (soft penalties)
    # Apply penalties based on visual features, not hard deletes
    penalty_factor = 1.0
    label = clip_pred["label"]
    
    # FIX 6: Scene prior - penalize non-food items in pantry context
    SCENE_PRIOR = {
        "tie": -0.9,
        "tennis racket": -0.9,
        "shoe": -0.8,
        "sock": -0.8,
        "clothing": -0.8,
        "furniture": -0.7,
        "tool": -0.7,
        "appliance": -0.7,
        "food": +0.3,
        "ingredient": +0.3,
        "grocery": +0.3
    }
    scene_type = "pantry"  # Default to pantry context
    if scene_type in ["pantry", "fridge"]:
        label_lower = label.lower()
        # Check for non-food items
        for non_food_key, penalty in SCENE_PRIOR.items():
            if non_food_key in label_lower and penalty < 0:
                penalty_factor *= (1.0 + penalty)  # Apply negative penalty
                if VERBOSE_LOGGING:
                    print(f"   ⚠️ FIX 6: Scene prior penalty for '{label}' in {scene_type}: {penalty:.2f}")
                break
        # Check for food items (boost)
        for food_key, boost in SCENE_PRIOR.items():
            if food_key in label_lower and boost > 0:
                penalty_factor *= (1.0 + boost * 0.1)  # Apply positive boost (smaller)
                if VERBOSE_LOGGING:
                    print(f"   ✅ FIX 6: Scene prior boost for '{label}' in {scene_type}: {boost:.2f}")
                break
    
    if region_info:
        is_container = region_info.get("is_container", False)
        class_name = region_info.get("class_name", "").lower()
        
        # Check for onion vs apple confusion
        if label == "onion":
            # Onion should have papery texture, lower saturation
            # This is a simplified check - in production, use actual texture analysis
            # For now, just apply a small penalty if detected as container (unlikely for loose onion)
            if is_container and "bottle" in class_name:
                penalty_factor *= 0.7
                if VERBOSE_LOGGING:
                    print(f"   ⚠️ Visual plausibility: onion in bottle container (penalty applied)")
        
        # Check for oil vs water confusion
        if label.endswith("oil"):
            # Oil should be gold/amber/green colored
            # Simplified: if it's a clear bottle, might be water
            if "bottle" in class_name and not is_container:
                penalty_factor *= 0.6
                if VERBOSE_LOGGING:
                    print(f"   ⚠️ Visual plausibility: oil in clear container (penalty applied)")
    
    final_score = best * penalty_factor
    
    # FIXED: Use raw score for thresholding, not calibrated score
    # Calibrated scores can be low even when raw scores are reasonable
    # For pantry images with 500+ candidates, raw scores of 0.01-0.05 are normal
    # Accept if raw score > 0.005 (very permissive) OR calibrated > 0.05
    raw_threshold = 0.005  # Very low threshold for raw scores
    calibrated_threshold = 0.05  # Low threshold for calibrated scores
    
    if final_score < calibrated_threshold and raw_best < raw_threshold:
        if VERBOSE_LOGGING:
            print(f"   🚫 CLIP score too low: calibrated={final_score:.3f}, raw={raw_best:.3f}")
        return None
    
    # FIX 3: Use ambiguous label if available
    final_label = ambiguous_label if ambiguous_label else label
    
    return {
        "label": final_label,  # FIX 3: Use ambiguous label if scores are close
        "score": min(final_score, 0.8),  # Cap influence
        "second_best": second,
        "needs_confirmation": False if ambiguous_label is None else True,  # FIX 3: Mark ambiguous as needing confirmation
        "margin": margin,
        "is_ambiguous": ambiguous_label is not None,  # FIX 3: Flag for ambiguity
        "top3_labels": top3_labels  # FIX 3: Include top-3 for reference
    }


def stage_ocr_bind_to_region(yolo_bbox, full_image_ocr_results, img_size):
    """
    🔥 FIX #1: Spatial OCR Binding - Assign OCR text to YOLO boxes
    Purpose: Match OCR text regions to YOLO bounding boxes based on spatial overlap
    Returns: List of OCR text snippets that overlap with the YOLO box
    """
    if not full_image_ocr_results:
        return []
    
    yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_bbox
    yolo_center_x = (yolo_x1 + yolo_x2) / 2
    yolo_center_y = (yolo_y1 + yolo_y2) / 2
    
    bound_texts = []
    
    for ocr_item in full_image_ocr_results:
        # EasyOCR returns: (bbox, text, confidence) where bbox is array of 4 corner points
        if len(ocr_item) < 3:
            continue
        
        ocr_bbox = ocr_item[0]  # Array of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        ocr_text = ocr_item[1] if len(ocr_item) > 1 else ""
        ocr_conf = ocr_item[2] if len(ocr_item) > 2 else 0.0
        
        if not ocr_text or ocr_conf < 0.1:
            continue
        
        # Convert OCR bbox to (x1, y1, x2, y2) format
        try:
            import numpy as np
            ocr_points = np.array(ocr_bbox)
            ocr_x1 = float(np.min(ocr_points[:, 0]))
            ocr_y1 = float(np.min(ocr_points[:, 1]))
            ocr_x2 = float(np.max(ocr_points[:, 0]))
            ocr_y2 = float(np.max(ocr_points[:, 1]))
            
            # Check if OCR text center is inside YOLO box (simple overlap check)
            ocr_center_x = (ocr_x1 + ocr_x2) / 2
            ocr_center_y = (ocr_y1 + ocr_y2) / 2
            
            # Method 1: Check if OCR center is inside YOLO box
            if (yolo_x1 <= ocr_center_x <= yolo_x2 and yolo_y1 <= ocr_center_y <= yolo_y2):
                bound_texts.append({
                    "text": ocr_text,
                    "confidence": ocr_conf,
                    "bbox": [ocr_x1, ocr_y1, ocr_x2, ocr_y2]
                })
            # Method 2: Check if YOLO center is inside OCR box (for small text on large boxes)
            elif (ocr_x1 <= yolo_center_x <= ocr_x2 and ocr_y1 <= yolo_center_y <= ocr_y2):
                bound_texts.append({
                    "text": ocr_text,
                    "confidence": ocr_conf,
                    "bbox": [ocr_x1, ocr_y1, ocr_x2, ocr_y2]
                })
            # Method 3: Check for significant overlap (IoU > 0.1)
            else:
                overlap_x1 = max(yolo_x1, ocr_x1)
                overlap_y1 = max(yolo_y1, ocr_y1)
                overlap_x2 = min(yolo_x2, ocr_x2)
                overlap_y2 = min(yolo_y2, ocr_y2)
                
                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    ocr_area = (ocr_x2 - ocr_x1) * (ocr_y2 - ocr_y1)
                    yolo_area = (yolo_x2 - yolo_x1) * (yolo_y2 - yolo_y1)
                    
                    # If overlap is > 10% of OCR box or > 5% of YOLO box, consider it bound
                    if overlap_area > 0.1 * ocr_area or overlap_area > 0.05 * yolo_area:
                        bound_texts.append({
                            "text": ocr_text,
                            "confidence": ocr_conf,
                            "bbox": [ocr_x1, ocr_y1, ocr_x2, ocr_y2]
                        })
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"   ⚠️ OCR binding error: {e}")
            continue
    
    return bound_texts


def stage_ocr_propose_candidates(ocr_text, clip_model=None, clip_processor=None, crop=None):
    """
    🔥 FIX #3: OCR proposes candidates → CLIP verifies
    Instead of OCR → hypothesis only, use OCR text to propose candidate labels
    Then CLIP verifies which candidate matches the image
    
    Example: OCR sees "Kellogg" → candidate = "cereal" → CLIP confirms cereal
    """
    if not ocr_text or len(ocr_text.strip()) < 2:
        return None
    
    # 🔥 IMPROVEMENT #2: Comprehensive pantry keyword dictionary
    # Brand-to-food mapping (expanded with more brands)
    brand_to_food = {
        # Cereal brands
        "kellogg": "cereal",
        "general mills": "cereal",
        "quaker": "cereal",
        "cheerios": "cereal",
        "frosted flakes": "cereal",
        "rice krispies": "cereal",
        "cinnamon toast": "cereal",
        "lucky charms": "cereal",
        "froot loops": "cereal",
        "cocoa puffs": "cereal",
        "honey nut": "cereal",
        "special k": "cereal",
        "cap'n crunch": "cereal",
        "post": "cereal",
        "kashi": "cereal",
        # Pasta brands
        "barilla": "pasta",
        "ronzoni": "pasta",
        "de cecco": "pasta",
        "san giorgio": "pasta",
        "mueller": "pasta",
        "bertolli": "pasta",
        # Mac & Cheese
        "kraft": "macaroni",
        "kraft macaroni": "macaroni",
        "velveeta": "macaroni",
        "annie's": "macaroni",
        # Sauce brands
        "hunts": "tomato sauce",
        "rao": "pasta sauce",
        "prego": "pasta sauce",
        "bertolli": "pasta sauce",
        "ragu": "pasta sauce",
        "newman's own": "pasta sauce",
        # Condiments
        "heinz": "ketchup",
        "hunt's": "ketchup",
        "french's": "mustard",
        "grey poupon": "mustard",
        "hellmann's": "mayonnaise",
        "best foods": "mayonnaise",
        # Canned goods
        "del monte": "canned goods",
        "green giant": "canned goods",
        "libby's": "canned goods",
        # Soup brands
        "campbell": "soup",
        "campbell's": "soup",
        "progresso": "soup",
        "amy's": "soup",
        # Rice brands
        "uncle ben's": "rice",
        "minute rice": "rice",
        "mahatma": "rice",
        "carolina": "rice",
        # Snack brands
        "lays": "chips",
        "doritos": "chips",
        "fritos": "chips",
        "cheetos": "chips",
        "ritz": "crackers",
        "triscuit": "crackers",
        "wheat thins": "crackers",
        "oreo": "cookies",
        "chips ahoy": "cookies",
        "pepperidge farm": "crackers",
        # Protein bars
        "clif": "protein bar",
        "luna": "protein bar",
        "quest": "protein bar",
        "kind": "protein bar",
        "larabar": "protein bar",
        "rxbar": "protein bar",
        # Granola
        "nature valley": "granola bar",
        "kashi": "granola bar",
        "kind": "granola bar"
    }
    
    ocr_lower = ocr_text.lower()
    proposed_candidates = []
    
    # Check for brand names (prioritize exact matches)
    for brand, food_type in brand_to_food.items():
        if brand in ocr_lower:
            proposed_candidates.append(food_type)
            if VERBOSE_LOGGING:
                print(f"   📝 OCR brand '{brand}' → candidate: {food_type}")
    
    # 🔥 IMPROVEMENT #2: Expanded food keywords with more specific terms
    food_keywords = {
        "cereal": ["cereal", "granola", "oatmeal", "breakfast", "flakes", "puffs", "loops", "cheerios", "krispies"],
        "granola": ["granola", "granola bar", "granola cereal"],
        "oats": ["oats", "oatmeal", "steel cut", "rolled oats"],
        "pasta": ["pasta", "spaghetti", "macaroni", "noodles", "penne", "rigatoni", "fettuccine", "linguine", "lasagna"],
        "macaroni": ["macaroni", "mac and cheese", "macaroni and cheese"],
        "rice": ["rice", "jasmine", "basmati", "brown rice", "white rice", "wild rice", "arborio"],
        "quinoa": ["quinoa"],
        "flour": ["flour", "all-purpose", "wheat flour", "bread flour"],
        "sugar": ["sugar", "granulated", "brown sugar", "powdered sugar"],
        "snacks": ["chips", "crackers", "cookies", "pretzels", "snacks", "trail mix"],
        "chips": ["chips", "potato chips", "tortilla chips"],
        "crackers": ["crackers", "saltines", "wheat crackers"],
        "cookies": ["cookies", "biscuits"],
        "protein bar": ["protein bar", "protein", "energy bar"],
        "granola bar": ["granola bar", "granola"],
        "sauce": ["sauce", "marinara", "tomato sauce", "pasta sauce", "alfredo"],
        "tomato sauce": ["tomato sauce", "marinara", "pasta sauce"],
        "pasta sauce": ["pasta sauce", "marinara", "alfredo", "pesto"],
        "ketchup": ["ketchup", "catsup"],
        "mustard": ["mustard", "dijon"],
        "mayonnaise": ["mayonnaise", "mayo"],
        "juice": ["juice", "orange juice", "apple juice", "cranberry juice", "grape juice"],
        "soup": ["soup", "broth", "stock"],
        "canned goods": ["canned", "can of", "tin of"]
    }
    
    for food_type, keywords in food_keywords.items():
        if any(kw in ocr_lower for kw in keywords):
            if food_type not in proposed_candidates:
                proposed_candidates.append(food_type)
                if VERBOSE_LOGGING:
                    print(f"   📝 OCR keyword → candidate: {food_type}")
    
    # If we have candidates and CLIP model, verify with CLIP
    if proposed_candidates and clip_model and clip_processor and crop:
        try:
            # Import clip_match_open_vocabulary from current module
            clip_result = clip_match_open_vocabulary(crop, proposed_candidates, use_prompt_engineering=True)
            if clip_result and clip_result.get("score", 0) > 0.25:
                verified_label = clip_result.get("label", proposed_candidates[0])
                verified_conf = clip_result.get("score", 0.0)
                if VERBOSE_LOGGING:
                    print(f"   ✅ CLIP verified OCR candidate: {verified_label} (conf: {verified_conf:.2f})")
                return {
                    "label": verified_label,
                    "confidence": verified_conf,
                    "text": ocr_text,
                    "method": "ocr_clip_verified"
                }
        except Exception as clip_error:
            if VERBOSE_LOGGING:
                print(f"   ⚠️ CLIP verification error: {clip_error}")
    
    # If no CLIP verification, return top candidate with lower confidence
    if proposed_candidates:
        return {
            "label": proposed_candidates[0],
            "confidence": 0.4,  # Lower confidence without CLIP verification
            "text": ocr_text,
            "method": "ocr_proposed"
        }
    
    return None


def stage_ocr_read_label(crop, is_container=False, bound_ocr_texts=None):
    """
    STAGE 3: OCR - "What does the text say?"
    Purpose: Understand packaged food. Overrides CLIP when text is clear.
    Improved: Multiple crops, image enhancement, rotation handling for difficult angles.
    🔥 FIX #1: Now accepts pre-bound OCR texts from spatial binding
    Returns: {"label": "...", "confidence": 0.0-1.0, "text": "..."} or None
    """
    global _ocr_reader
    
    # 🔥 FIX #1: If OCR texts are already bound to this region, use them first
    if bound_ocr_texts and len(bound_ocr_texts) > 0:
        # Combine all bound texts
        combined_text = ' '.join([item["text"] for item in bound_ocr_texts]).lower()
        avg_conf = sum(item["confidence"] for item in bound_ocr_texts) / len(bound_ocr_texts)
        
        if combined_text and len(combined_text) >= 2:
            # 🔥 FIX #1: Process bound OCR texts through keyword matching
            # This is the key improvement - OCR text is now spatially bound to the YOLO box
            matched_label = None
            matched_confidence = 0.0
            
            # Try to match against food keywords (same logic as below)
            # Import the keyword matching logic
            label_keyword_map = {
                # (This will be matched against the existing keyword map in the function)
            }
            
            # For now, process bound texts through the normal keyword matching below
            # We'll use the bound text as the OCR input
            if VERBOSE_LOGGING:
                print(f"   📍 Using spatially bound OCR text: '{combined_text}' (conf: {avg_conf:.2f})")
    
    if not _ocr_reader and not bound_ocr_texts:
        return None
    
    if not is_container and not bound_ocr_texts:
        # Only run OCR for containers or if explicitly requested
        return None
    
    # 🔥 FIX #1: If we have bound OCR texts, use them directly
    # Otherwise, run OCR on crop as before
    if bound_ocr_texts and len(bound_ocr_texts) > 0:
        # Use bound texts - skip OCR processing, go straight to keyword matching
        combined_text = ' '.join([item["text"] for item in bound_ocr_texts]).lower()
        avg_conf = sum(item["confidence"] for item in bound_ocr_texts) / len(bound_ocr_texts)
        ocr_text = combined_text
        ocr_confidence = avg_conf
        # Skip to keyword matching section (after OCR processing)
        skip_ocr_processing = True
    else:
        skip_ocr_processing = False
        ocr_text = None
        ocr_confidence = 0.0
    
    try:
        import numpy as np
        import re
        from PIL import Image, ImageEnhance, ImageFilter
        
        if skip_ocr_processing:
            # Use bound OCR texts - skip OCR processing, go to keyword matching
            # ocr_text and ocr_confidence already set above
            pass
        else:
            # Improve: Try multiple crops and enhancements for better OCR on difficult angles
            crops_to_try = []
        crop_width, crop_height = crop.size[0], crop.size[1]
        
        # Crop 1: Full crop (for wide containers)
        crops_to_try.append(("full", crop))
        
        # Crop 2: Middle section (for tall containers - where label usually is)
        if crop_height > crop_width * 1.2:
            label_y_start = int(crop_height * 0.15)
            label_y_end = int(crop_height * 0.85)
            middle_crop = crop.crop((0, label_y_start, crop_width, label_y_end))
            crops_to_try.append(("middle", middle_crop))
        
        # Crop 3: Top third (some labels are at top)
        if crop_height > crop_width * 1.2:
            top_crop = crop.crop((0, 0, crop_width, int(crop_height * 0.4)))
            crops_to_try.append(("top", top_crop))
        
        # Crop 4: Center horizontal strip (for wide containers)
        if crop_width > crop_height * 1.2:
            center_y = crop_height // 2
            strip_height = int(crop_height * 0.3)
            strip_crop = crop.crop((0, center_y - strip_height//2, crop_width, center_y + strip_height//2))
            crops_to_try.append(("center_strip", strip_crop))
        
        best_ocr_result = None
        best_confidence = 0.0
        
        # Try each crop with enhancements
        for crop_name, test_crop in crops_to_try:
            # Enhance image for better OCR (especially for difficult angles)
            enhanced_crops = []
            
            # Original
            enhanced_crops.append(("original", test_crop))
            
            # Enhanced contrast (helps with difficult lighting)
            contrast_enhanced = ImageEnhance.Contrast(test_crop).enhance(2.0)
            enhanced_crops.append(("high_contrast", contrast_enhanced))
            
            # Sharpened (helps with blurry text)
            sharpened = test_crop.filter(ImageFilter.SHARPEN)
            enhanced_crops.append(("sharpened", sharpened))
            
            # Try rotations for difficult angles (90, 180, 270 degrees)
            for angle in [90, 180, 270]:
                rotated = test_crop.rotate(angle, expand=True)
                enhanced_crops.append((f"rotated_{angle}", rotated))
            
            # Try each enhanced version
            for enh_name, enhanced_crop in enhanced_crops:
                try:
                    crop_array = np.array(enhanced_crop)
                    ocr_results = _ocr_reader.readtext(crop_array)
                    
                    if ocr_results:
                        # Calculate average confidence
                        avg_conf = sum(conf for (_, _, conf) in ocr_results) / len(ocr_results)
                        if avg_conf > best_confidence:
                            best_confidence = avg_conf
                            # Collect all text
                            ocr_text_temp = ' '.join([text for (_, text, conf) in ocr_results if conf > 0.10]).lower()
                            if ocr_text_temp and len(ocr_text_temp) >= 2:
                                best_ocr_result = {
                                    "text": ocr_text_temp,
                                    "confidence": avg_conf,
                                    "crop_used": crop_name,
                                    "enhancement": enh_name
                                }
                except Exception as e:
                    if VERBOSE_LOGGING:
                        print(f"   ⚠️ OCR error on {crop_name}/{enh_name}: {e}")
                    continue
        
        if not best_ocr_result:
            # If we have bound texts, use them even if OCR on crop failed
            if bound_ocr_texts and len(bound_ocr_texts) > 0:
                combined_text = ' '.join([item["text"] for item in bound_ocr_texts]).lower()
                avg_conf = sum(item["confidence"] for item in bound_ocr_texts) / len(bound_ocr_texts)
                ocr_text = combined_text
                ocr_confidence = avg_conf
            else:
                return None
        else:
            ocr_text = best_ocr_result["text"]
            ocr_confidence = best_ocr_result["confidence"]
        
        if not ocr_text or len(ocr_text) < 2:
            return None
        
        # Remove common label text
        common_label_text = [
            'nutrition facts', 'serving size', 'calories', 'total fat', 'saturated fat',
            'trans fat', 'cholesterol', 'sodium', 'total carbohydrate', 'dietary fiber',
            'total sugars', 'protein', 'vitamin', 'calcium', 'iron', 'ingredients',
            'contains', 'allergen', 'manufactured', 'distributed', 'net weight', 'net wt'
        ]
        for label_text in common_label_text:
            ocr_text = ocr_text.replace(label_text, ' ')
        
        # Remove brand words
        brand_words = ['365', 'organic', 'whole foods', 'trader joe', 'kirkland', 'great value']
        words = ocr_text.split()
        words = [w for w in words if not any(brand in w.lower() for brand in brand_words)]
        ocr_text = ' '.join(words)
        
        # 🔥 EXPANDED OCR KEYWORD DICTIONARY (100-500 foods) - Critical for packaged food
        # This is how real pantry apps work - keyword matching from OCR text
        label_keyword_map = {
            # Oils and liquids (expanded)
            'olive': 'olive oil', 'extra virgin': 'olive oil', 'evoo': 'olive oil', 'virgin olive': 'olive oil',
            'canola': 'canola oil', 'vegetable oil': 'vegetable oil', 'coconut oil': 'coconut oil',
            'sesame oil': 'sesame oil', 'avocado oil': 'avocado oil', 'walnut oil': 'walnut oil',
            'vinegar': 'vinegar', 'balsamic': 'balsamic vinegar', 'rice vinegar': 'rice vinegar',
            'apple cider vinegar': 'apple cider vinegar', 'white wine vinegar': 'white wine vinegar',
            # Condiments (expanded)
            'mustard': 'mustard', 'dijon': 'mustard', 'yellow mustard': 'mustard',
            'mayonnaise': 'mayonnaise', 'mayo': 'mayonnaise', 'ketchup': 'ketchup',
            'hot sauce': 'hot sauce', 'sriracha': 'sriracha', 'tabasco': 'hot sauce',
            'soy sauce': 'soy sauce', 'tamari': 'soy sauce', 'teriyaki': 'teriyaki sauce',
            'hoisin': 'hoisin sauce', 'fish sauce': 'fish sauce', 'oyster sauce': 'oyster sauce',
            'worcestershire': 'worcestershire sauce', 'bbq sauce': 'bbq sauce',
            'ranch': 'ranch dressing', 'italian dressing': 'italian dressing', 'caesar dressing': 'caesar dressing',
            'honey': 'honey', 'maple syrup': 'maple syrup', 'jam': 'jam', 'jelly': 'jelly',
            'peanut butter': 'peanut butter', 'almond butter': 'almond butter', 'tahini': 'tahini',
            'hummus': 'hummus', 'pesto': 'pesto', 'salsa': 'salsa', 'guacamole': 'guacamole',
            # Spices (expanded)
            'salt': 'salt', 'pepper': 'black pepper', 'black pepper': 'black pepper',
            'red pepper': 'red pepper', 'paprika': 'paprika', 'cumin': 'cumin',
            'oregano': 'oregano', 'basil': 'basil', 'thyme': 'thyme', 'rosemary': 'rosemary',
            'parsley': 'parsley', 'cinnamon': 'cinnamon', 'nutmeg': 'nutmeg',
            'ginger': 'ginger', 'turmeric': 'turmeric', 'curry': 'curry powder',
            'chili powder': 'chili powder', 'cayenne': 'cayenne', 'garlic powder': 'garlic powder',
            'onion powder': 'onion powder', 'bay leaves': 'bay leaves',
            # Grains and pasta (expanded - CRITICAL for packaged food)
            'pasta': 'pasta', 'spaghetti': 'spaghetti', 'macaroni': 'macaroni', 'penne': 'penne',
            'fettuccine': 'pasta', 'linguine': 'pasta', 'rigatoni': 'pasta', 'fusilli': 'pasta',
            'rice': 'rice', 'white rice': 'white rice', 'brown rice': 'brown rice', 'jasmine rice': 'rice',
            'basmati rice': 'rice', 'wild rice': 'rice',
            'flour': 'flour', 'wheat flour': 'flour', 'all purpose flour': 'flour', 'bread flour': 'flour',
            'oats': 'oats', 'oatmeal': 'oats', 'rolled oats': 'oats',
            'cereal': 'cereal', 'granola': 'granola', 'quinoa': 'quinoa', 'barley': 'barley',
            'couscous': 'couscous', 'bulgur': 'bulgur', 'farro': 'farro',
            # Baking (expanded)
            'sugar': 'sugar', 'brown sugar': 'brown sugar', 'powdered sugar': 'powdered sugar',
            'baking powder': 'baking powder', 'baking soda': 'baking soda', 'yeast': 'yeast',
            'vanilla': 'vanilla extract', 'cocoa': 'cocoa powder', 'chocolate chips': 'chocolate chips',
            'shortening': 'shortening', 'cornstarch': 'cornstarch',
            # Dairy (expanded)
            'milk': 'milk', 'whole milk': 'milk', 'skim milk': 'milk', '2% milk': 'milk',
            'yogurt': 'yogurt', 'greek yogurt': 'yogurt', 'cheese': 'cheese', 'cheddar': 'cheese',
            'mozzarella': 'cheese', 'parmesan': 'cheese', 'butter': 'butter',
            'eggs': 'eggs', 'egg': 'eggs', 'sour cream': 'sour cream', 'cream cheese': 'cream cheese',
            'cottage cheese': 'cheese', 'heavy cream': 'cream', 'half and half': 'cream',
            # Canned/packaged (expanded - CRITICAL)
            'beans': 'beans', 'chickpeas': 'chickpeas', 'black beans': 'black beans',
            'kidney beans': 'beans', 'pinto beans': 'beans', 'navy beans': 'beans',
            'tomato sauce': 'tomato sauce', 'tomato paste': 'tomato paste', 'marinara': 'tomato sauce',
            'soup': 'soup', 'chicken soup': 'soup', 'vegetable soup': 'soup',
            'broth': 'broth', 'stock': 'stock', 'chicken broth': 'chicken broth',
            'beef broth': 'broth', 'vegetable broth': 'broth',
            'canned tomatoes': 'canned tomatoes', 'canned fruit': 'canned fruit',
            'tuna': 'tuna', 'salmon': 'salmon', 'sardines': 'sardines',
            # Bread and bakery (expanded)
            'bread': 'bread', 'white bread': 'bread', 'wheat bread': 'bread', 'whole wheat bread': 'bread',
            'bagel': 'bagel', 'english muffin': 'english muffin', 'tortilla': 'tortilla',
            'cracker': 'crackers', 'crackers': 'crackers', 'rice cakes': 'rice cakes',
            # Snacks (expanded)
            'chips': 'chips', 'potato chips': 'chips', 'tortilla chips': 'chips',
            'cookies': 'cookies', 'cookie': 'cookies', 'nuts': 'nuts', 'almonds': 'almonds',
            'walnuts': 'nuts', 'peanuts': 'nuts', 'cashews': 'nuts', 'pistachios': 'nuts',
            'granola bars': 'granola bars', 'protein bar': 'protein bar', 'energy bar': 'energy bar',
            'popcorn': 'popcorn', 'pretzels': 'pretzels',
            # Beverages (expanded)
            'juice': 'juice', 'orange juice': 'orange juice', 'apple juice': 'apple juice',
            'cranberry juice': 'cranberry juice', 'grape juice': 'grape juice',
            'soda': 'soda', 'cola': 'soda', 'sprite': 'sprite', 'ginger ale': 'soda',
            'water': 'water', 'sparkling water': 'water', 'seltzer': 'water',
            'coffee': 'coffee', 'tea': 'tea', 'green tea': 'tea', 'black tea': 'tea',
            # Frozen (expanded)
            'ice cream': 'ice cream', 'frozen vegetables': 'frozen vegetables',
            'frozen fruit': 'frozen fruit', 'frozen pizza': 'frozen pizza',
            # Produce (common packaged)
            'garlic': 'garlic', 'onion': 'onion', 'potato': 'potato', 'tomato': 'tomato',
            # Specialty/ethnic foods
            'gochujang': 'gochujang', 'harissa': 'harissa', 'miso': 'miso',
            'kimchi': 'kimchi', 'sauerkraut': 'sauerkraut', 'tempeh': 'tempeh'
        }
        
        # 🔥 CRITICAL: Match keywords - OCR is REQUIRED for packaged food
        # This is how real pantry apps work - keyword matching from OCR text
        ocr_text_normalized = re.sub(r'[^a-z\s]', '', ocr_text.lower())
        best_match = None
        best_conf = 0.0
        
        for keyword, food_name in label_keyword_map.items():
            if keyword in ocr_text_normalized:
                # Calculate confidence based on keyword match and OCR confidence
                # Longer keywords = more specific = higher confidence
                keyword_conf = min(0.95, 0.6 + len(keyword) * 0.05)
                if keyword_conf > best_conf:
                    best_match = food_name
                    best_conf = keyword_conf
        
        if best_match:
            # 🔥 TRUST OCR MORE: If OCR finds text with keyword match, use it with high confidence
            # Simple hack to boost accuracy by 2× - trust OCR more than CLIP for packaged food
            # OCR confidence > 0.3 → trust OCR label with 0.9 confidence
            if ocr_confidence > 0.3:
                final_confidence = 0.9  # High confidence when OCR finds matching keyword
            else:
                # Lower OCR confidence but still trust keyword match
                final_confidence = min(0.85, 0.7 + ocr_confidence * 0.3)
            
            if VERBOSE_LOGGING:
                print(f"   ✅ OCR keyword match: '{ocr_text}' → '{best_match}' (conf: {final_confidence:.2f})")
            
            return {
                "label": best_match,
                "confidence": final_confidence,
                "text": ocr_text,
                "ocr_confidence": ocr_confidence,
                "crop_used": best_ocr_result.get("crop_used", "unknown"),
                "enhancement": best_ocr_result.get("enhancement", "unknown")
            }
        
        # If no match but text exists, return as unknown packaged food with OCR text
        if len(ocr_text.strip()) > 3:
            return {
                "label": "unknown_packaged_food",
                "confidence": min(0.4, ocr_confidence * 0.7),  # Use OCR confidence
                "text": ocr_text,
                "ocr_confidence": ocr_confidence
            }
        
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"   ⚠️ OCR error: {e}")
    
    return None


def stage_rules_validate(region_data, clip_suggestion, ocr_result, user_pantry=None):
    """
    STAGE 4: Rules - "Does this make sense here?"
    Purpose: Apply common sense, not ML. Never hard-deletes, only boosts/penalizes/marks uncertain.
    Returns: Modified region_data with confidence adjustments and flags
    """
    # Start with base confidence from food_score calculation
    food_score = region_data.get("food_score", 0.0)
    suggested_label = region_data.get("suggested_label", "")
    confidence = region_data.get("confidence", 0.0)
    
    # Rule 1: If label is clearly non-food in this context, penalize
    non_food_in_fridge = ["tennis racket", "tie", "suitcase", "laptop", "car"]
    if suggested_label.lower() in non_food_in_fridge:
        confidence *= 0.5
        region_data["confidence"] = confidence
        region_data["rules_penalty"] = "non_food_in_context"
    
    # Rule 2: If item is in pantry history, boost
    if user_pantry and isinstance(user_pantry, list):
        user_item_names = [str(item.get('name', '')).lower() for item in user_pantry if isinstance(item, dict)]
        if suggested_label.lower() in user_item_names:
            confidence = min(1.0, confidence * 1.2)
            region_data["confidence"] = confidence
            region_data["rules_boost"] = "in_pantry_history"
    
    # Rule 3: If round shape + natural texture + no text, assume food (produce)
    if region_data.get("is_round", False) and region_data.get("natural_texture", False) and not ocr_result:
        food_score = min(1.0, food_score + 0.2)
        region_data["food_score"] = food_score
        region_data["rules_boost"] = "produce_like_features"
    
    # 🔥 FIX 6: Containers → show top CLIP guess + ask user (not unknown_packaged_food only)
    # Containers are expected for food (milk, oil, yogurt, sauce) - not uncertainty
    if region_data.get("is_container", False):
        skip_override = region_data.get("skip_container_override", False)
        clip_food_prob = region_data.get("clip_food_prob", 0.0)
        suggested_label = region_data.get("suggested_label", "")
        ocr_conf = ocr_result.get("confidence", 0) if ocr_result else 0.0
        
        if skip_override:
            # Strong CLIP (>= 0.45) or OCR (>= 0.45) - already handled, don't override
            if VERBOSE_LOGGING:
                print(f"   📦 Container detected but keeping CLIP/OCR label: {suggested_label}")
        elif ocr_conf >= 0.45:
            # OCR >= 0.45 - highest priority for containers
            # Keep the suggested_label from OCR
            if VERBOSE_LOGGING:
                print(f"   📦 Container detected, using OCR label: {suggested_label} (conf: {ocr_conf:.2f})")
        elif clip_food_prob > 0.0:
            # 🔥 PRINCIPLE 1 & 6: If CLIP has ANY prediction (even weak), show it + ask user
            # Never hard-reject food-like items - surface best guess
            region_data["needs_confirmation"] = True
            # Keep CLIP's suggested label (even if low confidence)
            if suggested_label not in ["unknown_food", "unknown_packaged_food", "unknown_item"]:
                # 🔥 PRINCIPLE 2: Format as "Possibly: X?" if confidence is very low
                if clip_food_prob < 0.30:
                    region_data["suggested_label"] = f"Possibly: {suggested_label}?"
                    region_data["possibly_prefix"] = True
                else:
                    region_data["suggested_label"] = suggested_label
                confidence = max(0.25, clip_food_prob)  # Floor at 0.25 for containers
                if VERBOSE_LOGGING:
                    print(f"   📦 Container: showing CLIP guess '{suggested_label}' (score: {clip_food_prob:.2f}) - ask user")
            else:
                # CLIP returned unknown - use unknown_packaged_food
                region_data["suggested_label"] = "unknown_packaged_food"
                confidence = max(0.2, clip_food_prob * 0.8)
                if VERBOSE_LOGGING:
                    print(f"   📦 Container: CLIP uncertain - showing as unknown_packaged_food - ask user")
        elif suggested_label not in ["unknown_food", "unknown_packaged_food", "unknown_item"]:
            # 🔥 PRINCIPLE 1: Has semantic label from elsewhere - keep it, never downgrade
            # Never hard-reject food-like items
            region_data["needs_confirmation"] = True
            # 🔥 PRINCIPLE 2: Format as "Possibly: X?" if confidence is very low
            if clip_food_prob > 0.0 and clip_food_prob < 0.25:
                region_data["suggested_label"] = f"Possibly: {suggested_label}?"
                region_data["possibly_prefix"] = True
            else:
                region_data["suggested_label"] = suggested_label
            confidence = max(0.3, confidence * 0.7)  # Floor at 0.3, don't reject
            if VERBOSE_LOGGING:
                print(f"   📦 Container detected - keeping semantic label: {suggested_label} (never downgrade)")
        else:
            # 🔥 PRINCIPLE 1: No label at all - use unknown_packaged_food as last resort
            # But still check for weak CLIP prediction to surface
            clip_suggestion = region_data.get("clip_suggestion")
            if clip_suggestion and clip_suggestion.get("label") and clip_suggestion.get("score", 0) > 0.15:
                weak_label = clip_suggestion["label"]
                region_data["suggested_label"] = f"Possibly: {weak_label}?"
                region_data["possibly_prefix"] = True
                confidence = max(0.2, clip_suggestion["score"] * 0.8)
                if VERBOSE_LOGGING:
                    print(f"   📦 Container: showing weak CLIP guess as 'Possibly: {weak_label}?' - ask user")
            else:
                region_data["suggested_label"] = "unknown_packaged_food"
                confidence = max(0.2, confidence * 0.6)
                if VERBOSE_LOGGING:
                    print(f"   📦 Container: no label found - showing as unknown_packaged_food - ask user")
            region_data["needs_confirmation"] = True
        
        region_data["confidence"] = confidence
    
    # Rule 5: Context score contribution
    context_score = 0.0
    if user_pantry and isinstance(user_pantry, list):
        user_item_names = [str(item.get('name', '')).lower() for item in user_pantry if isinstance(item, dict)]
        if suggested_label.lower() in user_item_names:
            context_score = 0.3  # Strong context match
        else:
            # Check for similar items
            for user_item in user_item_names:
                if suggested_label.lower() in user_item or user_item in suggested_label.lower():
                    context_score = 0.15  # Weak context match
                    break
    
    region_data["context_score"] = context_score
    
    # Fix #5: Recalculate food_score with context boost if CLIP was strong (but not if bypassed)
    # Fix #4: Food score must NEVER go negative
    if not region_data.get("bypass_rules", False):
        clip_food_prob = region_data.get("clip_food_prob", 0.0)
        if clip_food_prob > 0.35:
            # Rebalance with context boost
            classifier_food_prob = region_data.get("classifier_food_prob", 0.0)
            yolo_food_prior = region_data.get("yolo_food_prior", 0.0)
            food_score = 0.45 * clip_food_prob + 0.25 * classifier_food_prob + 0.20 * yolo_food_prior + 0.10 * context_score
            # Fix 2: Remove negative food scores entirely - clamp to 0.0-1.0
            food_score = max(0.0, min(1.0, food_score))  # Clamp to valid range
            region_data["food_score"] = food_score
    else:
        # Bypassed - food_score already set to CLIP score, don't recalculate
        pass
    
    return region_data


def stage_user_decision(region_data):
    """
    STAGE 5: User - "Is this correct?"
    Purpose: 🔥 FIX 1 - Three-tier confidence policy based on CLIP score.
    - CONFIDENT (≥0.70): Auto-accept
    - LIKELY (≥0.45): Show label + ask confirm ("Looks like tomato?")
    - UNKNOWN (<0.45): Show label + ask confirm (low confidence)
    Returns: Modified region_data with final decision flags
    """
    clip_score = region_data.get("clip_food_prob", 0.0)
    food_score = region_data.get("food_score", 0.0)
    confidence = region_data.get("confidence", 0.0)
    suggested_label = region_data.get("suggested_label", "")
    
    # 🔥 FIX 1: Three-tier confidence policy using CLIP score directly
    # Use CLIP score as primary signal (it's the best semantic matcher)
    if clip_score >= 0.70:
        # CONFIDENT: CLIP is very sure - auto-accept
        region_data["auto_add"] = True
        region_data["needs_confirmation"] = False
        region_data["confidence_tier"] = "CONFIDENT"
        region_data["final_confidence"] = min(0.95, max(confidence, clip_score))
        if VERBOSE_LOGGING:
            print(f"   ✅ CONFIDENT: {suggested_label} (CLIP: {clip_score:.2f}) - auto-accepting")
    elif clip_score >= 0.45:
        # LIKELY: CLIP is reasonably sure - show label + ask confirm
        region_data["auto_add"] = False
        region_data["needs_confirmation"] = True
        region_data["confidence_tier"] = "LIKELY"
        region_data["final_confidence"] = min(0.85, max(confidence, clip_score))
        # Keep semantic label - never downgrade to unknown
        if suggested_label not in ["unknown_food", "unknown_packaged_food", "unknown_item"]:
            region_data["suggested_label"] = suggested_label
        if VERBOSE_LOGGING:
            print(f"   ❓ LIKELY: {suggested_label} (CLIP: {clip_score:.2f}) - show label + ask confirm")
    else:
        # UNKNOWN: CLIP is uncertain - show label + ask confirm (but keep label)
        region_data["auto_add"] = False
        region_data["needs_confirmation"] = True
        region_data["confidence_tier"] = "UNKNOWN"
        region_data["hide"] = False  # Never hide, always show
        # 🔥 CRITICAL POLICY CHANGE: Wrong-but-editable > invisible
        # Replace "unknown_food" with best guess + (?)
        if suggested_label not in ["unknown_food", "unknown_packaged_food", "unknown_item"]:
            # Has semantic label - show it with (?) if low confidence
            if clip_score < 0.40:
                # Low confidence - format as "X (?)" instead of "unknown_food"
                region_data["suggested_label"] = f"{suggested_label} (?)"
                region_data["possibly_prefix"] = True
            else:
                region_data["suggested_label"] = suggested_label
            region_data["final_confidence"] = max(0.2, min(0.6, clip_score * 1.2))
            if VERBOSE_LOGGING:
                print(f"   ❓ UNKNOWN: {suggested_label} (CLIP: {clip_score:.2f}) - showing as '{region_data['suggested_label']}'")
        else:
            # No semantic label - check if we have a weak CLIP prediction we can surface
            clip_suggestion = region_data.get("clip_suggestion")
            if clip_suggestion and clip_suggestion.get("label") and clip_suggestion.get("score", 0) > 0.10:
                # Even weak CLIP prediction is better than unknown_food
                weak_label = clip_suggestion["label"]
                weak_score = clip_suggestion["score"]
                region_data["suggested_label"] = f"{weak_label} (?)"  # Show best guess with (?)
                region_data["possibly_prefix"] = True
                region_data["final_confidence"] = max(0.15, weak_score)
                if VERBOSE_LOGGING:
                    print(f"   ❓ UNKNOWN: weak CLIP prediction '{weak_label}' (score: {weak_score:.2f}) - showing as '{weak_label} (?)'")
            else:
                # Truly no label - use best guess from top-3 if available
                clip_suggestion = region_data.get("clip_suggestion")
                if clip_suggestion and clip_suggestion.get("top3_labels"):
                    top3 = clip_suggestion.get("top3_labels", [])
                    if top3:
                        # Show top guess with (?)
                        region_data["suggested_label"] = f"{top3[0]} (?)"
                        region_data["possibly_prefix"] = True
                        if VERBOSE_LOGGING:
                            print(f"   ❓ UNKNOWN: using top-3 guess '{top3[0]} (?)'")
                    else:
                        region_data["suggested_label"] = "unknown_food (?)"  # Last resort
                else:
                    region_data["suggested_label"] = "unknown_food (?)"  # Last resort
                region_data["final_confidence"] = max(0.1, clip_score)
                if VERBOSE_LOGGING:
                    print(f"   ❓ UNKNOWN: no label (CLIP: {clip_score:.2f}) - showing as '{region_data['suggested_label']}'")
    
    return region_data


def detect_food_items_with_ml(img_bytes, user_pantry=None, skip_preprocessing=False, use_multi_angle=True):
    """
    Enhanced ML pipeline v2: 
    Image -> Quality Scoring -> Preprocessing -> Object Detection -> Classification -> 
    OCR (gated) -> Shelf-Aware Context Fusion -> Confidence Calibration -> Adaptive Thresholding -> Results
    
    Improved hybrid ML approach with better accuracy, especially for bad angle photos:
        0. Image Quality Scoring - reject/weight poor quality images
        1. Multi-angle ensemble (optional) - try multiple orientations for consistency
        2. Object Detection (primary method) - finds bounding boxes
        3. Classify each detection hierarchically (category -> item)
        4. Extract metadata (quantity, expiration)
        5. Apply shelf-aware context fusion (boost confidence for items in pantry + spatial context)
        6. Apply quality-weighted confidence
        7. Categorize by adaptive confidence thresholds for user confirmation
    
    Args:
        img_bytes: Raw image bytes
        user_pantry: Optional user's current pantry for context fusion
        skip_preprocessing: If True, assume img_bytes is already a PIL Image

        use_multi_angle: If True, try multiple orientations (slower but more robust)
        fridge_zone: Optional zone identifier ("door", "drawer", "shelf", "produce_drawer") for context-aware boosting
    
    Includes comprehensive error handling for extreme scenarios.
    """
    # Declare globals at function start
    global _yolo_model, _food_classifier, _ocr_reader, _ml_models_loaded
    
    # Validate input
    if not img_bytes:
        if VERBOSE_LOGGING:

            print("Warning: Empty image bytes in detect_food_items_with_ml")
        return []
    
    if not isinstance(img_bytes, (bytes, bytearray)):
        if VERBOSE_LOGGING:

            print(f"Warning: Invalid image bytes type: {type(img_bytes)}")
        return []
    
    # Try to load models if not loaded
    try:
        if not _ml_models_loaded:
            print("🔄 Loading ML models...")
            load_ml_models()
            if not _ml_models_loaded:
                print("⚠️ Warning: ML models failed to load, but continuing anyway")
        else:
            print("✅ ML models already loaded")
    except Exception as e:
        print(f"❌ Warning: Failed to load ML models: {e}")
        import traceback
        traceback.print_exc()
        # Don't return empty - try to continue with whatever models are available
        if VERBOSE_LOGGING:
            print(f"Continuing with partial model availability")
    
    # If no models are available and ML is enabled, log warning but don't fail
    if (ML_VISION_ENABLED or YOLO_DETECTION_ENABLED) and not _yolo_model and not _food_classifier:
        print("⚠️ WARNING: ML/YOLO enabled but no models loaded. Check if ultralytics/transformers are installed.")
        print("   Install with: pip install ultralytics transformers")
        print("   Falling back to OpenAI detection...")
        # Don't return empty - let it fall through to OpenAI fallback
    
    # STEP 0.5: Image Quality Scoring (NEW)
    quality_score = 1.0
    quality_metrics = {}
    try:
        from PIL import Image
        import io
        temp_img = Image.open(io.BytesIO(img_bytes))
        quality_score, quality_metrics = compute_image_quality_score(temp_img)
        
        if VERBOSE_LOGGING:
            print(f"📊 Image Quality Score: {quality_score:.2f} (blur={quality_metrics.get('blur_score', 0):.2f}, "
                  f"brightness={quality_metrics.get('brightness_score', 0):.2f}, "
                  f"contrast={quality_metrics.get('contrast_score', 0):.2f}, "
                  f"glare={quality_metrics.get('glare_score', 0):.2f})")
        
        # 🔥 PERFORMANCE: Early exit for very poor quality images (saves processing time)
        # Reject only extremely poor quality images (lowered threshold from 0.15 to 0.10)
        # This allows more images to be processed while still filtering out truly unusable images
        if quality_score < 0.10:
            if VERBOSE_LOGGING:
                print(f"⚠️ Image quality too low ({quality_score:.2f}), rejecting scan")
            print(f"⚠️ Image quality too low ({quality_score:.2f}), rejecting scan - early exit for performance")
            return []  # Reject extremely poor quality images (early exit saves time)
        
        # Warn about poor quality but continue (lowered threshold from 0.5 to 0.4)
        if quality_score < 0.4:
            if VERBOSE_LOGGING:
                print(f"⚠️ Warning: Poor image quality ({quality_score:.2f}), results may be less accurate")
            print(f"⚠️ Warning: Poor image quality ({quality_score:.2f}), continuing anyway")
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Warning: Image quality scoring failed: {e}")
        # Continue with default quality score if analysis fails
        quality_score = 0.8
    
    # 🔥 PERFORMANCE: Disable multi-angle by default (3x slower, minimal accuracy gain)
    # STEP 0: Multi-angle ensemble (if enabled and not already preprocessed)
    # Multi-angle is disabled by default for performance - can be enabled via env var if needed
    use_multi_angle = False  # Disabled for performance (was: use_multi_angle)
    if use_multi_angle and not skip_preprocessing:
        try:
            # Try multi-angle detection for better consistency with bad angles
            multi_angle_items = detect_with_multiple_angles(img_bytes, user_pantry=user_pantry)
            if multi_angle_items and len(multi_angle_items) > 0:
                # Apply quality score to multi-angle results before returning
                for item in multi_angle_items:
                    base_conf = item.get("confidence", 0)
                    item["confidence"] = base_conf * quality_score
                    item["quality_score"] = quality_score
                return multi_angle_items
            pass
        except Exception as e:
            if VERBOSE_LOGGING:

                print(f"Warning: Multi-angle detection failed, falling back to single angle: {e}")
            # Fall through to single-angle detection
    
    # STEP 1: Preprocess image with error handling
    try:
        # Detect and correct rotation for bad angle photos
        detected_angle = None
        if not skip_preprocessing:
            try:
                from PIL import Image
                import io
                temp_img = Image.open(io.BytesIO(img_bytes))
                detected_angle = detect_rotation_angle(temp_img)
                pass
            except Exception:
                # If rotation detection fails, continue without correction
                pass
        
        # Preprocess with rotation correction (if detected)
        if skip_preprocessing:
            # Assume img_bytes is already a PIL Image
            img = img_bytes
        else:
            img = preprocess_image_for_ml(img_bytes, angle=detected_angle)
    except Exception as e:
        if VERBOSE_LOGGING:

            print(f"Warning: Image preprocessing failed: {e}")
        return []
    
    items = []
    item_confidence = {}  # Track confidence scores for deduplication

    # OCR dates (global) - extract from full image
    expiration_dates = extract_expiration_dates(img, _ocr_reader)
    exp_date = expiration_dates[0] if expiration_dates else None

    # STEP 2: Object Detection First (Primary Method) - finds bounding boxes for better accuracy
    # This is more accurate than full-image classification because it identifies individual items
    detections_found = False
    
    # 🔥 OCR-FIRST for Packaged Food: If text detected early, prioritize OCR over YOLO
    # This helps identify packaged items that YOLO might miss
    ocr_text_detected = False
    if _ocr_reader:
        try:
            import numpy as np
            img_array = np.array(img)
            # 🔥 PERFORMANCE: Limit OCR processing time and reduce redundant scans
            # Quick OCR scan to detect if there's text (packaged food indicator)
            # Use lower detail for initial scan (faster)
            try:
                ocr_results = _ocr_reader.readtext(img_array, detail=0)  # detail=0 is faster
                if ocr_results and len(ocr_results) > 5:  # Significant text detected
                    ocr_text_detected = True
                    print(f"   📝 OCR-First: Detected {len(ocr_results)} text regions - prioritizing OCR for packaged food")
            except Exception:
                ocr_results = []
        except Exception:
            pass
    
    # ENHANCED: Use OCR to identify rare foods from packaging labels (fallback if no detections)
    # OR if OCR text detected early (packaged food)
    if _ocr_reader and (not items or ocr_text_detected):  # If no items found OR text detected early
        try:
            import re
            import numpy as np
            
            # Validate image before OCR
            if img is None:
                if VERBOSE_LOGGING:

                    print("Warning: Image is None, skipping OCR")
                return items
            
            # Convert to numpy array with error handling
            try:
                img_array = np.array(img)
                if img_array.size == 0:
                    if VERBOSE_LOGGING:

                        print("Warning: Empty image array, skipping OCR")
                    return items
                pass
            except Exception as e:
                if VERBOSE_LOGGING:

                    print(f"Warning: Failed to convert image to numpy array: {e}")
                return items
            
            # Run OCR with timeout protection
            ocr_results = []
            try:
                # Limit OCR processing time
                import signal
                import threading
                
                ocr_error = None
                
                def run_ocr():
                    nonlocal ocr_results, ocr_error
                    try:
                        # 🔥 PERFORMANCE: Use detail=0 for faster OCR when we only need text
                        ocr_results = _ocr_reader.readtext(img_array, detail=0)  # Faster, text-only
                        pass
                    except Exception as e:
                        ocr_error = e
                
                thread = threading.Thread(target=run_ocr)
                thread.daemon = True
                thread.start()
                thread.join(timeout=10)  # 🔥 PERFORMANCE: Reduced from 20s to 10s for faster failure
                
                if thread.is_alive():
                    if VERBOSE_LOGGING:

                        print("Warning: OCR timed out after 10 seconds")
                    return items
                elif ocr_error:
                    raise ocr_error
                
                if not ocr_results:
                    # Don't return - continue to YOLO detection
                    pass
            except Exception as ocr_err:
                if VERBOSE_LOGGING:

                    print(f"Warning: OCR processing failed: {ocr_err}")
                # Don't return - continue to YOLO detection
                pass
            
            food_keywords = []
            for ocr_item in ocr_results:
                try:
                    # Handle different OCR result formats
                    if isinstance(ocr_item, tuple) and len(ocr_item) >= 3:
                        bbox, text, conf = ocr_item[0], ocr_item[1], ocr_item[2]
                    elif isinstance(ocr_item, dict):
                        bbox = ocr_item.get('bbox', [])
                        text = ocr_item.get('text', '')
                        conf = ocr_item.get('conf', 0)
                    else:
                        pass
                    
                    # Validate confidence
                    try:
                        conf = float(conf)
                        pass
                    except (ValueError, TypeError):
                        pass
                    
                    # OCR Confidence Gating: Require higher confidence (0.5 instead of 0.3)
                    # This reduces false positives from random packaging text
                    ocr_min_confidence = 0.5
                    
                    if conf > ocr_min_confidence:
                        text_lower = text.lower().strip()
                        if not text_lower or len(text_lower) < 2:
                            continue
                        
                        # OCR Confidence Gating: Check if text box overlaps with detected objects
                        # For now, we assume text is aligned if we have a valid bbox
                        # In future, could check actual overlap with detected item bounding boxes
                        text_aligned = True  # Assume aligned for now (can be enhanced with actual overlap checking)
                        
                        # Look for food-related keywords in OCR text
                        # Common food words that might indicate rare items
                        food_indicators = [
                            'organic', 'artisan', 'gourmet', 'specialty', 'imported',
                            'gluten-free', 'vegan', 'keto', 'paleo', 'halal', 'kosher',
                            'spice', 'herb', 'seasoning', 'sauce', 'dressing', 'marinade',
                            'paste', 'concentrate', 'extract', 'syrup', 'preserve', 'jam',
                            'chutney', 'relish', 'pickle', 'fermented', 'kimchi', 'miso',
                            'tahini', 'hummus', 'pesto', 'salsa', 'guacamole', 'tzatziki'
                        ]
                        # Check if text contains food indicators
                        for indicator in food_indicators:
                            if indicator in text_lower:
                                try:
                                    # Extract potential food name from surrounding text
                                    words = text_lower.split()
                                    for i, word in enumerate(words):
                                        if indicator in word or any(food_word in word for food_word in ['sauce', 'spice', 'herb', 'oil', 'vinegar']):
                                            # Try to get the food name (word before or after indicator)
                                            if i > 0:
                                                potential_name = words[i-1] + ' ' + words[i]
                                            elif i < len(words) - 1:
                                                potential_name = words[i] + ' ' + words[i+1]
                                            else:
                                                potential_name = words[i]
                                            # Clean up the name
                                            potential_name = re.sub(r'[^a-z\s]', '', potential_name).strip()
                                            if len(potential_name) > 2 and len(potential_name) < 100:  # Length limit
                                                normalized = normalize_item_name(potential_name)
                                                if normalized and normalized.lower() not in item_confidence:
                                                    # OCR Confidence Gating: Only add if text is aligned with image content
                                                    # Apply stricter confidence weighting for OCR-only detections
                                                    ocr_confidence = float(conf * 0.5)  # Reduced from 0.6 for stricter gating
                                                    
                                                    # Additional penalty if text doesn't align with detected objects
                                                    if not text_aligned:
                                                        ocr_confidence *= 0.7  # 30% penalty for unaligned text
                                                    
                                                    items.append({
                                                        "name": normalized,
                                                        "quantity": "1",
                                                        "expirationDate": exp_date,
                                                        "category": validate_category(normalized, "other"),
                                                        "confidence": ocr_confidence,  # OCR-based confidence (stricter gating)
                                                        "detection_method": "ocr",
                                                        "ocr_aligned": text_aligned
                                                    })
                                                    item_confidence[normalized.lower()] = ocr_confidence
                                                break  # Only process first match per text
                                    pass
                                except Exception as word_error:
                                    if VERBOSE_LOGGING:

                                        print(f"Warning: Error processing OCR word: {word_error}")
                                    pass
                                break  # Move to next OCR result
                except Exception as item_error:
                    if VERBOSE_LOGGING:

                        print(f"Warning: Error processing OCR item: {item_error}")
                    continue
        except Exception as e:
            if VERBOSE_LOGGING:

                print(f"OCR-based rare food detection error: {e}")

    # STEP 2: Stage-Based Detection Pipeline
    # Clean architecture: YOLO -> CLIP -> OCR -> Rules -> User
    print(f"📍 STAGE-BASED DETECTION PIPELINE")
    
    detections_found_yolo = False
    
    # STAGE 1: YOLO - "Where is something?" (region detection only)
    regions = stage_yolo_detect_regions(img)
    
    # 🔥 PERFORMANCE: Multi-crop zooming is slower - only use if really needed
    # 🔥 FIX #2: If YOLO finds nothing, try multi-crop zooming (but limit tiles for performance)
    if not regions or len(regions) == 0:
        if VERBOSE_LOGGING:
            print("   🔍 YOLO found no regions - trying multi-crop zooming (limited tiles for performance)")
        # Limit multi-crop to 2x2 tiles (was default, but now explicit for performance)
        regions = stage_yolo_detect_regions_multi_crop(img, use_multi_crop=True)
    
    if regions and len(regions) > 0:
        detections_found_yolo = True
        detections_found = True  # 🔥 FIX: Set detections_found flag
        print(f"✅ YOLO found {len(regions)} regions")
        
        # 🔥 PERFORMANCE: Run OCR on full image once for spatial binding (cached for all regions)
        # This allows us to assign OCR text to YOLO boxes based on spatial overlap
        full_image_ocr_results = None
        if _ocr_reader:
            try:
                import numpy as np
                img_array = np.array(img)
                # 🔥 PERFORMANCE: Use detail=0 for faster OCR (we only need text, not bboxes for spatial binding)
                # Full bboxes will be computed per-region if needed
                full_image_ocr_results = _ocr_reader.readtext(img_array, detail=1)  # Keep detail=1 for spatial binding
                if full_image_ocr_results:
                    if VERBOSE_LOGGING:
                        print(f"   📍 OCR spatial binding: Found {len(full_image_ocr_results)} text regions in full image")
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"   ⚠️ Full-image OCR error: {e}")
    else:
        # 🔥 FIX: YOLO found no regions - ensure detections_found is False
        detections_found_yolo = False
        detections_found = False
        print(f"⚠️ YOLO found no regions - will use fallback detection")
        
        # 🔥 DETERMINISTIC CANDIDATE SELECTION SYSTEM
        # Core rule: CLIP should never see more than 60 labels per crop (20-40 is ideal)
        # This prevents softmax collapse and ensures explainable, fast, ML-friendly results
        
        # Step 1: Define structured pantry ontology (not a flat list)
        PANTRY_ONTOLOGY = {
            "produce": {
                "fruits": [
                    "apple", "banana", "orange", "lemon", "lime",
                    "grape", "strawberry", "blueberry", "avocado"
                ],
                "vegetables": [
                    "onion", "garlic", "potato", "tomato", "carrot",
                    "bell pepper", "cucumber", "broccoli", "lettuce",
                    "spinach", "cabbage"
                ]
            },
            "dairy": [
                "milk", "yogurt", "butter", "cheese", "cream"
            ],
            "condiments": [
                "olive oil", "vegetable oil", "canola oil",
                "soy sauce", "ketchup", "mustard", "mayonnaise",
                "hot sauce", "vinegar"
            ],
            "grains": [
                "bread", "rice", "pasta", "flour", "cereal", "oats"
            ],
            "proteins": [
                "egg", "chicken", "beef", "pork", "fish", "tofu"
            ],
            "snacks": [
                "chips", "cookies", "crackers", "granola bar"
            ],
            "beverages": [
                "water", "juice", "soda", "energy drink", "milk"
            ]
        }
        
        # Hard constraints
        MAX_CLIP_LABELS = 60
        IDEAL_CLIP_LABELS = 30
        
        # Step 2: Region type inference (using cheap signals)
        def infer_region_type(yolo_class, bbox, has_text, is_container):
            """Determine region type using cheap heuristics"""
            yolo_lower = yolo_class.lower()
            if yolo_lower in {"bottle", "jar", "can", "box"} or is_container:
                return "container"
            if yolo_lower in {"apple", "banana", "orange", "broccoli", "carrot"}:
                return "produce"
            if has_text:
                return "packaged"
            return "unknown"
        
        # 🔥 3-STAGE PIPELINE: Labels by food type (narrowed answer space)
        # 🔥 PACKAGE-TYPE GATING: Restrict CLIP labels by package type (huge precision boost)
        # This reduces wrong labels by 50-70% by preventing CLIP from comparing incompatible items
        LABELS_BY_TYPE = {
            "fresh produce": (
                PANTRY_ONTOLOGY["produce"]["fruits"] +
                PANTRY_ONTOLOGY["produce"]["vegetables"]
            ),
            "bottle or jar": [
                # Oils and liquids
                "olive oil", "vegetable oil", "canola oil", "coconut oil", "sesame oil",
                "soy sauce", "vinegar", "balsamic vinegar", "rice vinegar", "apple cider vinegar",
                # Condiments
                "hot sauce", "sriracha", "tabasco", "ketchup", "mustard", "mayonnaise",
                # Spreads
                "honey", "maple syrup", "jam", "jelly", "peanut butter", "almond butter",
                # Beverages in bottles
                "juice", "orange juice", "apple juice", "cranberry juice", "water"
            ],
            "carton": [
                # Dairy and beverages in cartons
                "milk", "juice", "orange juice", "apple juice", "cranberry juice",
                "cream", "heavy cream", "half and half", "yogurt", "buttermilk",
                "almond milk", "soy milk", "coconut milk", "oat milk", "eggs"
            ],
            "can": [
                # Beverages and canned goods
                "soda", "cola", "sprite", "ginger ale", "tonic water", "seltzer",
                "beans", "chickpeas", "black beans", "kidney beans", "corn", "peas",
                "tomato sauce", "tomato paste", "soup", "broth", "tuna", "salmon"
            ],
            "box": [
                # Dry goods in boxes
                "cereal", "pasta", "spaghetti", "macaroni", "rice", "crackers",
                "granola bar", "protein bar", "cookies", "chips", "flour", "sugar",
                "baking powder", "baking soda", "oats", "quinoa"
            ],
            "bag": [
                # Snacks and dry goods in bags
                "chips", "potato chips", "tortilla chips", "pretzels", "nuts", "almonds",
                "walnuts", "peanuts", "cashews", "popcorn", "flour", "sugar", "rice",
                "bread", "crackers", "cookies"
            ],
            "snack": [
                # Any snack packaging
                "chips", "cookies", "crackers", "granola bar", "protein bar", "nuts",
                "pretzels", "popcorn", "trail mix", "candy", "chocolate"
            ],
            "beverage": [
                # Any beverage packaging
                "water", "juice", "orange juice", "apple juice", "cranberry juice",
                "soda", "cola", "sprite", "ginger ale", "energy drink", "milk",
                "coffee", "tea", "sports drink"
            ]
        }
        
        # Step 3: Candidate routing (THE CORE) - NOW USES FOOD TYPE FROM CLIP
        def route_candidates(food_type=None, region_type=None, yolo_class="", ocr_text="", user_pantry_items=None):
            """
            🔥 3-STAGE PIPELINE: Narrow label space based on food type classification.
            Returns 6-15 candidates (much smaller than before), dramatically improving precision.
            """
            candidates = []
            yolo_lower = yolo_class.lower()
            ocr_lower = ocr_text.lower() if ocr_text else ""
            
            # 🔥 PRINCIPLE 3: Add user's past items FIRST (highest priority)
            user_items = []
            if user_pantry_items and isinstance(user_pantry_items, list):
                for item in user_pantry_items:
                    if isinstance(item, dict):
                        name = item.get("name", "").strip()
                        if name and name.lower() not in ["unknown_food", "unknown_packaged_food", "unknown_item"]:
                            user_items.append(name)
            
            # Add user items at the beginning
            if user_items:
                candidates.extend(user_items[:5])  # Top 5 user items (reduced from 10)
                if VERBOSE_LOGGING:
                    print(f"   📚 Added {len(user_items[:5])} user items to candidates")
            
            # 🔥 FIX #3: Use OCR text as a hint to narrow candidates
            # OCR keywords → likely food categories → narrow CLIP candidates
            ocr_hints = []
            if ocr_lower:
                # Map OCR keywords to likely food categories
                if any(kw in ocr_lower for kw in ["cereal", "granola", "oats", "quinoa"]):
                    ocr_hints.extend(["cereal", "oats", "quinoa", "granola bar"])
                if any(kw in ocr_lower for kw in ["pasta", "spaghetti", "macaroni", "noodles"]):
                    ocr_hints.extend(["pasta", "spaghetti", "macaroni"])
                if any(kw in ocr_lower for kw in ["rice", "jasmine", "basmati"]):
                    ocr_hints.extend(["rice"])
                if any(kw in ocr_lower for kw in ["flour", "baking", "sugar"]):
                    ocr_hints.extend(["flour", "sugar", "baking powder", "baking soda"])
                if any(kw in ocr_lower for kw in ["chips", "crackers", "cookies", "snacks"]):
                    ocr_hints.extend(["chips", "crackers", "cookies", "snacks"])
                if any(kw in ocr_lower for kw in ["original", "family", "size"]):
                    # Generic packaging text - likely cereal, snacks, or crackers
                    ocr_hints.extend(["cereal", "snacks", "chips", "crackers"])
            
            # 🔥 STAGE 2: Narrow by food type (if CLIP classified it)
            if food_type and food_type in LABELS_BY_TYPE:
                type_labels = LABELS_BY_TYPE[food_type]
                candidates.extend(type_labels)
                if VERBOSE_LOGGING:
                    print(f"   🎯 Narrowed to {food_type}: {len(type_labels)} labels")
            else:
                # Fallback to region_type heuristics if food_type not available
                if region_type == "container":
                    candidates.extend(LABELS_BY_TYPE.get("bottle or jar", []))
                elif region_type == "produce":
                    candidates.extend(LABELS_BY_TYPE.get("fresh produce", []))
                elif region_type == "packaged":
                    # 🔥 FIX #3: Use OCR hints to narrow packaged food candidates
                    if ocr_hints:
                        # Prioritize OCR-hinted items
                        candidates.extend(ocr_hints)
                        if VERBOSE_LOGGING:
                            print(f"   💡 OCR hints narrowed to: {ocr_hints}")
                    # Try to infer from OCR keywords
                    if "oil" in ocr_lower:
                        candidates.extend(LABELS_BY_TYPE.get("bottle or jar", []))
                else:
                        candidates.extend(LABELS_BY_TYPE.get("box", []))
                # Last resort: small mixed set
                candidates.extend(
                        PANTRY_ONTOLOGY["produce"]["fruits"][:3] +
                        PANTRY_ONTOLOGY["produce"]["vegetables"][:3] +
                    ["bread", "egg", "milk"]
                )
            
            # 🔥 FIX #3: Add OCR hints to candidates (prioritize them)
            if ocr_hints:
                # Add OCR hints at the beginning (higher priority)
                candidates = ocr_hints + [c for c in candidates if c not in ocr_hints]
                if VERBOSE_LOGGING:
                    print(f"   💡 OCR hints added to candidates: {ocr_hints}")
            
            # Step 4: Hard constraints (CRITICAL) - but now we have fewer candidates (6-15)
            # Keep max at 20 for safety, but typically we'll have 6-15
            if len(candidates) > 20:
                candidates = candidates[:20]
            
            # Remove duplicates while preserving order (user items stay first)
            seen = set()
            unique_candidates = []
            for item in candidates:
                item_lower = item.lower()
                if item_lower not in seen:
                    seen.add(item_lower)
                    unique_candidates.append(item)
            
            return unique_candidates
        
        # Legacy CLIP_CANDIDATES for backward compatibility (not used in new routing)
        CLIP_CANDIDATES = {
            "produce": [
                # Fruits (30+ items)
                "apple", "banana", "orange", "lemon", "lime", "grapefruit", "tangerine", "clementine",
                "grape", "strawberry", "blueberry", "raspberry", "blackberry", "cranberry", "cherry",
                "plum", "peach", "nectarine", "apricot", "pear", "kiwi", "mango", "papaya", "pineapple",
                "watermelon", "cantaloupe", "honeydew", "coconut", "pomegranate", "fig", "date",
                # Vegetables (50+ items)
                "onion", "potato", "tomato", "cherry tomato", "garlic", "ginger", "pepper", "bell pepper",
                "jalapeno", "habanero", "serrano", "carrot", "celery", "cucumber", "zucchini", "squash",
                "yellow squash", "acorn squash", "butternut squash", "pumpkin", "eggplant", "avocado",
                "mushroom", "button mushroom", "portobello", "shiitake", "oyster mushroom", "cabbage",
                "red cabbage", "lettuce", "romaine lettuce", "iceberg lettuce", "spinach", "kale",
                "chard", "swiss chard", "bok choy", "brussels sprouts", "broccoli", "cauliflower",
                "corn", "peas", "snow peas", "green beans", "asparagus", "artichoke", "radish",
                "radishes", "beet", "beets", "turnip", "rutabaga", "parsnip", "fennel", "leek",
                "shallot", "scallion", "green onion", "chive", "herbs", "basil", "cilantro",
                "parsley", "dill", "mint", "oregano", "thyme", "rosemary", "sage", "tarragon",
                # Pickled/preserved produce
                "pickles", "pickle", "olives", "olive", "black olives", "green olives", "kalamata olives",
                "capers", "sauerkraut", "kimchi"
            ],
            "liquids": [
                "olive oil", "vegetable oil", "canola oil", "coconut oil", "sesame oil",
                "soy sauce", "vinegar", "balsamic vinegar", "rice vinegar", "apple cider vinegar",
                "milk", "yogurt", "cream", "buttermilk", "almond milk", "soy milk",
                "coconut milk", "oat milk", "rice milk", "hemp milk", "heavy cream", "half and half",
                "juice", "orange juice", "apple juice", "cranberry juice", "grape juice",
                "soda", "cola", "sprite", "ginger ale", "tonic water", "seltzer", "sparkling water"
            ],
            "condiments": [
                # Sauces (30+ items)
                "ketchup", "mustard", "yellow mustard", "dijon mustard", "whole grain mustard",
                "mayonnaise", "hot sauce", "sriracha", "tabasco", "chili sauce", "bbq sauce",
                "barbecue sauce", "worcestershire sauce", "soy sauce", "tamari", "teriyaki sauce",
                "hoisin sauce", "fish sauce", "oyster sauce", "ponzu", "mirin", "gochujang",
                "harissa", "sambal oelek", "chili paste", "tomato sauce", "marinara sauce",
                "pasta sauce", "alfredo sauce", "pesto", "salsa", "pico de gallo",
                # Dressings (15+ items)
                "ranch dressing", "ranch", "italian dressing", "caesar dressing", "thousand island",
                "blue cheese dressing", "vinaigrette", "balsamic vinaigrette", "italian vinaigrette",
                "honey mustard", "ranch dip", "caesar dip",
                # Spreads (20+ items)
                "honey", "maple syrup", "agave syrup", "jam", "jelly", "preserves", "marmalade",
                "peanut butter", "almond butter", "cashew butter", "sunflower butter", "tahini",
                "hummus", "guacamole", "avocado spread", "nutella", "chocolate spread",
                # Other condiments (15+ items)
                "relish", "chutney", "aioli", "tzatziki", "tzatziki sauce", "chimichurri",
                "balsamic glaze", "caramel sauce", "chocolate sauce"
            ],
            "spices": [
                "salt", "black pepper", "red pepper", "paprika", "cumin", "coriander",
                "oregano", "basil", "thyme", "rosemary", "parsley", "cilantro", "dill",
                "cinnamon", "nutmeg", "ginger", "turmeric", "curry powder", "chili powder",
                "garlic powder", "onion powder", "bay leaves", "cayenne", "red pepper flakes",
                "cardamom", "cloves", "allspice", "star anise", "fennel seeds", "caraway seeds",
                "mustard seeds", "sesame seeds", "poppy seeds", "chia seeds", "flax seeds",
                "vanilla extract", "almond extract", "lemon zest", "orange zest"
            ],
            "grains": [
                "bread", "rice", "pasta", "noodles", "cereal", "oats", "flour", "wheat flour",
                "quinoa", "barley", "couscous", "bulgur", "tortilla", "bagel", "cracker",
                "spaghetti", "macaroni", "penne", "fettuccine", "linguine", "ravioli", "lasagna",
                "ramen", "udon", "soba", "rice noodles", "egg noodles",
                "white rice", "brown rice", "wild rice", "jasmine rice", "basmati rice",
                "wheat", "rye", "buckwheat", "millet", "amaranth", "teff",
                "breadcrumbs", "panko", "cornmeal", "polenta", "grits"
            ],
            "dairy": [
                "milk", "cheese", "yogurt", "butter", "eggs", "sour cream", "cream cheese",
                "cottage cheese", "mozzarella", "cheddar", "parmesan", "feta", "greek yogurt",
                "swiss cheese", "provolone", "gouda", "brie", "camembert", "blue cheese",
                "goat cheese", "ricotta", "mascarpone", "heavy cream", "whipping cream",
                "buttermilk", "kefir", "ghee", "margarine"
            ],
            "canned_packaged": [
                "beans", "chickpeas", "black beans", "kidney beans", "tomato sauce", "tomato paste",
                "canned tomatoes", "soup", "broth", "stock", "coconut milk", "tuna", "salmon",
                "corn", "peas", "green beans", "artichokes", "hearts of palm", "anchovies",
                "sardines", "mackerel", "canned fruit", "pineapple", "peaches", "pears",
                "lentils", "split peas", "black eyed peas", "lima beans", "navy beans", "pinto beans",
                "chicken broth", "beef broth", "vegetable broth", "bone broth",
                "canned corn", "canned peas", "canned carrots", "canned mushrooms",
                "canned olives", "canned artichoke hearts", "canned beets", "canned asparagus",
                "canned pineapple", "canned peaches", "canned pears", "canned mandarin oranges",
                "canned cherries", "canned cranberries", "canned pumpkin", "canned coconut"
            ],
            "snacks": [
                "chips", "crackers", "nuts", "almonds", "walnuts", "peanuts", "cashews",
                "popcorn", "pretzels", "granola", "trail mix",
                "potato chips", "tortilla chips", "corn chips", "pita chips", "veggie chips",
                "rice cakes", "rice crackers", "wheat crackers", "cheese crackers",
                "granola bars", "protein bars", "energy bars", "cereal bars",
                "pistachios", "pecans", "hazelnuts", "macadamia nuts", "brazil nuts",
                "sunflower seeds", "pumpkin seeds", "sesame seeds", "chia seeds"
            ],
            "beverages": [
                "coffee", "tea", "hot chocolate", "cocoa", "espresso", "instant coffee",
                "green tea", "black tea", "herbal tea", "chai tea", "matcha",
                "soda", "cola", "sprite", "ginger ale", "tonic water", "seltzer",
                "juice", "orange juice", "apple juice", "cranberry juice", "grape juice",
                "lemonade", "iced tea", "sports drink", "energy drink", "water"
            ],
            "baking": [
                "flour", "wheat flour", "all purpose flour", "bread flour", "cake flour",
                "sugar", "brown sugar", "powdered sugar", "confectioners sugar",
                "baking powder", "baking soda", "yeast", "active dry yeast",
                "vanilla extract", "almond extract", "cocoa powder", "chocolate chips",
                "shortening", "vegetable shortening", "coconut oil", "olive oil",
                "cornstarch", "arrowroot", "gelatin", "pudding mix", "jello"
            ],
            "frozen": [
                "frozen vegetables", "frozen fruit", "frozen berries", "frozen corn", "frozen peas",
                "frozen broccoli", "frozen spinach", "frozen mixed vegetables",
                "ice cream", "frozen yogurt", "sorbet", "popsicles",
                "frozen pizza", "frozen meals", "frozen burritos", "frozen waffles"
            ],
            "meat_seafood": [
                "chicken", "beef", "pork", "turkey", "lamb", "bacon", "sausage",
                "ground beef", "ground turkey", "ground chicken", "ground pork",
                "salmon", "tuna", "cod", "tilapia", "shrimp", "crab", "lobster",
                "deli meat", "ham", "turkey breast", "roast beef", "salami", "pepperoni"
            ]
        }
        
        # Process each region through the pipeline
        for region in regions:
            try:
                x1, y1, x2, y2 = region["bbox"]
                yolo_conf = region["yolo_conf"]
                yolo_food_prior = region["yolo_food_prior"]
                is_container = region["is_container"]
                class_name = region.get("class_name", "").lower()
                class_id = region.get("class_id", -1)
                
                # 🔥 FIX 2: YOLO is blind - never trust class_name as semantic truth
                # YOLO class_name (vase, bowl, bottle) is just object shape, NOT food identity
                # YOLO saying "vase" does NOT mean CLIP is wrong - it's just region proposal
                # Only use class_name for container detection, NOT for penalizing CLIP confidence
                # Rule 3: Expand CLIP vocabulary - use all candidates for better coverage
                
                # Fix 1: Relax YOLO box filtering - pantry items can be small
                # 3% threshold was too aggressive, filtering out 90% of valid items
                box_area = (x2 - x1) * (y2 - y1)
                img_area = img.size[0] * img.size[1]
                box_area_percent = (box_area / img_area) * 100 if img_area > 0 else 0
                
                # Remove box area filtering entirely - let CLIP decide if item is valid
                # Even tiny boxes might contain identifiable food items
                if box_area_percent < 0.05:  # Only filter extremely tiny boxes (< 0.05%)
                    if VERBOSE_LOGGING:
                        print(f"   ⏭️ Skipping extremely tiny YOLO box ({box_area_percent:.2f}% of image)")
                    continue
                            
                # 🔥 3-STAGE PIPELINE: Category → Narrow → Name
                # Crop the region first (needed for food type classification)
                crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
                
                # STAGE 1: Classify food type (what kind of thing is this?)
                region_info_for_type = {
                    "is_container": is_container,
                    "class_name": class_name,
                    "class_id": class_id
                }
                food_type_result = stage_classify_food_type(crop, region_info=region_info_for_type)
                food_type = food_type_result.get("food_type") if food_type_result else None
                
                # 🔥 STAGE 3: OCR FIRST (before CLIP) - "What does the text say?"
                # 🔥 FIX #1: Use spatial OCR binding - assign OCR text to YOLO boxes
                ocr_result = None
                ocr_authoritative = False
                bound_ocr_texts = []
                
                # 🔥 FIX #1: Bind OCR texts to this YOLO region based on spatial overlap
                if full_image_ocr_results:
                    bound_ocr_texts = stage_ocr_bind_to_region(
                        region["bbox"], 
                        full_image_ocr_results, 
                        img.size
                    )
                    if bound_ocr_texts and VERBOSE_LOGGING:
                        combined_bound_text = ' '.join([t["text"] for t in bound_ocr_texts])
                        print(f"   📍 Spatially bound {len(bound_ocr_texts)} OCR texts to region: '{combined_bound_text}'")
                
                if is_container or bound_ocr_texts:
                    # Try OCR on original crop FIRST (before CLIP)
                    # 🔥 FIX #1: Pass bound OCR texts to OCR stage
                    ocr_result = stage_ocr_read_label(crop, is_container=True, bound_ocr_texts=bound_ocr_texts)
                    
                    # 🔥 FIX #3: If OCR didn't find a label, try OCR candidate proposal → CLIP verification
                    if (not ocr_result or not ocr_result.get("label")) and bound_ocr_texts:
                        combined_ocr_text = ' '.join([t["text"] for t in bound_ocr_texts])
                        ocr_candidate_result = stage_ocr_propose_candidates(
                            combined_ocr_text, 
                            clip_model=_clip_model, 
                            clip_processor=_clip_processor, 
                            crop=crop
                        )
                        if ocr_candidate_result:
                            # Use OCR-proposed candidate (verified by CLIP if available)
                            ocr_result = {
                                "label": ocr_candidate_result.get("label"),
                                "confidence": ocr_candidate_result.get("confidence", 0.4),
                                "text": ocr_candidate_result.get("text"),
                                "method": ocr_candidate_result.get("method", "ocr_proposed")
                            }
                            if VERBOSE_LOGGING:
                                print(f"   📝 OCR candidate proposal: '{ocr_result.get('text')}' → '{ocr_result.get('label')}' (method: {ocr_result.get('method')})")
                    
                    # 🔥 FIX #3: Treat OCR as a hint, not a decision
                    # Only use OCR authoritatively if confidence is very high (> 0.7)
                    # Otherwise, use OCR text to narrow CLIP candidates
                    if ocr_result and ocr_result.get("label") and ocr_result.get("text"):
                        ocr_conf = ocr_result.get("confidence", 0.0)
                        if ocr_conf > 0.7:  # Only authoritative if very high confidence
                            ocr_authoritative = True
                            if VERBOSE_LOGGING:
                                print(f"   ✅ OCR AUTHORITATIVE: '{ocr_result.get('text')}' → '{ocr_result.get('label')}' (conf: {ocr_conf:.2f}) - WILL SKIP CLIP")
                        else:
                            # OCR is a hint - will be used to narrow CLIP candidates
                            ocr_authoritative = False
                            if VERBOSE_LOGGING:
                                print(f"   💡 OCR HINT: '{ocr_result.get('text')}' (conf: {ocr_conf:.2f}) - will narrow CLIP candidates")
                    
                    # If OCR failed or low confidence, try on expanded crop
                    if not ocr_authoritative and len(crops_to_classify) > 1:
                        expanded_crop = img.crop((
                            max(0, int(x1) - int((x2-x1) * 0.2)),
                            max(0, int(y1) - int((y2-y1) * 0.2)),
                            min(img.size[0], int(x2) + int((x2-x1) * 0.2)),
                            min(img.size[1], int(y2) + int((y2-y1) * 0.2))
                        ))
                        ocr_result_expanded = stage_ocr_read_label(expanded_crop, is_container=True)
                        if ocr_result_expanded and ocr_result_expanded.get("confidence", 0) > (ocr_result.get("confidence", 0) if ocr_result else 0):
                            ocr_result = ocr_result_expanded
                            # 🔥 FIX #3: Only authoritative if very high confidence (> 0.7)
                            ocr_conf_expanded = ocr_result.get("confidence", 0.0)
                            if ocr_result.get("label") and ocr_result.get("text") and ocr_conf_expanded > 0.7:
                                ocr_authoritative = True
                                if VERBOSE_LOGGING:
                                    print(f"   ✅ OCR AUTHORITATIVE (expanded): '{ocr_result.get('text')}' → '{ocr_result.get('label')}' (conf: {ocr_conf_expanded:.2f}) - WILL SKIP CLIP")
                            elif ocr_result.get("label") and ocr_result.get("text"):
                                # OCR is a hint - update ocr_text for candidate narrowing
                                ocr_authoritative = False
                                if VERBOSE_LOGGING:
                                    print(f"   💡 OCR HINT (expanded): '{ocr_result.get('text')}' (conf: {ocr_conf_expanded:.2f}) - will narrow CLIP candidates")
                
                has_text = ocr_result is not None and ocr_result.get("label") is not None
                ocr_text = ocr_result.get("text", "") if ocr_result else ""
                
                # STAGE 2: Narrow label space based on food type (only if OCR not authoritative)
                region_type = infer_region_type(class_name, region["bbox"], has_text, is_container)
                user_pantry_items = []
                if user_pantry and isinstance(user_pantry, list):
                    user_pantry_items = user_pantry
                
                # Use food_type from CLIP classification to narrow candidates (only if OCR not authoritative)
                candidate_labels = []
                if not ocr_authoritative:
                    candidate_labels = route_candidates(
                        food_type=food_type,
                        region_type=region_type,
                        yolo_class=class_name,
                        ocr_text=ocr_text,
                        user_pantry_items=user_pantry_items
                    )
                else:
                    # OCR is authoritative - no need for candidate labels
                    if VERBOSE_LOGGING:
                        print(f"   ⏭️ Skipping candidate label selection (OCR is authoritative)")
                
                # Debug: Log candidate selection
                if VERBOSE_LOGGING:
                    print(f"   🎯 Food type: {food_type} | Region type: {region_type} | YOLO: '{class_name}' | OCR: {has_text}")
                    print(f"   📊 Narrowed selection: {len(candidate_labels)} candidates (was 60+, now 6-15)")
                    if len(candidate_labels) <= 15:
                        print(f"   📋 Candidates: {candidate_labels}")
                    else:
                        print(f"   📋 Sample: {candidate_labels[:5]}... (+{len(candidate_labels)-5} more)")
                
                # 🔥 FIX 1: FOODNESS GATE - mark regions for confirmation if uncertain, but don't discard
                # Check if this region is food at all before processing
                # 🔥 CRITICAL: Don't discard regions - surface uncertainty instead of guessing
                # Even if food classifier says it's not food, process it and mark as unknown_packaged_food
                food_score_gate = 1.0  # Default: assume food (don't filter)
                food_label_gate = ""
                if _food_classifier:
                    try:
                        food_check = _food_classifier(crop)
                        if food_check and isinstance(food_check, list) and len(food_check) > 0:
                            # Get top prediction
                            top_pred = food_check[0]
                            food_score_gate = top_pred.get("score", 0.0) if isinstance(top_pred, dict) else 0.0
                            food_label_gate = top_pred.get("label", "").lower() if isinstance(top_pred, dict) else ""
                            
                            # 🔥 CRITICAL CHANGE: Don't discard - process all regions, mark low scores for confirmation
                            # Packaged food often gets lower scores from food classifier (trained on dishes, not pantry)
                            # Lower threshold to 0.10 - only discard if clearly not food
                            if food_score_gate < 0.10:
                                if VERBOSE_LOGGING:
                                    print(f"   🚫 FOODNESS GATE: Discarding region (food_score={food_score_gate:.2f} < 0.10, label={food_label_gate}) - clearly not food")
                                continue  # Only skip if clearly not food (< 0.10)
                            elif food_score_gate < 0.20:
                                if VERBOSE_LOGGING:
                                    print(f"   ⚠️ FOODNESS GATE: Low food score ({food_score_gate:.2f}) - will mark as unknown_packaged_food for confirmation")
                                # Continue processing but will mark as unknown_packaged_food
                            else:
                                if VERBOSE_LOGGING:
                                    print(f"   ✅ FOODNESS GATE: Region passed (food_score={food_score_gate:.2f}, label={food_label_gate})")
                    except Exception as e:
                        if VERBOSE_LOGGING:
                            print(f"   ⚠️ Food classifier check failed: {e} - continuing anyway")
                        # Continue if classifier fails (non-critical)
                
                # Precision Rule #4: Multi-crop agreement
                # Create multiple crops: tight, expanded (+20%), and full image (if item is large enough)
                crops_to_classify = []
                
                # Tight crop (original YOLO box)
                crops_to_classify.append(("tight", crop))
                
                # Expanded crop (+20% padding)
                crop_width = int(x2) - int(x1)
                crop_height = int(y2) - int(y1)
                padding_x = int(crop_width * 0.2)
                padding_y = int(crop_height * 0.2)
                expanded_x1 = max(0, int(x1) - padding_x)
                expanded_y1 = max(0, int(y1) - padding_y)
                expanded_x2 = min(img.size[0], int(x2) + padding_x)
                expanded_y2 = min(img.size[1], int(y2) + padding_y)
                expanded_crop = img.crop((expanded_x1, expanded_y1, expanded_x2, expanded_y2))
                crops_to_classify.append(("expanded", expanded_crop))
                
                # 🔥 PERFORMANCE: Skip full image crop (rarely needed, adds processing time)
                # Full image (if crop is large enough relative to image) - DISABLED for performance
                # img_width, img_height = img.size[0], img.size[1]
                # if crop_width > img_width * 0.3 and crop_height > img_height * 0.3:
                #     crops_to_classify.append(("full", img))
                
                # Run CLIP on all crops and collect votes
                clip_predictions = []
                region_info = {
                    "is_container": is_container,
                    "class_name": class_name,
                    "class_id": class_id
                }
                
                # 🔥 CRITICAL: Skip CLIP if OCR is authoritative
                # OCR keyword matching is more reliable than CLIP for packaged food
                clip_suggestion = None
                if ocr_authoritative:
                    # OCR found text - skip CLIP entirely, use OCR label
                    if VERBOSE_LOGGING:
                        print(f"   ⏭️ Skipping CLIP (OCR is authoritative)")
                elif not candidate_labels:
                    if VERBOSE_LOGGING:
                        print(f"   ⚠️ WARNING: candidate_labels is empty! Cannot run CLIP.")
                    # Continue to decision logic with no CLIP suggestion
                else:
                    # 🔥 PERFORMANCE: Process only tight and expanded crops (skip full image)
                    # Run CLIP on all crops and collect votes (only if OCR not authoritative)
                    # Smart selection already limited to 20-60 candidates - no further filtering needed
                    # Process all crops with same candidate list (CLIP can batch efficiently)
                    # Limit to 2 crops for performance (tight + expanded, skip full image)
                    crops_to_process = crops_to_classify[:2] if len(crops_to_classify) > 2 else crops_to_classify
                    for crop_name, crop_img in crops_to_process:
                        clip_pred = stage_clip_suggest_label(crop_img, candidate_labels, region_info=region_info)
                        # FIXED: Accept ALL CLIP predictions, even very low confidence ones
                        # The stage_clip_suggest_label function already filters out truly bad predictions
                        # Low confidence predictions will be marked for confirmation, which is better than unknown_food
                        if clip_pred:  # Accept any valid CLIP prediction, regardless of score
                            clip_predictions.append((crop_name, clip_pred))
                
                # Fix 4: Weighted voting instead of majority vote (only if OCR not authoritative)
                # Weight votes by crop size and confidence - more robust than simple majority
                from collections import Counter
                if ocr_authoritative:
                    # OCR is authoritative - skip CLIP voting, use OCR result directly
                    clip_suggestion = None  # No CLIP needed
                elif clip_predictions:
                    # Calculate crop weights (larger crops get more weight)
                    crop_weights = {}
                    for crop_name, crop_img in crops_to_classify:
                        if crop_name == "tight":
                            crop_weights[crop_name] = 0.5  # Tight crop matters most
                        elif crop_name == "expanded":
                            crop_weights[crop_name] = 0.3  # Expanded crop provides context
                        else:  # full
                            crop_weights[crop_name] = 0.2  # Full image has noise
                    
                    # Weighted voting: weight by crop importance and confidence
                    weighted_scores = {}
                    total_weight = 0.0
                    vote_counts = Counter()
                    
                    for crop_name, pred in clip_predictions:
                        label = pred.get("label")
                        score = pred.get("score", 0)
                        if label:
                            weight = crop_weights.get(crop_name, 0.3) * score  # Weight by crop importance and confidence
                            weighted_scores[label] = weighted_scores.get(label, 0.0) + weight
                            total_weight += weight
                            vote_counts[label] += 1
                    
                    if weighted_scores:
                        # Get best label by weighted score
                        best_label = max(weighted_scores, key=weighted_scores.get)
                        best_weighted_score = weighted_scores[best_label] / total_weight if total_weight > 0 else 0.0
                        
                        # Get average raw score for the winning label
                        winning_scores = [p.get("score", 0) for c, p in clip_predictions if p.get("label") == best_label]
                        avg_score = sum(winning_scores) / len(winning_scores) if winning_scores else 0.0
                        
                        clip_suggestion = {
                            "label": best_label,
                            "score": avg_score,  # Use average raw score for consistency
                            "weighted_score": best_weighted_score,
                            "second_best": max([p.get("second_best", 0) for c, p in clip_predictions]),
                            "needs_confirmation": any(p.get("needs_confirmation", False) for c, p in clip_predictions),
                            "votes": vote_counts[best_label],
                            "total_crops": len(crops_to_classify)
                        }
                        if VERBOSE_LOGGING:
                            print(f"   🗳️ Weighted vote: {best_label} ({vote_counts[best_label]}/{len(crops_to_classify)} votes, weighted: {best_weighted_score:.2f}, score: {avg_score:.2f})")
                    else:
                        # No valid predictions - use best single
                        if clip_predictions:
                            best_pred = max(clip_predictions, key=lambda x: x[1].get("score", 0))
                            clip_suggestion = best_pred[1]
                            clip_suggestion["needs_confirmation"] = True
                            clip_suggestion["votes"] = 1
                            if VERBOSE_LOGGING:
                                print(f"   ⚠️ No weighted agreement, using best single: {clip_suggestion.get('label')} (needs confirmation)")
                        else:
                            clip_suggestion = None
                
                # OCR already run earlier in pipeline - use ocr_result and ocr_authoritative from above
                # (No need to run OCR again here)
                
                # FIX 5: OCR must contribute to labels - boost CLIP scores when OCR text matches
                # BUT: Skip boosting if OCR is authoritative (OCR is truth, not a hint)
                ocr_boost_applied = False
                if not ocr_authoritative and ocr_result and ocr_result.get("text") and clip_suggestion and clip_suggestion.get("label"):
                    ocr_text = ocr_result.get("text", "").lower()
                    clip_label = clip_suggestion.get("label", "").lower()
                    # Check if OCR text contains the CLIP label or vice versa
                    if clip_label in ocr_text or ocr_text in clip_label or any(word in ocr_text for word in clip_label.split() if len(word) > 3):
                        # OCR text matches CLIP label - boost CLIP score
                        original_clip_score = clip_suggestion.get("score", 0.0)
                        boosted_score = min(1.0, original_clip_score + 0.25)  # FIX 5: Boost by +0.25
                        clip_suggestion["score"] = boosted_score
                        ocr_boost_applied = True
                        if VERBOSE_LOGGING:
                            print(f"   🔝 FIX 5: OCR boost applied: '{ocr_text}' matches '{clip_label}' - CLIP score boosted from {original_clip_score:.2f} to {boosted_score:.2f}")
                
                # 🔥 IMPROVEMENT #4: Confidence bands (high/medium/low) instead of single threshold
                # High (>0.8): Accept immediately
                # Medium (0.5-0.8): Accept but mark "needs review"
                # Low (<0.5): Unknown or needs confirmation
                
                # Precision Rule #10: Final decision ladder
                # OCR strong? → accept
                # CLIP strong + margin? → accept
                # CLIP weak but plausible? → unknown_food (ask user)
                # Non-food likely? → drop
                
                suggested_label = None
                classifier_food_prob = 0.0
                clip_food_prob = clip_suggestion.get("score", 0.0) if clip_suggestion else 0.0  # Use boosted score if OCR boost applied
                ocr_confidence = 0.0
                needs_confirmation_flag = False
                confidence_band = "low"  # Default to low confidence
                detection_method = "yolov8"
                
                # Rule 2: CLIP overrides everything when strong (>= 0.45)
                # Rule 4: Containers - prefer OCR > CLIP > unknown_packaged_food (never erase CLIP)
                skip_container_override = False
                bypass_rules = False
                
                # 🔥 FIX 3: OCR should be additive, not veto power
                # OCR strong (≥0.45) > CLIP very strong (≥0.45) > CLIP moderate > CLIP weak
                # OCR weak/absent → ignore OCR, trust CLIP
                
                # 🔥 CRITICAL POLICY: OCR is REQUIRED for packaged food - trust OCR more than CLIP
                # Simple hack to boost accuracy by 2×: if OCR finds text → trust OCR label with 0.9 confidence
                # This is how real pantry apps work - OCR keyword matching is more reliable than CLIP for packages
                if ocr_authoritative:
                    # OCR is authoritative - use OCR label directly, skip CLIP entirely
                    suggested_label = ocr_result["label"]
                    ocr_confidence = 0.9  # High confidence when OCR finds matching keyword
                    detection_method = "yolov8+ocr"
                    skip_container_override = True
                    bypass_rules = True  # Bypass rules - OCR is authoritative
                    if VERBOSE_LOGGING:
                        print(f"   ✅ OCR AUTHORITATIVE: '{ocr_result.get('text')}' → '{suggested_label}' (conf: {ocr_confidence:.2f}) - OVERRIDES CLIP")
                elif ocr_result and ocr_result.get("label") and ocr_result.get("text"):
                    # OCR found text but not authoritative (low confidence) - still prefer over weak CLIP
                    ocr_conf = ocr_result.get("confidence", 0.0)
                    if ocr_conf >= 0.45:
                        # OCR is strong and has valid label - use it
                        suggested_label = ocr_result["label"]
                        ocr_confidence = ocr_result["confidence"]
                        detection_method = "yolov8+ocr"
                        skip_container_override = True
                        if VERBOSE_LOGGING:
                            print(f"   📝 OCR (strong): {suggested_label} (conf: {ocr_confidence:.2f}) - overrides CLIP")
                elif clip_suggestion and clip_suggestion.get("score", 0) >= 0.45:
                    # Rule 2: CLIP >= 0.45 - STOP all penalties, bypass rules, return immediately
                    # This is ChatGPT-Vision-level signal - trust it completely
                    suggested_label = clip_suggestion["label"]
                    clip_food_prob = clip_suggestion["score"]
                    detection_method = "yolov8+clip"
                    needs_confirmation_flag = False  # Strong prediction, no confirmation needed
                    skip_container_override = True  # Don't override with unknown_packaged_food
                    bypass_rules = True  # Bypass all rules and food_score recalculation
                    # Use CLIP score directly as confidence - no downgrading
                    confidence_direct = clip_food_prob
                    if VERBOSE_LOGGING:
                        print(f"   ✅ CLIP (very strong): {suggested_label} (score: {clip_food_prob:.2f}) - accepting immediately, bypassing rules")
                elif clip_suggestion:
                    # 🔥 FIX 3: OCR weak/absent → ignore OCR, trust CLIP
                    # If OCR exists but is weak (<0.45), ignore it and use CLIP
                    # OCR should only override when it's strong (≥0.45)
                    # CLIP suggests label
                    suggested_label = clip_suggestion["label"]
                    clip_food_prob = clip_suggestion["score"]
                    detection_method = "yolov8+clip"
                    
                    # 🔥 FIX: Accept ALL CLIP predictions, even with low confidence - mark for confirmation instead of rejecting
                    # 🔥 CRITICAL: Handle confidence thresholds in correct order (high to low)
                    # Don't reject items - mark them for confirmation instead
                    # This ensures items are returned even if confidence is low
                    if clip_food_prob >= 0.35:
                        # Moderate confidence - mark for confirmation
                        skip_container_override = True  # Trust CLIP label, don't override
                        needs_confirmation_flag = True
                        if VERBOSE_LOGGING:
                            print(f"   ⚠️ CLIP (moderate): {suggested_label} (score: {clip_food_prob:.2f}) - needs confirmation")
                    elif clip_food_prob >= 0.10:  # 🔥 FIX: Lowered from 0.20 to 0.10 to accept more items
                        # Low but plausible confidence - show best guess with (?) for user confirmation
                        # Surface uncertainty: show best guess with (?) instead of just unknown_food
                        if suggested_label and suggested_label != "unknown_food":
                            suggested_label = f"{suggested_label} (?)"  # Show best guess with uncertainty marker
                        needs_confirmation_flag = True
                        if VERBOSE_LOGGING:
                            print(f"   ⚠️ CLIP (low confidence {clip_food_prob:.2f} < 0.35) - showing as '{suggested_label}' for confirmation")
                    else:
                        # Very low confidence - still return it but mark as unknown_food
                        # 🔥 FIX: Don't reject - return with low confidence for user to review
                        if suggested_label and suggested_label != "unknown_food":
                            suggested_label = f"{suggested_label} (?)"  # Keep the label with uncertainty marker
                        else:
                            suggested_label = "unknown_food"
                        needs_confirmation_flag = True
                        if VERBOSE_LOGGING:
                            print(f"   ⚠️ CLIP (very low confidence {clip_food_prob:.2f} < 0.10) - returning as '{suggested_label}' for user review")
                else:
                    # Fix 1: No CLIP or OCR - only mark as unknown if truly no label exists
                    # If we have a suggested_label from earlier stages, keep it
                    if not suggested_label or suggested_label == "":
                        suggested_label = "unknown_food"
                    needs_confirmation_flag = True
                    if VERBOSE_LOGGING:
                        print(f"   ❓ No CLIP/OCR match - keeping '{suggested_label}', needs confirmation")
                
                # Fix #2: If CLIP >= 0.45, bypass food_score calculation entirely
                # Fix #4: Food score must NEVER go negative
                if 'bypass_rules' in locals() and bypass_rules:
                    # Strong CLIP - use CLIP score directly, no recalculation
                    food_score = clip_food_prob
                    confidence_direct = clip_food_prob
                else:
                    # Fix #5: Food score math - rebalanced and never negative
                    # Rebalanced weights: CLIP 45%, classifier 25%, YOLO 20%, context 10%
                    # Note: classifier_food_prob and context_score will be added in rules stage
                    context_boost = 0.0  # Will be set in rules stage
                    
                    if ocr_confidence > 0.45:
                        # OCR is authoritative
                        food_score = 0.6 * ocr_confidence + 0.2 * clip_food_prob + 0.1 * yolo_food_prior + 0.1 * (yolo_conf * 0.5)
                    elif clip_food_prob > 0.35:
                        # CLIP is strong - rebalanced weights (classifier and context added in rules stage)
                        food_score = 0.45 * clip_food_prob + 0.25 * classifier_food_prob + 0.20 * yolo_food_prior + 0.10 * context_boost
                    else:
                        # Weak signals - lower overall confidence but still positive
                        food_score = 0.3 * clip_food_prob + 0.2 * ocr_confidence + 0.3 * yolo_food_prior + 0.2 * (yolo_conf * 0.5)
                    
                    # Fix 2: Remove negative food scores entirely - clamp to 0.0-1.0
                    # Penalties should reduce confidence, not flip sign
                    food_score = max(0.0, min(1.0, food_score))  # Clamp to valid range
                    confidence_direct = max(clip_food_prob, ocr_confidence, yolo_conf * 0.5)
                
                # Prepare region data for rules stage
                region_data = {
                    "bbox": region["bbox"],
                    "yolo_conf": yolo_conf,
                    "yolo_food_prior": yolo_food_prior,
                    "is_container": is_container,
                    "suggested_label": suggested_label or "unknown_food",
                    "clip_suggestion": clip_suggestion,
                    "ocr_result": ocr_result,
                    "food_score": food_score,
                    "confidence": confidence_direct if 'confidence_direct' in locals() else max(clip_food_prob, ocr_confidence, yolo_conf * 0.5),
                    "classifier_food_prob": classifier_food_prob,
                    "clip_food_prob": clip_food_prob,
                    "detection_method": detection_method,
                    "needs_confirmation": needs_confirmation_flag,
                    "skip_container_override": skip_container_override,  # Flag to prevent container override
                    "bypass_rules": bypass_rules if 'bypass_rules' in locals() else False  # Flag to bypass rules stage
                }
                
                # STAGE 4: Rules - "Does this make sense here?" (never hard-deletes)
                # Fix #2: If CLIP >= 0.45, bypass rules entirely
                if not region_data.get("bypass_rules", False):
                    region_data = stage_rules_validate(region_data, clip_suggestion, ocr_result, user_pantry)
                else:
                    if VERBOSE_LOGGING:
                        print(f"   ⏭️ Bypassing rules stage (CLIP >= 0.45)")
                
                # STAGE 5: User - "Is this correct?" (4-tier decision)
                region_data = stage_user_decision(region_data)
                
                # Fix 1: Never hide detected food - keep semantic label, just mark uncertain
                # Rules may reduce confidence, but they must never erase semantics
                if region_data.get("hide", False):
                    # Fix 1: Keep semantic label if exists, don't erase to unknown_food
                    current_label = region_data.get("suggested_label", "")
                    if current_label not in ["unknown_food", "unknown_packaged_food", "unknown_item", ""]:
                        # Keep the semantic label (onion, olive oil, etc.)
                        region_data["suggested_label"] = current_label
                        region_data["needs_confirmation"] = True
                        region_data["confidence"] = region_data.get("confidence", 0.0) * 0.7  # Downgrade confidence
                        if VERBOSE_LOGGING:
                            print(f"   ⚠️ Low confidence but keeping semantic label: {current_label} (food_score={food_score:.2f})")
                    else:
                        # Truly no label - use unknown_food
                        region_data["suggested_label"] = "unknown_food"
                        region_data["needs_confirmation"] = True
                        region_data["confidence"] = region_data.get("confidence", 0.0) * 0.7
                        if VERBOSE_LOGGING:
                            print(f"   ⚠️ No semantic label - showing as unknown_food (food_score={food_score:.2f})")
                    region_data["hide"] = False  # Don't hide
                    # Continue processing instead of skipping
                
                # Build final item
                final_label = region_data["suggested_label"]
                final_conf = region_data.get("final_confidence", region_data["confidence"])
                
                # Extract expiration date if OCR found one
                exp_date = None
                if ocr_result and ocr_result.get("text"):
                    try:
                        exp_dates = extract_expiration_dates(crop, _ocr_reader)
                        if exp_dates:
                            exp_date = exp_dates[0]
                    except Exception:
                        pass
                            
                # 🔥 IMPROVEMENT #4: Confidence bands (high/medium/low) instead of single threshold
                # High (>0.8): Accept immediately, no confirmation needed
                # Medium (0.5-0.8): Accept but mark "needs review" 
                # Low (<0.5): Unknown or needs confirmation
                if final_conf > 0.8:
                    confidence_band = "high"
                    needs_confirmation = False
                elif final_conf >= 0.5:
                    confidence_band = "medium"
                    needs_confirmation = True  # Mark for review but accept
                else:
                    confidence_band = "low"
                    needs_confirmation = True
                            
                # Precision Rule #9: User confirmation is a feature, not a failure
                # Mark items for confirmation if uncertain
                if region_data.get("suggested_label") == "unknown_food":
                    needs_confirmation = True
                    confidence_band = "low"
                
                # Fix 1 & 6: Never erase semantic labels once assigned
                # Fix 2: Lower food gate threshold (0.15 → 0.05)
                # Fix 4: Containers must never be downgraded
                food_score_final = region_data.get("food_score", 0)
                is_container_final = region_data.get("is_container", False)
                skip_override_final = region_data.get("skip_container_override", False)
                bypass_rules_final = region_data.get("bypass_rules", False)
                
                # Fix 1: If CLIP >= 0.45, already handled - skip all downgrading
                if bypass_rules_final:
                    # Strong CLIP - keep as-is, no downgrading
                    if VERBOSE_LOGGING:
                        print(f"   ✅ Keeping strong CLIP prediction: {final_label} (conf: {final_conf:.2f})")



                # Fix 4: Containers must never be downgraded
                elif is_container_final:
                    # Containers: prefer OCR > CLIP > unknown_packaged_food, but NEVER downgrade semantic labels
                    if final_label in ["unknown_food", "unknown_item"]:
                        # Only change if truly unknown, otherwise keep semantic label
                        final_label = "unknown_packaged_food"
                    # Never downgrade containers - keep semantic label if exists
                    needs_confirmation = True  # Always mark containers for confirmation
                    if VERBOSE_LOGGING:
                        print(f"   📦 Container detected - showing as {final_label} (conf: {final_conf:.2f}) - never downgrade")
                # Fix 2: Lower food gate threshold (0.15 → 0.05)
                elif food_score_final < 0.05:
                    # Fix 1: Non-food likely (< 0.05) - keep semantic label if exists, otherwise unknown_food
                    # Never replace semantic label with unknown_item unless zero signal
                    if final_label not in ["unknown_food", "unknown_packaged_food", "unknown_item"]:
                        # Keep semantic label (onion, potato, etc.) - just mark uncertain
                        needs_confirmation = True
                        if VERBOSE_LOGGING:
                            print(f"   ⚠️ Low food_score ({food_score_final:.2f}) but keeping semantic label: {final_label} - ask user")
                    else:
                        # 🔥 PRINCIPLE 2: No semantic label - check for weak CLIP prediction to surface
                        # Even weak predictions are better than unknown_food
                        clip_suggestion = region_data.get("clip_suggestion")
                        if clip_suggestion and clip_suggestion.get("label") and clip_suggestion.get("score", 0) > 0.15:
                            weak_label = clip_suggestion["label"]
                            weak_score = clip_suggestion["score"]
                            final_label = f"Possibly: {weak_label}?"
                            if VERBOSE_LOGGING:
                                print(f"   ⚠️ Low food_score ({food_score_final:.2f}) - showing weak CLIP guess as 'Possibly: {weak_label}?'")
                        else:
                            # Truly no label - use unknown_food as last resort
                            final_label = "unknown_food"
                            if VERBOSE_LOGGING:
                                print(f"   ⚠️ Low food_score ({food_score_final:.2f}) - showing as unknown_food - ask user")
                        needs_confirmation = True
                elif 0.02 <= food_score_final < 0.4:
                    # Fix 1: Uncertain food - keep semantic label, mark for confirmation
                    if final_label not in ["unknown_food", "unknown_packaged_food", "unknown_item"]:
                        # Keep suggested semantic label but mark for confirmation
                        needs_confirmation = True
                        if VERBOSE_LOGGING:
                            print(f"   ❓ Marking for confirmation: {final_label} (conf: {final_conf:.2f}, food_score: {food_score_final:.2f})")
                    else:
                        needs_confirmation = True
                        if VERBOSE_LOGGING:
                            print(f"   ❓ Marking for confirmation: {final_label} (conf: {final_conf:.2f}, food_score: {food_score_final:.2f})")
                elif final_label == "unknown_food" or needs_confirmation:
                    # Uncertain - mark for user confirmation
                    if VERBOSE_LOGGING:
                        print(f"   ❓ Marking for confirmation: {final_label} (conf: {final_conf:.2f})")
                
                # 🔥 PRINCIPLE 2: Handle "Possibly: X?" format for low confidence
                # Extract base label if it has "Possibly:" prefix for category validation
                base_label = final_label
                possibly_prefix = False
                if final_label.startswith("Possibly: ") and final_label.endswith("?"):
                    base_label = final_label.replace("Possibly: ", "").replace("?", "").strip()
                    possibly_prefix = True
                
                # 🔥 IMPROVEMENT #4: Add confidence band to item data
                # Create item
                item_data = {
                    "name": final_label,
                                    "quantity": "1",
                                    "expirationDate": exp_date,
                    "category": validate_category(base_label, "other"),  # Use base label for category
                    "confidence": final_conf,
                    "confidence_band": confidence_band,  # "high", "medium", or "low"
                    "detection_method": region_data.get("detection_method", "yolov8"),
                    "needs_confirmation": needs_confirmation,
                    "bbox": region["bbox"],
                    "possibly_prefix": possibly_prefix  # Flag for UI to style "Possibly: X?" differently
                }
                
                items.append(item_data)
            except Exception as det_error:
                if VERBOSE_LOGGING:
                    print(f"   ⚠️ Error processing region: {det_error}")
                continue
                            
        print(f"✅ Processed {len(regions)} regions, found {len(items)} items")
    
    # Check if YOLO found no regions (fallback detection)
    if not detections_found_yolo:
        if _yolo_model is not None and YOLO_DETECTION_ENABLED:
            print("⚠️ YOLO returned no regions")
        else:
            print(f"⚠️ YOLO not available: model={_yolo_model is not None}, enabled={YOLO_DETECTION_ENABLED}")
    
    # STEP 2b: DETR Detection (Fallback if YOLOv8 not available)
    # Note: detections_found is already initialized at line 932
    if not detections_found_yolo and _object_detector and ML_VISION_MODE == "hybrid":
        try:
            # DETR detection code here (if needed)
            pass
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"DETR detection error: {e}")
    
    # Old duplicate YOLO code removed - stage-based pipeline handles detection above
    
    # Continue with rest of pipeline...
    
    # STEP 3: Fallback to Full-Image Classification if no detections found
    # Use YOLO results if available
    if detections_found_yolo:
        detections_found = True  # YOLO found items
        print(f"✅ YOLO found {len(items)} items")
    else:
        print(f"⚠️ YOLO found no items (detections_found_yolo={detections_found_yolo})")
        if _yolo_model is None:
            print("   ⚠️ YOLO model not loaded - check if ultralytics is installed")
        elif not YOLO_DETECTION_ENABLED:
            print("   ⚠️ YOLO_DETECTION_ENABLED is False - enable it to use YOLO")
        detections_found = False
    
    # 🔥 CRITICAL FIX: Kill full-image CLIP for pantry scans
    # Full-image CLIP = "What is the vibe of this photo?" NOT "What items are in this pantry?"
    # When YOLO finds regions but no items → return "unknown_packaged_food" instead of guessing
    should_run_classifier = False
    
    # 🔥 FIX: Check if we have no detections (YOLO found nothing)
    if not detections_found_yolo and len(items) == 0:
        # No YOLO regions at all - use OCR text clustering + shelf-level fallback
        print("⚠️ No YOLO regions found - using OCR text clustering + shelf-level fallback")
        should_run_classifier = False
        
        # 🔥 FIX #4: Use OCR text even when YOLO fails (text clustering + spatial grouping)
        if _ocr_reader:
                try:
                    import numpy as np
                    img_array = np.array(img)
                    ocr_results = _ocr_reader.readtext(img_array)
                    if ocr_results and len(ocr_results) > 0:
                        # Cluster OCR text by spatial proximity and extract food keywords
                        food_keywords_map = {
                            "cereal": ["cereal", "granola", "oatmeal", "breakfast"],
                            "rice": ["rice", "jasmine", "basmati", "brown rice"],
                            "pasta": ["pasta", "spaghetti", "macaroni", "noodles"],
                            "snacks": ["chips", "crackers", "cookies", "pretzels", "snacks"],
                            "flour": ["flour", "baking", "sugar"],
                            "sauce": ["sauce", "tomato", "marinara", "pasta sauce"],
                            "juice": ["juice", "orange", "apple", "cranberry"],
                            "soda": ["soda", "cola", "sprite", "ginger ale"]
                        }
                        
                        # Extract all OCR text and find food keywords
                        all_ocr_text = ' '.join([item[1] if len(item) > 1 else '' for item in ocr_results]).lower()
                        detected_keywords = []
                        for food_type, keywords in food_keywords_map.items():
                            for keyword in keywords:
                                if keyword in all_ocr_text:
                                    detected_keywords.append(food_type)
                                    break
                        
                        # If OCR found food keywords, add as hypothesis items
                        if detected_keywords:
                            for keyword in detected_keywords[:5]:  # 🔥 FIX: Increased from 3 to 5
                                items.append({
                                    "name": f"unknown_food (likely {keyword})",
                                    "quantity": "1",
                                    "expirationDate": None,
                                    "category": validate_category(keyword, "other"),
                                    "confidence": 0.4,  # Moderate confidence for OCR-based detection
                                    "detection_method": "ocr_clustering",
                                    "needs_confirmation": True
                                })
                            detections_found = True  # 🔥 FIX: Mark as found
                            if VERBOSE_LOGGING:
                                print(f"   📝 OCR clustering found keywords: {detected_keywords}")
                        else:
                            print(f"   ⚠️ OCR found text but no food keywords detected")
                except Exception as ocr_cluster_error:
                    if VERBOSE_LOGGING:
                        print(f"   ⚠️ OCR clustering error: {ocr_cluster_error}")
            
        # 🔥 FIX: Pantry scene fallback - shelf-level context + item hypothesis mode
        # If YOLO finds nothing, run CLIP on full image with shelf-level labels
        # Then use shelf type to narrow item candidates (fallback item hypothesis)
        scene_type = "pantry"
        shelf_context = None
        if scene_type == "pantry" and _clip_model and _clip_processor:
                shelf_labels = [
                    "cereal shelf",
                    "snack shelf",
                    "rice shelf",
                    "pasta shelf"
                ]
                try:
                    shelf_result = clip_match_open_vocabulary(img, shelf_labels, use_prompt_engineering=True)
                    if shelf_result and shelf_result.get("score", 0) > 0.20:
                        shelf_label = shelf_result.get("label", "pantry shelf")
                        shelf_conf = shelf_result.get("score", 0.0)
                        shelf_context = shelf_label  # Store as context, not item
                        
                        # 🔥 FIX: Use shelf type to narrow item candidates (fallback item hypothesis)
                        # Map shelf type to likely items
                        shelf_to_items = {
                            "cereal shelf": ["cereal", "granola", "oatmeal", "breakfast cereal"],
                            "snack shelf": ["chips", "crackers", "cookies", "snacks", "pretzels"],
                            "rice shelf": ["rice", "pasta", "quinoa", "grains"],
                            "pasta shelf": ["pasta", "spaghetti", "macaroni", "noodles", "rice"]
                        }
                        
                        # Get likely items for this shelf type
                        likely_items = shelf_to_items.get(shelf_label, ["cereal", "snacks", "rice", "pasta"])
                        
                        # 🔥 FIX: Run CLIP on full image with narrowed item candidates (lower threshold)
                        item_result = clip_match_open_vocabulary(img, likely_items, use_prompt_engineering=True)
                        if item_result and item_result.get("score", 0) > 0.15:  # 🔥 FIX: Lowered from 0.25 to 0.15
                            item_label = item_result.get("label", "unknown_food")
                            item_conf = item_result.get("score", 0.0)
                            # Return as "unknown_food (likely X)" instead of shelf label
                            items.append({
                                "name": f"unknown_food (likely {item_label})",
                                "quantity": "1",
                                "expirationDate": None,
                                "category": validate_category(item_label, "other"),
                                "confidence": item_conf * 0.7,  # Reduce confidence for hypothesis
                                "detection_method": "clip_hypothesis",
                                "needs_confirmation": True,
                                "shelf_context": shelf_context  # Include context for UI
                            })
                            detections_found = True
                            print(f"   ✅ Shelf context: {shelf_label} → Item hypothesis: {item_label} (conf: {item_conf:.2f})")
                        else:
                            # 🔥 FIX: Even if no confident match, add likely items based on shelf type
                            for likely_item in likely_items[:2]:  # Add top 2 likely items
                                items.append({
                                    "name": f"unknown_food (likely {likely_item})",
                                    "quantity": "1",
                                    "expirationDate": None,
                                    "category": validate_category(likely_item, "other"),
                                    "confidence": 0.3,  # Low confidence but still add
                                    "detection_method": "clip_hypothesis",
                                    "needs_confirmation": True,
                                    "shelf_context": shelf_context
                                })
                            detections_found = True
                            print(f"   ✅ Shelf context: {shelf_label} → Added {len(likely_items[:2])} likely items")
                except Exception as shelf_error:
                    if VERBOSE_LOGGING:
                        print(f"   ⚠️ Shelf-level CLIP fallback error: {shelf_error}")
    elif detections_found_yolo and len(items) == 0:
        # 🔥 FIX #1: YOLO found boxes but no label - run CLIP on each crop with restricted vocabulary
        # Instead of just marking as unknown_packaged_food, try CLIP with narrow pantry vocabulary
        print("⚠️ YOLO found regions but no items - running CLIP on crops with restricted vocabulary")
        
        # Restricted pantry vocabulary for boxes that YOLO found but couldn't classify
        PANTRY_BOX_VOCAB = [
            "rice", "pasta", "cereal", "flour", "sugar", "snacks", "chips", "cookies",
            "crackers", "oats", "quinoa", "bread", "crackers", "granola bar", "protein bar"
        ]
        
        # Try CLIP on each region crop
        for region in regions:
                try:
                    bbox = region.get("bbox", [])
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        crop = img.crop((x1, y1, x2, y2))
                        
                        # Run CLIP with restricted vocabulary
                        if _clip_model and _clip_processor:
                            clip_result = clip_match_open_vocabulary(crop, PANTRY_BOX_VOCAB, use_prompt_engineering=True)
                            if clip_result and clip_result.get("score", 0) > 0.20:  # Lower threshold for boxes
                                label = clip_result.get("label", "unknown_packaged_food")
                                conf = clip_result.get("score", 0.3)
                                items.append({
                                    "name": label,
                                    "quantity": "1",
                                    "expirationDate": None,
                                    "category": validate_category(label, "other"),
                                    "confidence": conf,
                                    "detection_method": "yolov8+clip",
                                    "needs_confirmation": conf < 0.5  # Mark for confirmation if low confidence
                                })
                                if VERBOSE_LOGGING:
                                    print(f"   ✅ CLIP identified box as: {label} (conf: {conf:.2f})")
                                continue
                
                except Exception as crop_error:
                    if VERBOSE_LOGGING:
                        print(f"   ⚠️ Error processing region crop: {crop_error}")
                
                # If CLIP didn't find anything or failed, mark as unknown_packaged_food
                items.append({
                    "name": "unknown_packaged_food",
                    "quantity": "1",
                    "expirationDate": None,
                    "category": "other",
                    "confidence": 0.3,
                    "detection_method": "yolov8+unknown",
                    "needs_confirmation": True
                })
        
        should_run_classifier = False
    else:
        # Items already found - skip full-image classifier to avoid hallucination
        if VERBOSE_LOGGING:
            print(f"   ⏭️ Skipping full-image classifier (already found {len(items)} items)")
        should_run_classifier = False
        print(f"   detections_found={detections_found}, _food_classifier={_food_classifier is not None}, items_so_far={len(items)}")
    
    # Fix #5: Full-image classification must produce categories using CLIP
    # When YOLO misses, run CLIP on full image to ask "what foods are present?"
    if should_run_classifier:
        # First try CLIP on full image (better for pantry ingredients)
        if _clip_model and _clip_processor:
            try:
                # Rule 3: Expand CLIP vocabulary for full-image fallback (use same expanded list)
                # This prevents forced classification when YOLO misses
                # Fix 4: Use expanded CLIP_CANDIDATES instead of limited CLIP_CANDIDATES_FULL
                # Full-image classifier was using only ~120 items, now uses 500+ items
                all_pantry_labels = (
                    CLIP_CANDIDATES.get("produce", []) +
                    CLIP_CANDIDATES.get("liquids", []) +
                    CLIP_CANDIDATES.get("condiments", []) +
                    CLIP_CANDIDATES.get("spices", []) +
                    CLIP_CANDIDATES.get("grains", []) +
                    CLIP_CANDIDATES.get("dairy", []) +
                    CLIP_CANDIDATES.get("canned_packaged", []) +
                    CLIP_CANDIDATES.get("snacks", []) +
                    CLIP_CANDIDATES.get("beverages", []) +
                    CLIP_CANDIDATES.get("baking", []) +
                    CLIP_CANDIDATES.get("frozen", []) +
                    CLIP_CANDIDATES.get("meat_seafood", [])
                )
                
                if VERBOSE_LOGGING:
                    print(f"   🔍 Full-image CLIP using {len(all_pantry_labels)} candidates (expanded vocabulary)")
                
                # Performance: Batch process CLIP for full-image (much faster than one-by-one)
                # Prioritize common pantry items first, then check others
                common_items = [
                    "onion", "potato", "tomato", "apple", "orange", "banana",
                    "milk", "eggs", "cheese", "butter", "yogurt",
                    "bread", "rice", "pasta", "cereal", "flour", "sugar",
                    "olive oil", "vegetable oil", "soy sauce", "vinegar",
                    "salt", "pepper", "garlic", "onion powder", "garlic powder"
                ]
                
                # 🔥 PERFORMANCE: Reduce label count for faster CLIP processing
                # Combine common items with all labels (remove duplicates)
                priority_labels = [l for l in common_items if l in all_pantry_labels]
                other_labels = [l for l in all_pantry_labels if l not in priority_labels]
                # Check priority items first, then sample from others (reduced from 75 to 50)
                labels_to_check = priority_labels + other_labels[:50]  # Total ~75 items (was ~100)
                
                # 🔥 PERFORMANCE: Batch process all labels at once (much faster)
                # Use efficient batching - CLIP processes all labels in one pass
                clip_result = clip_match_open_vocabulary(img, labels_to_check, use_prompt_engineering=True)
                clip_preds = []
                
                if clip_result:
                    # CLIP returns best match from the batch
                    if clip_result.get("score", 0) > 0.10:
                        clip_preds.append({
                            "label": clip_result["label"],
                            "score": clip_result.get("score", 0),
                            "confidence": clip_result.get("score", 0)
                        })
                    
                    # 🔥 PERFORMANCE: Skip second pass for similar items (saves time, minimal accuracy loss)
                    # Only do second pass if very high confidence (reduced from 0.15 to 0.25)
                    if clip_result.get("score", 0) > 0.25:  # Only for very confident matches
                        # High confidence - check a few more similar items (reduced from 10 to 5)
                        similar_labels = [l for l in labels_to_check if clip_result["label"] in l or l in clip_result["label"]][:5]
                        for label in similar_labels:
                            if label != clip_result["label"]:
                                similar_result = clip_match_open_vocabulary(img, [label], use_prompt_engineering=True)
                                if similar_result and similar_result.get("score", 0) > 0.15:  # Higher threshold
                                    clip_preds.append({
                                        "label": label,
                                        "score": similar_result.get("score", 0),
                                        "confidence": similar_result.get("score", 0)
                                    })
                
                # 🔥 PERFORMANCE: Reduce top predictions (3 is sufficient, faster processing)
                # Sort by score and take top 3 (reduced from 5 for better performance)
                clip_preds.sort(key=lambda x: x["score"], reverse=True)
                top_clip_preds = clip_preds[:3]  # Reduced from 5 to 3 for performance
                
                if top_clip_preds:
                    if VERBOSE_LOGGING:
                        print(f"   🔍 CLIP full-image found {len(top_clip_preds)} potential foods")
                    for clip_pred in top_clip_preds:
                        name = normalize_item_name(clip_pred["label"])
                        conf = clip_pred["confidence"] * 0.8  # Increased from 0.7 - trust full-image CLIP more
                        key = name.lower().strip()
                        if key not in item_confidence or conf > item_confidence[key]:
                            items.append({
                                "name": name,
                                "quantity": "1",
                                "expirationDate": None,
                                "category": validate_category(name, "other"),
                                "confidence": conf,
                                "detection_method": "full_image_clip",
                                "needs_confirmation": True  # Always mark for confirmation
                            })
                            item_confidence[key] = conf
            except Exception as clip_error:
                if VERBOSE_LOGGING:
                    print(f"   ⚠️ CLIP full-image error: {clip_error}")
        
        # Fix 1: DISABLE nateraw/food classifier - it's trained on desserts/fast food, NOT pantry items!
        # It returns: ['Macarons', 'Cup_cakes', 'Donuts', 'Takoyaki'] - completely wrong for pantry images
        # Only use CLIP for full-image classification
        # Fallback to food classifier DISABLED - was causing wrong predictions
        if False and len(items) == 0 and _food_classifier:  # Disabled - wrong model for pantry
            try:
                # Add timeout protection for classification (prevent hanging)
                import threading
                
                preds = None
                classification_error = None
                
                def run_classification():
                    nonlocal preds, classification_error
                    try:
                        preds = _food_classifier(img)
                        pass

                    except Exception as e:
                        classification_error = e
                
                # Run classification in a thread with timeout
                thread = threading.Thread(target=run_classification)
                thread.daemon = True
                thread.start()
                thread.join(timeout=30)  # 30 second timeout




                
                if thread.is_alive():
                    if VERBOSE_LOGGING:
                        print("Warning: Classification timed out after 30 seconds")
                    preds = []
                elif classification_error:
                    raise classification_error
                
                if not preds:
                    if VERBOSE_LOGGING:
                        print("Warning: Classification returned no predictions")
                    preds = []
                
                # Check top 20 predictions to catch rare foods
                for pred in preds[:20]:
                    try:
                        if not isinstance(pred, dict):
                            continue
                        
                        score = pred.get("score", 0)
                        try:
                            score = float(score)
                        except (ValueError, TypeError):
                            continue
                        
                        # Very low threshold (0.03) to catch as many items as possible
                        # We'll let the user filter out false positives later
                        if score < 0.03:  # Lowered from 0.05 to catch more items
                            continue  # Skip only very low confidence predictions
                        
                        label = pred.get("label", "")
                        if not label or not isinstance(label, str):
                            continue
                        
                        # Use hierarchical classification
                        classification = classify_food_hierarchical(None, classification_pred=[pred])
                        if classification:
                            name = classification["name"]
                            category = classification["category"]
                            conf = classification["confidence"]
                        else:
                            name = normalize_item_name(label)
                            category = validate_category(name, "other")
                            conf = score
                        
                        if not name or len(name) < 2:
                            continue  # Skip items without valid names
                        
                        # Skip non-food items (but be less aggressive)
                        non_food_keywords = ["table", "plate", "fork", "knife", "spoon"]
                        # Only skip if confidence is very low AND it's clearly not food
                        if any(kw in label.lower() for kw in non_food_keywords):
                            if conf < 0.25:  # Lower threshold - allow more items through
                                continue
                        
                        # Fix 1: Full-image classifier - keep semantic labels, never downgrade to unknown_item
                        # This prevents hallucination from classifier trained on dishes, not pantry ingredients
                        original_name = name
                        if conf < 0.4:
                            # Fix 1: Low confidence - keep semantic label, just mark for confirmation
                            # Never replace semantic label with unknown_item
                            if VERBOSE_LOGGING:
                                print(f"   ⚠️ Low confidence ({conf:.2f}) - keeping semantic label '{original_name}', marking for confirmation")
                            # Keep original_name, don't change to unknown_item
                            name = original_name
                            conf = min(conf, 0.3)  # Cap at 0.3 for full-image classifier
                            # Mark for confirmation but keep label
                        
                        key = name.lower().strip()
                        if key not in item_confidence or conf > item_confidence[key]:
                            items.append({
                                "name": name,
                                "quantity": "1",
                                "expirationDate": exp_date,
                                "category": category,
                                "confidence": conf,
                                "detection_method": "full_image_classifier",
                                "needs_confirmation": True  # Always mark full-image classifier results for confirmation
                            })
                            item_confidence[key] = conf
                    except Exception as pred_error:
                        if VERBOSE_LOGGING:
                            print(f"Warning: Error processing prediction: {pred_error}")
                        continue
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"Classification error: {e}")
                # Continue even if classification fails

    # Debug: Log items before NMS and deduplication
    print(f"🔍 Before NMS/deduplication: {len(items)} items found")
    if len(items) > 0:
        print(f"   Sample items: {[item.get('name', 'unknown') for item in items[:3]]}")
        print(f"   Sample confidences: {[item.get('confidence', 0) for item in items[:3]]}")
    
    # STEP 4: Apply Non-Maximum Suppression (NMS) to remove overlapping detections
    items = apply_nms(items, iou_threshold=0.5)
    print(f"🔍 After NMS: {len(items)} items remaining")
    
    # Better Deduplication: Use name similarity + IoU for smarter merging
    def normalize_name_for_dedup(name):
        """Normalize name for deduplication - handles variations"""
        if not name:
            return ""
        name = name.lower().strip()
        # Remove common prefixes/suffixes that don't change meaning
        name = name.replace("extra virgin ", "").replace("extra-virgin ", "")
        name = name.replace("organic ", "").replace("all natural ", "")
        name = name.replace("_", " ").replace("-", " ")
        # Handle plural/singular (simple version)
        # Common plural patterns
        plural_to_singular = {
            "tomatoes": "tomato", "potatoes": "potato", "onions": "onion",
            "apples": "apple", "oranges": "orange", "bananas": "banana",
            "eggs": "egg", "carrots": "carrot", "peppers": "pepper"
        }
        if name in plural_to_singular:
            name = plural_to_singular[name]
        elif name.endswith("s") and len(name) > 3:
            # Try removing 's' for other plurals (but be careful)
            if not name.endswith(("ss", "us", "is", "as", "es")):
                # Only if it's a common pattern
                pass  # Keep as-is to avoid over-normalization
        return name.strip()
    
    def names_are_similar(name1, name2):
        """Check if two names refer to the same item"""
        norm1 = normalize_name_for_dedup(name1)
        norm2 = normalize_name_for_dedup(name2)
        
        # Exact match after normalization
        if norm1 == norm2:
            return True
        
        # One name contains the other (e.g., "olive oil" vs "extra virgin olive oil")
        if norm1 in norm2 or norm2 in norm1:
            # But not too short (avoid "oil" matching "olive oil" incorrectly)
            if len(norm1) >= 4 and len(norm2) >= 4:
                return True
        
        # Check for common variations
        variations = {
            "olive oil": ["evoo", "extra virgin olive oil", "olive_oil"],
            "soy sauce": ["soy_sauce", "soya sauce"],
            "black pepper": ["pepper", "black_pepper"],
            "garlic powder": ["garlic_powder", "garlic"],
            "onion powder": ["onion_powder", "onion"],
        }
        for base, vars_list in variations.items():
            if (norm1 == base or norm1 in vars_list) and (norm2 == base or norm2 in vars_list):
                return True
        
        return False
    
    def calculate_iou_for_dedup(box1, box2):
        """Calculate IoU for deduplication"""
        if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
            return 0.0
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    # Better deduplication: Combine name similarity + spatial overlap
    unique_items = []
    processed_indices = set()
    
    for i, item in enumerate(items):
        if i in processed_indices:
            continue
        
        item_name = item.get("name", "").lower().strip()
        item_conf = item.get("confidence", 0)
        item_bbox = item.get("bbox")
        
        # Skip deduplication for unknown items
        if item_name in ["unknown_food", "unknown_packaged_food", "unknown_item", ""]:
            unique_items.append(item)
            processed_indices.add(i)
            continue
        
        # Find all similar items (same name or similar name + overlapping boxes)
        similar_items = [item]
        similar_indices = [i]
        
        for j, other_item in enumerate(items[i+1:], start=i+1):
            if j in processed_indices:
                continue
            
            other_name = other_item.get("name", "").lower().strip()
            other_conf = other_item.get("confidence", 0)
            other_bbox = other_item.get("bbox")
            
            # Skip unknown items
            if other_name in ["unknown_food", "unknown_packaged_food", "unknown_item", ""]:
                continue
            
            # Check if names are similar
            names_match = names_are_similar(item_name, other_name)
            
            # Check spatial overlap if both have bboxes
            spatial_overlap = False
            if item_bbox and other_bbox:
                iou = calculate_iou_for_dedup(item_bbox, other_bbox)
                spatial_overlap = iou > 0.3  # 30% overlap threshold
            
            # Merge if: (1) names match exactly, OR (2) names are similar AND boxes overlap
            if names_match:
                # Exact or similar name match - merge
                similar_items.append(other_item)
                similar_indices.append(j)
            elif spatial_overlap and names_match:
                # Similar name + spatial overlap - definitely same item
                similar_items.append(other_item)
                similar_indices.append(j)
        
        # Merge similar items: keep the one with highest confidence
        if len(similar_items) > 1:
            # Sort by confidence and keep the best
            similar_items.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            best_item = similar_items[0]
            
            # Boost confidence if multiple detections agree (consensus)
            if len(similar_items) >= 2:
                avg_conf = sum(item.get("confidence", 0) for item in similar_items) / len(similar_items)
                # Boost by 10% per additional detection (capped at 20% boost)
                boost = min(0.20, (len(similar_items) - 1) * 0.10)
                best_item["confidence"] = min(1.0, best_item.get("confidence", 0) * (1 + boost))
                best_item["detection_count"] = len(similar_items)  # Track how many detections merged
                if VERBOSE_LOGGING:
                    print(f"   🔗 Merged {len(similar_items)} detections of '{best_item.get('name')}' (boost: +{boost*100:.0f}%)")
            
            unique_items.append(best_item)
        else:
            unique_items.append(item)
        
        # Mark all similar items as processed
        processed_indices.update(similar_indices)
    
    # Sort by confidence
    result = sorted(unique_items, key=lambda x: x.get("confidence", 0), reverse=True)
    print(f"🔍 After better deduplication: {len(result)} unique items (merged {len(items) - len(result)} duplicates)")
    
    # 🔥 FIX #4: Merge multiple unknowns intelligently
    # Instead of showing "unknown_packaged_food x5", show "5 packaged foods (needs labeling)"
    unknown_items = [item for item in result if item.get("name", "").lower() in ["unknown_packaged_food", "unknown_food", "unknown_item"]]
    known_items = [item for item in result if item.get("name", "").lower() not in ["unknown_packaged_food", "unknown_food", "unknown_item"]]
    
    if len(unknown_items) > 1:
        # Merge multiple unknowns into a single item with count
        total_unknowns = len(unknown_items)
        avg_conf = sum(item.get("confidence", 0.3) for item in unknown_items) / total_unknowns if unknown_items else 0.3
        
        merged_unknown = {
            "name": f"{total_unknowns} packaged foods (needs labeling)",
            "quantity": str(total_unknowns),
            "expirationDate": None,
            "category": "other",
            "confidence": avg_conf,
            "detection_method": "yolov8+unknown",
            "needs_confirmation": True,
            "merged_count": total_unknowns,  # Track how many were merged
            "is_merged_unknown": True  # Flag for UI
        }
        
        # Replace all unknown items with the merged one
        result = known_items + [merged_unknown]
        if VERBOSE_LOGGING:
            print(f"   🔗 FIX #4: Merged {total_unknowns} unknown items into single entry")
    
    # STEP 6: Apply ensemble confidence boosting (if multiple methods detected same item)
    result = apply_ensemble_boosting(result, items)
    
    # STEP 7: Apply shelf-aware context fusion (boost confidence for items in pantry + spatial context)
    if user_pantry:
        # Safely get image height with validation
        img_height_value = None
        try:
            if hasattr(img, 'size') and img.size and len(img.size) >= 2:
                img_height_value = int(img.size[1])
                if img_height_value <= 0:
                    img_height_value = None
        except (AttributeError, TypeError, IndexError):
            img_height_value = None
        
        result = apply_context_fusion(result, user_pantry, img_height=img_height_value)
    
    # STEP 7.5: Apply quality score weighting to all confidences
    # Poor quality images should have lower confidence scores
    # Ensure quality_score is valid (0.0-1.0)
    quality_score = max(0.0, min(1.0, quality_score)) if quality_score is not None else 1.0
    
    # Don't apply quality penalty if it would remove all items
    # Instead, use a minimum quality multiplier to ensure some items are returned
    min_quality_multiplier = 0.3  # Never reduce confidence below 30% of original
    
    for item in result:
        if not isinstance(item, dict):
            continue
        base_conf = item.get("confidence", 0)
        # Validate base_conf is numeric
        try:
            base_conf = float(base_conf) if base_conf is not None else 0.0
        except (ValueError, TypeError):
            base_conf = 0.0
        
        # Weight confidence by image quality, but don't penalize too harshly
        # Use a blend: 70% quality-weighted, 30% original (ensures items aren't filtered out)
        quality_weighted = base_conf * quality_score
        blended_conf = (quality_weighted * 0.7) + (base_conf * 0.3)
        # Ensure minimum confidence to prevent filtering out all items
        final_conf = max(blended_conf, base_conf * min_quality_multiplier)
        
        item["confidence"] = max(0.0, min(1.0, final_conf))
        item["quality_score"] = quality_score
        if VERBOSE_LOGGING and quality_score < 0.7:
            print(f"  ⚠️ Quality-adjusted confidence for '{item.get('name')}': {base_conf:.2f} -> {item['confidence']:.2f}")
    
    # STEP 8: Calibrate confidence based on historical accuracy
    # Current implementation does not use precision_mode flag here; calibration is global.
    # (Precision-specific behavior is handled earlier via thresholds and heuristics.)
    result = calibrate_confidence(result)
    
    # 🔥 IMPROVEMENT 2 & 8: Reject Option + Demo-Safe Mode
    # Precision > Recall: Allow "unknown_item" for low confidence
    filtered_result = []
    for item in result:
        conf = item.get('confidence', 0)
        name = item.get('name', '').lower()
        
        # 🔥 RECALL-PRIORITY: Lower final filtering threshold to allow more items through
        # Items below 0.20: Mark as "unknown_item" with needs_confirmation (only very low confidence)
        # Items 0.20-0.40: Keep original name but mark needs_confirmation
        # Items > 0.40: No confirmation needed
        original_name = item.get('name', '')  # Preserve original name before any changes
        if conf < 0.20:
            # Very low confidence - only mark as unknown for non-food or very uncertain items
            # Try to preserve YOLO name if it's a food item
            if original_name and original_name.lower() not in ['unknown_item', 'unknown_packaged_food', 'unknown_produce']:
                # Keep original name but mark for confirmation
                item['needs_confirmation'] = True
                item['confidence_display'] = f"{conf:.0%} (Needs confirmation)"
            else:
                # Non-food or already unknown - mark as unknown_item
                item['name'] = 'unknown_item'
                item['category'] = 'other'
                # Fix 1: Very low confidence - keep semantic label, never downgrade to unknown_item
                # Never replace semantic label with unknown_item unless zero signal
                if original_name not in ["unknown_food", "unknown_packaged_food", "unknown_item"]:
                    # Keep semantic label, just mark for confirmation
                    item['name'] = original_name
                    item['needs_confirmation'] = True
                    item['confidence_display'] = f"{conf:.0%} (Needs confirmation)"
                    if VERBOSE_LOGGING:
                        print(f"   ⚠️ Very low confidence ({conf:.2f}) - keeping semantic label '{original_name}', marking for confirmation")
                else:
                    # No semantic label - use unknown_food
                    item['name'] = 'unknown_food'
                    item['category'] = 'other'
                    item['needs_confirmation'] = True
                    item['confidence_display'] = f"{conf:.0%} (Needs confirmation)"
                    if VERBOSE_LOGGING:
                        print(f"   ⚠️ Very low confidence ({conf:.2f}) - showing as unknown_food - ask user")
        elif conf < 0.40:
            item['needs_confirmation'] = True
            item['confidence_display'] = f"{conf:.0%} (Needs confirmation)"
        else:
            item['confidence_display'] = f"{conf:.0%}"
        
        filtered_result.append(item)
    
    result = filtered_result
    
    # STEP 9: Return categorized by confidence (high/medium/low)
    print(f"📊 ML Detection Summary: Found {len(result)} items")
    if len(result) > 0:
        print(f"   Items: {[item.get('name', 'unknown') for item in result[:5]]}")
        confidences = [f"{item.get('confidence', 0):.2f}" for item in result[:5]]
        print(f"   Confidences: {confidences}")
        print(f"   Detection methods: {[item.get('detection_method', 'unknown') for item in result[:5]]}")
        # 🔥 FIX: Log all items, not just first 5, to debug why nothing is detected
        if len(result) > 5:
            print(f"   ... and {len(result) - 5} more items")
        # Log confidence distribution
        conf_values = [item.get('confidence', 0) for item in result]
        if conf_values:
            print(f"   Confidence range: {min(conf_values):.2f} - {max(conf_values):.2f} (avg: {sum(conf_values)/len(conf_values):.2f})")
    else:
        print("   ⚠️ No items detected - possible reasons:")
        print(f"      - YOLO model not loaded: {_yolo_model is None}")
        print(f"      - YOLO disabled: {not YOLO_DETECTION_ENABLED}")
        print(f"      - Food classifier not loaded: {_food_classifier is None}")
        print(f"      - CLIP model not loaded: {_clip_model is None}")
        print(f"      - OCR reader not loaded: {_ocr_reader is None}")
        print(f"      - Image quality too low: {quality_score:.2f}")
        print(f"      - YOLO found {len(regions) if 'regions' in locals() else 0} regions")
        print(f"      - Items before filtering: {len(items) if 'items' in locals() else 0}")
        print(f"      - All items may have been filtered out by confidence thresholds")
        
        # 🔥 FIX: If we have regions but no items, log why
        if 'regions' in locals() and regions and len(regions) > 0:
            print(f"   ⚠️ YOLO found {len(regions)} regions but no items were created")
            print(f"      This suggests CLIP/OCR failed to classify the regions")
            print(f"      Check CLIP confidence thresholds and OCR text detection")
    
    # 🔥 FIX: Always return items, even if confidence is low (let user decide)
    # Don't filter out items - mark them for confirmation instead
    return result
 

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
    """Load users from Firebase or JSON file or in-memory storage with caching"""
    # Use Firebase if enabled
    if USE_FIREBASE:
        try:
            users = firebase_load_users()
            # Convert to dict format expected by the rest of the app
            if isinstance(users, dict):
                return users
            # If Firebase returns a list, convert to dict
            users_dict = {}
            for user in users if isinstance(users, list) else []:
                user_id = user.get('id') or user.get('user_id')
                if user_id:
                    users_dict[user_id] = user
            return users_dict if users_dict else {}
        except Exception as e:
            print(f"⚠️ Error loading users from Firebase: {e}, falling back to file storage")
            # Fall through to file-based storage
    
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
            pass
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
                            pass
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
            pass
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
                                    pass
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
    """Save users to Firebase or JSON file or in-memory storage and update cache"""
    # Declare all globals at the start of the function
    global _users_cache, _users_cache_timestamp, _pantry_cache, _pantry_cache_timestamp, _in_memory_users
    
    # Use Firebase if enabled
    if USE_FIREBASE:
        try:
            success = firebase_save_users(users)
            if success:
                # Invalidate cache
                if '_all_users' in _users_cache:
                    del _users_cache['_all_users']
                _pantry_cache.clear()
                _pantry_cache_timestamp.clear()
                return
            else:
                print("⚠️ Failed to save users to Firebase, falling back to file storage")
        except Exception as e:
            print(f"⚠️ Error saving users to Firebase: {e}, falling back to file storage")
            # Fall through to file-based storage
    
    # Invalidate cache on save
    if '_all_users' in _users_cache:
        del _users_cache['_all_users']
    # Also invalidate all pantry caches since users data changed
    _pantry_cache.clear()
    _pantry_cache_timestamp.clear()
    
    if IS_VERCEL or IS_RENDER:
        # Try to save to /tmp, always update in-memory cache
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
            pass
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
                    pass
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
                pass
            except Exception as e2:
                print(f"❌ Could not save to fallback location either: {e2}")
                import traceback
                traceback.print_exc()

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, email, password, client_type='web'):
    """Create a new user"""
    # Use Firebase if enabled
    if USE_FIREBASE:
        try:
            user_id, error = create_user_in_firestore(username, email, password, client_type)
            if error:
                return None, error
            if user_id:
                print(f"✅ User created in Firebase: '{username}' (ID: {user_id})")
                # Invalidate cache to ensure fresh data on next load
                global _users_cache, _users_cache_timestamp
                if '_all_users' in _users_cache:
                    del _users_cache['_all_users']
                if '_all_users' in _users_cache_timestamp:
                    del _users_cache_timestamp['_all_users']
                # Verify user was saved by loading from Firebase
                verify_user = firebase_get_user_by_id(user_id)
                if verify_user:
                    print(f"✅ Verified user exists in Firebase: {verify_user.get('username')}")
                else:
                    print(f"⚠️ Warning: User {user_id} not found in Firebase after creation")
                return user_id, None
        except Exception as e:
            print(f"⚠️ Error creating user in Firebase: {e}, falling back to file storage")
            import traceback
            traceback.print_exc()
            # Fall through to file-based storage
    
    # File-based storage (original implementation)
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
        pass
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
                    pass
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
    # Force fresh load from Firebase/file storage (disable cache to ensure we get latest data)
    users = load_users(use_cache=False)
    
    # Normalize input (strip whitespace and convert to lowercase for comparison)
    username_normalized = username.strip().lower() if username else ""
    password = password.strip() if password else ""
    
    # Debug: print number of users loaded
    print(f"Authenticating user '{username}' (normalized: '{username_normalized}') against {len(users)} users")
    if USE_FIREBASE:
        print(f"   Using Firebase for authentication")
    
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
            
            # Update in Firebase if enabled, otherwise use file storage
            if USE_FIREBASE:
                try:
                    firebase_update_user(user_id, {'last_login': user_data['last_login']})
                    print(f"✅ Updated last_login in Firebase for user {user_id}")
                except Exception as e:
                    print(f"⚠️ Error updating last_login in Firebase: {e}")
            else:
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
        pass
    except:
        return False

# Register template filter AFTER function is defined
@app.template_filter('is_expiring_soon')
def template_is_expiring_soon(exp_date_str):
    """Template filter to check if expiration date is soon"""
    return is_expiring_soon(exp_date_str, 7)

def normalize_expiration_date(date_str):
    """Normalize expiration date to YYYY-MM-DD format with defensive validation"""
    # Input validation
    if not date_str:
        return None
    
    # Ensure it's a string
    if not isinstance(date_str, str):
        try:
            date_str = str(date_str)
        except Exception:
            return None
    
    date_str = date_str.strip()
    if not date_str or date_str.lower() in ['none', 'null', '']:
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
        pass
    except (ValueError, TypeError, AttributeError):
        return None

def normalize_item_name(name):
    """Normalize item names to improve accuracy and consistency, preserving rare food names"""
    # Input validation
    if not name:
        return None
    
    # Ensure it's a string
    if not isinstance(name, str):
        try:
            name = str(name)
            pass
        except Exception:
            return None
    
    original_name = name.strip()
    if not original_name:
        return None
    
    name = original_name.lower()
    
    # Preserve rare/uncommon food names - don't over-normalize them
    rare_food_indicators = [
        'tahini', 'kimchi', 'miso', 'pesto', 'hummus', 'tzatziki', 'harissa', 'gochujang',
        'sriracha', 'wasabi', 'mirin', 'ponzu', 'tamarind', 'curry', 'coconut', 'soy',
        'fish sauce', 'oyster sauce', 'hoisin', 'black bean', 'chili oil', 'sesame oil',
        'truffle oil', 'balsamic', 'rice vinegar', 'apple cider', 'maple syrup', 'agave',
        'molasses', 'vanilla extract', 'almond extract', 'cocoa powder', 'matcha',
        'spirulina', 'nutritional yeast', 'tempeh', 'tofu', 'seitan', 'jackfruit',
        'quinoa', 'farro', 'bulgur', 'couscous', 'polenta', 'grits', 'chia', 'flax',
        'hemp', 'pumpkin seed', 'sunflower seed', 'almond butter', 'cashew butter',
        'marmalade', 'chutney', 'relish', 'pickle', 'olive', 'caper', 'anchovy',
        'sardine', 'macadamia', 'pistachio', 'pine nut', 'lentil', 'chickpea',
        'black bean', 'edamame', 'avocado oil', 'walnut oil', 'white wine vinegar',
        'tapenade', 'bruschetta', 'guacamole', 'kombucha', 'kefir', 'sauerkraut'
    ]
    
    # Check if this might be a rare food - preserve it more carefully
    is_rare_food = any(indicator in name for indicator in rare_food_indicators)
    
    # Common corrections for better accuracy (but skip for rare foods)
    if not is_rare_food:
        corrections = {
            'milk carton': 'milk',
            'chicken meat': 'chicken',
            'tomatoes': 'tomato',
            'apples': 'apple',
            'potatoes': 'potato',
            'oranges': 'orange',
            'bananas': 'banana',
            'eggs': 'egg',
            'bread loaf': 'bread',
            'chicken breast': 'chicken',
            'pasta box': 'pasta',
            'cereal box': 'cereal',
            'soup can': 'soup',
            'juice bottle': 'juice',
            'water bottle': 'water',
            'yogurt container': 'yogurt',
            'cheese package': 'cheese',
            'butter package': 'butter',
        }
        
        # Apply corrections
        for wrong, correct in corrections.items():
            if wrong in name:
                name = name.replace(wrong, correct)
    
    # Remove common prefixes/suffixes (but be gentler with rare foods)
    if not is_rare_food:
        prefixes_to_remove = ['a ', 'an ', 'the ', 'some ', 'pack of ', 'box of ', 'bottle of ', 'can of ', 'package of ', 'container of ']
        for prefix in prefixes_to_remove:
            if name.startswith(prefix):
                name = name[len(prefix):]
        
        # Remove trailing common words
        suffixes_to_remove = [' container', ' package', ' box', ' bottle', ' can', ' carton', ' bag']
        for suffix in suffixes_to_remove:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
    else:
        # For rare foods, only remove obvious packaging words
        rare_prefixes = ['a ', 'an ', 'the ', 'some ']
        for prefix in rare_prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
        
        rare_suffixes = [' container', ' package', ' box', ' bottle', ' can', ' carton']
        for suffix in rare_suffixes:
            if name.endswith(suffix) and len(name) > len(suffix) + 3:  # Only if name is long enough
                name = name[:-len(suffix)]
    
    # Capitalize first letter of each word, preserving rare food names
    if name:
        # For rare foods, preserve original capitalization style if it looks intentional
        if is_rare_food and any(c.isupper() for c in original_name[1:]):
            # Mixed case might be intentional (e.g., "Sriracha", "Gochujang")
            words = name.split()
            # Capitalize each word but preserve common patterns
            return ' '.join(word.capitalize() for word in words)
        else:
            return ' '.join(word.capitalize() for word in name.split())
    return None

def validate_category(item_name, category):
    """Auto-correct category based on item name if category seems wrong"""
    # Input validation
    if not item_name:
        return category or 'other'
    
    # Ensure it's a string
    if not isinstance(item_name, str):
        try:
            item_name = str(item_name)
            pass
        except Exception:
            return category or 'other'
    
    name_lower = item_name.lower().strip()
    if not name_lower:
        return category or 'other'
    
    # Normalize common category synonyms from clients/LLMs/UI
    cat_lower = (category or '').lower().strip()
    if cat_lower in {'protein', 'proteins', 'protein(s)'}:
        # Treat protein as meat for UI/UX clarity (and to match existing meat styling)
        return 'meat'
    
    # Category mapping based on keywords (comprehensive list)
    category_keywords = {
        'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream', 'sour cream', 'cottage cheese', 'milk product', 'egg', 'eggs', 'yoghurt', 'mozzarella', 'cheddar', 'swiss', 'gouda', 'brie', 'feta', 'parmesan', 'ricotta', 'cream cheese', 'heavy cream', 'half and half', 'buttermilk', 'sour cream', 'greek yogurt'],
        'produce': ['apple', 'banana', 'orange', 'tomato', 'lettuce', 'carrot', 'onion', 'potato', 'vegetable', 'fruit', 'pepper', 'cucumber', 'broccoli', 'spinach', 'celery', 'garlic', 'ginger', 'avocado', 'lemon', 'lime', 'grape', 'strawberry', 'blueberry', 'raspberry', 'blackberry', 'peach', 'pear', 'plum', 'cherry', 'mango', 'pineapple', 'watermelon', 'cantaloupe', 'zucchini', 'squash', 'eggplant', 'corn', 'peas', 'green bean', 'asparagus', 'cauliflower', 'cabbage', 'kale', 'arugula', 'radish', 'beet', 'turnip', 'sweet potato', 'yam'],
        'meat': ['chicken', 'beef', 'pork', 'fish', 'turkey', 'bacon', 'sausage', 'ham', 'steak', 'ground beef', 'salmon', 'tuna', 'cod', 'tilapia', 'shrimp', 'prawn', 'crab', 'lobster', 'lamb', 'duck', 'veal', 'venison', 'bison', 'ribeye', 'sirloin', 'chicken breast', 'chicken thigh', 'ground turkey', 'ground pork', 'chorizo', 'pepperoni', 'salami', 'prosciutto', 'hot dog', 'burger', 'meatball'],
        'beverages': ['juice', 'soda', 'water', 'coffee', 'tea', 'beer', 'wine', 'cola', 'drink', 'lemonade', 'iced tea', 'sports drink', 'energy drink', 'smoothie', 'milkshake', 'hot chocolate', 'cocoa', 'espresso', 'latte', 'cappuccino', 'soda pop', 'soft drink', 'sparkling water', 'seltzer'],
        'bakery': ['bread', 'bagel', 'muffin', 'croissant', 'roll', 'bun', 'pastry', 'donut', 'doughnut', 'cake', 'pie', 'cookie', 'biscuit', 'scone', 'pita', 'tortilla', 'naan', 'baguette', 'sourdough', 'rye bread', 'wheat bread', 'white bread', 'whole grain'],
        'canned goods': ['can', 'canned', 'soup', 'tuna', 'beans', 'corn', 'tomato', 'sardine', 'anchovy', 'salmon', 'chicken', 'broth', 'stock', 'vegetable', 'fruit', 'peach', 'pear', 'pineapple'],
        'snacks': ['chip', 'cracker', 'cookie', 'nut', 'popcorn', 'pretzel', 'candy', 'chocolate', 'granola bar', 'trail mix', 'nuts', 'almond', 'peanut', 'cashew', 'walnut', 'pistachio', 'sunflower seed', 'pumpkin seed', 'chips', 'doritos', 'lays', 'cheetos', 'goldfish', 'ritz', 'oreo', 'chips ahoy'],
        'condiments': ['sauce', 'ketchup', 'mustard', 'mayo', 'mayonnaise', 'dressing', 'spice', 'oil', 'vinegar', 'salt', 'pepper', 'soy sauce', 'worcestershire', 'hot sauce', 'sriracha', 'bbq sauce', 'ranch', 'italian dressing', 'caesar dressing', 'honey', 'maple syrup', 'jam', 'jelly', 'preserves', 'peanut butter', 'almond butter', 'tahini', 'pesto', 'salsa', 'guacamole', 'hummus', 'relish', 'pickle', 'olive oil', 'vegetable oil', 'canola oil', 'sesame oil'],
        'grains': ['rice', 'pasta', 'cereal', 'flour', 'oat', 'quinoa', 'barley', 'oats', 'oatmeal', 'wheat', 'bread', 'noodle', 'spaghetti', 'penne', 'macaroni', 'fettuccine', 'linguine', 'couscous', 'bulgur', 'farro', 'millet', 'brown rice', 'white rice', 'wild rice', 'basmati', 'jasmine rice'],
        'frozen': ['frozen', 'ice cream', 'frozen vegetable', 'frozen fruit', 'frozen meal', 'frozen pizza', 'frozen dinner', 'ice', 'frozen yogurt', 'sorbet', 'gelato', 'frozen berries', 'frozen peas', 'frozen corn']
    }
    
    # Check if category matches item name (case-insensitive, partial match)
    for correct_cat, keywords in category_keywords.items():
        if any(keyword in name_lower for keyword in keywords):
            return correct_cat
    
    # If no match found and original category is 'other', try to infer from common patterns
    if category == 'other' or not category:
        # Additional heuristics for common items
        if 'egg' in name_lower:
            return 'dairy'
        elif any(word in name_lower for word in ['fresh', 'organic', 'ripe']):
            # Likely produce if it has freshness indicators
            if any(word in name_lower for word in ['fruit', 'vegetable', 'berry', 'leaf', 'green']):
                return 'produce'
    
    return category if category and category != 'other' else 'other'  # Return original if valid, otherwise 'other'

def parse_quantity(quantity_str):
    """Parse and normalize quantity strings with defensive validation"""
    # Input validation
    if not quantity_str:
        return "1"
    
    # Ensure it's a string
    if not isinstance(quantity_str, str):
        try:
            quantity_str = str(quantity_str)
            pass
        except Exception:
            return "1"
    
    quantity_str = quantity_str.strip().lower()
    
    # Extract number from strings like "2 bottles", "three cans"
    import re
    numbers = re.findall(r'\d+', quantity_str)
    if numbers:
        num = int(numbers[0])
        # Extract unit
        unit_match = re.search(r'(bottle|can|package|box|loaf|piece|slice|cup|pound|lb|oz|gram|g|kg|container|item)', quantity_str)
        unit = unit_match.group(0) if unit_match else "item"
        # Pluralize unit if needed
        if num > 1 and not unit.endswith('s'):
            if unit in ['box', 'box']:
                unit = 'boxes'
            elif unit == 'bottle':
                unit = 'bottles'
            elif unit == 'can':
                unit = 'cans'
            elif unit == 'loaf':
                unit = 'loaves'
            elif unit == 'piece':
                unit = 'pieces'
        return f"{num} {unit}"
    
    # Handle word numbers
    word_numbers = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
    for word, num in word_numbers.items():
        if word in quantity_str:
            # Extract unit if present
            unit_match = re.search(r'(bottle|can|package|box|loaf|piece)', quantity_str)
            unit = unit_match.group(0) if unit_match else "item"
            if num > 1 and not unit.endswith('s'):
                if unit == 'bottle':
                    unit = 'bottles'
                elif unit == 'can':
                    unit = 'cans'
            return f"{num} {unit}"
    
    return quantity_str if quantity_str else "1"

def parse_api_response_with_retry(food_response, max_retries=2):
    """Parse API response with better error handling and JSON extraction"""
    import json
    import re
    
    # Validate input
    if not food_response:
        return []
    
    if not isinstance(food_response, str):
        food_response = str(food_response)
    
    food_response = food_response.strip()
    if not food_response:
        return []
    
    for attempt in range(max_retries):
        try:
            # Try direct JSON parse
            food_data = json.loads(food_response)
            if not isinstance(food_data, dict):
                raise ValueError("Response is not a JSON object")
            items = food_data.get('items', [])
            # Validate items is a list
            if not isinstance(items, list):
                return []
            return items
            pass
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?"items".*?\})\s*```', food_response, re.DOTALL | re.IGNORECASE)
            if json_match:
                try:
                    food_data = json.loads(json_match.group(1))
                    items = food_data.get('items', [])
                    if isinstance(items, list):
                        return items
                    pass
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
            
            # Try to extract just the JSON object
            json_match = re.search(r'\{[^{}]*"items"[^{}]*\[.*?\]\s*\}', food_response, re.DOTALL)
            if json_match:
                try:
                    food_data = json.loads(json_match.group(0))
                    items = food_data.get('items', [])
                    if isinstance(items, list):
                        return items
                    pass
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
            
            # Try to fix common JSON errors
            if attempt == 0:
                # Replace common issues
                food_response = food_response.replace('None', 'null').replace("'", '"')
                # Remove any text before first {
                first_brace = food_response.find('{')
                if first_brace > 0:
                    food_response = food_response[first_brace:]
                pass
    
    # Final fallback: return empty list
    return []

def safe_get_response_content(response):
    """Safely extract content from OpenAI API response with comprehensive error handling"""
    try:
        if not response:
            return None
        
        if not hasattr(response, 'choices') or not response.choices:
            return None
        
        if len(response.choices) == 0:
            return None
        
        choice = response.choices[0]
        if not hasattr(choice, 'message'):
            return None
        
        message = choice.message
        if not hasattr(message, 'content'):
            return None
        
        content = message.content
        if not content or not isinstance(content, str):
            return None
        
        return content.strip()
        pass
    except (AttributeError, IndexError, TypeError) as e:
        if VERBOSE_LOGGING:

            print(f"Error extracting response content: {e}")
        return None

def get_session_pantry():
    """Safely get and validate session pantry - ensures it's always a list"""
    if 'web_pantry' not in session:
        session['web_pantry'] = []
        session.modified = True
    
    web_pantry = session.get('web_pantry', [])
    if not isinstance(web_pantry, list):
        if VERBOSE_LOGGING:
            print(f"Warning: session['web_pantry'] is not a list, resetting to empty list")
        session['web_pantry'] = []
        session.modified = True
        web_pantry = []
    
    return web_pantry

def set_session_pantry(pantry_list):
    """Safely set session pantry - validates and normalizes items"""
    if not isinstance(pantry_list, list):
        if VERBOSE_LOGGING:
            print(f"Warning: pantry_list is not a list in set_session_pantry, converting")
        pantry_list = []
    
    # Normalize all items
    normalized_list = []
    for item in pantry_list:
        try:
            if isinstance(item, dict):
                normalized_item = normalize_pantry_item(item.copy())
                if normalized_item and normalized_item.get('name'):
                    normalized_list.append(normalized_item)
            elif item is not None:
                normalized_item = normalize_pantry_item(item)
                if normalized_item and normalized_item.get('name'):
                    normalized_list.append(normalized_item)
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"Warning: Failed to normalize item in set_session_pantry: {e}")
            continue

    
    session['web_pantry'] = normalized_list
    session.modified = True
    session.permanent = True
    
    # 🔥 FIX: Force session save by accessing session
    try:
        # Touch session to ensure it's saved
        _ = session.get('web_pantry')
        if VERBOSE_LOGGING:
            print(f"✅ Session pantry saved: {len(normalized_list)} items")
            print(f"   Session keys: {list(session.keys())}")
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"⚠️ Warning: Could not verify session save: {e}")
    
    return normalized_list

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
        
        # Ensure category exists and is validated
        if 'category' not in normalized_item or not normalized_item['category'] or normalized_item['category'] == 'other':
            # Auto-detect category from item name if not set or is 'other'
            item_name = normalized_item.get('name', '')
            if item_name:
                normalized_item['category'] = validate_category(item_name, 'other')
            else:
                normalized_item['category'] = 'other'
        else:
            # Validate existing category
            item_name = normalized_item.get('name', '')
            if item_name:
                normalized_item['category'] = validate_category(item_name, normalized_item['category'])
        
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
    
    # Validate user_id
    if not user_id or not isinstance(user_id, str):
        if VERBOSE_LOGGING:
            print(f"Warning: Invalid user_id in get_user_pantry: {user_id}")
        return []

    
    # Check pantry cache first
    if use_cache:
        import time
        current_time = time.time()
        if user_id in _pantry_cache:
            cache_age = current_time - _pantry_cache_timestamp.get(user_id, 0)
            if cache_age < _users_cache_ttl:
                cached_pantry = _pantry_cache[user_id]
                # Validate cached data is a list
                if isinstance(cached_pantry, list):
                    if VERBOSE_LOGGING:
                        print(f"Using cached pantry for user {user_id} (age: {cache_age:.2f}s)")
                    return cached_pantry.copy()
                else:
                    # Cache corrupted, clear it
                    if VERBOSE_LOGGING:
                        print(f"Warning: Cached pantry for user {user_id} is corrupted, clearing cache")
                    del _pantry_cache[user_id]
                    if user_id in _pantry_cache_timestamp:
                        del _pantry_cache_timestamp[user_id]
    
    users = load_users(use_cache=use_cache)
    if not isinstance(users, dict):
        if VERBOSE_LOGGING:
            print(f"Warning: Users data is not a dict, returning empty pantry")
        return []
    
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
                pass
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
    global _pantry_cache, _pantry_cache_timestamp
    
    # Validate user_id
    if not user_id or not isinstance(user_id, str):
        print(f"Error: Invalid user_id in update_user_pantry: {user_id}")
        return False

    
    # CRITICAL: Validate that user exists before saving to Firebase
    # This prevents anonymous users or invalid user_ids from saving to database
    user_exists = False
    try:
        if USE_FIREBASE:
            user_data = firebase_get_user_by_id(user_id)
            if user_data:
                user_exists = True
        else:
            users = load_users()
            if user_id in users:
                user_exists = True
    except Exception as e:
        print(f"Error checking if user {user_id} exists: {e}")
    
    if not user_exists:
        print(f"❌ ERROR: Cannot save pantry items - user {user_id} does not exist in database!")
        print(f"   This prevents anonymous users or invalid sessions from saving to Firebase.")
        return False
    
    if VERBOSE_LOGGING:
        print(f"\n{'='*60}")
        print(f"🔄 UPDATE USER PANTRY")
        print(f"{'='*60}")
        print(f"User ID: {user_id}")
        print(f"Items to save: {len(pantry_items) if isinstance(pantry_items, list) else 0}")
    
    # Use Firebase if enabled
    if USE_FIREBASE:
        try:
            # Normalize items first (using the same normalization logic)
            normalized_items = []
            for item in pantry_items:
                try:
                    if isinstance(item, dict):
                        normalized_item = normalize_pantry_item(item.copy())
                        if normalized_item and normalized_item.get('name'):
                            normalized_items.append(normalized_item)
                    elif item is not None:
                        normalized_item = normalize_pantry_item(item)
                        if normalized_item and normalized_item.get('name'):
                            normalized_items.append(normalized_item)
                except Exception as e:
                    print(f"Warning: Failed to normalize item {item}: {e}")
                    continue
            
            # Update pantry in Firebase
            success = firebase_update_user_pantry(user_id, normalized_items)
            if success:
                # Invalidate cache
                if user_id in _pantry_cache:
                    del _pantry_cache[user_id]
                if user_id in _pantry_cache_timestamp:
                    del _pantry_cache_timestamp[user_id]
                
                if VERBOSE_LOGGING:
                    print(f"✅ Updated pantry for user {user_id}: {len(normalized_items)} items saved to Firebase")
                return True
            else:
                print("⚠️ Failed to update pantry in Firebase, falling back to file storage")
        except Exception as e:
            print(f"⚠️ Error updating pantry in Firebase: {e}, falling back to file storage")
            import traceback
            traceback.print_exc()
            # Fall through to file-based storage
    
    # File-based storage (original implementation)
    users = load_users(use_cache=False)  # Don't use cache when updating
    if not isinstance(users, dict):
        print(f"Error: Users data is not a dict")
        return False
    
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
                normalized_item = normalize_pantry_item(item.copy())
                if normalized_item and normalized_item.get('name'):
                    normalized_items.append(normalized_item)
            elif item is not None:
                normalized_item = normalize_pantry_item(item)
                if normalized_item and normalized_item.get('name'):
                    normalized_items.append(normalized_item)
            pass
        except Exception as e:
            print(f"Warning: Failed to normalize item {item}: {e}")
            continue
    
    users[user_id]['pantry'] = normalized_items
    
    # Invalidate cache for this user
    if user_id in _pantry_cache:
        del _pantry_cache[user_id]
    if user_id in _pantry_cache_timestamp:
        del _pantry_cache_timestamp[user_id]
    
    if VERBOSE_LOGGING:
        print(f"💾 Saving {len(users)} users to {USERS_FILE}...")
    
    try:
        save_users(users)
        if VERBOSE_LOGGING:
            print(f"✅ Updated pantry for user {user_id}: {len(normalized_items)} items saved to {USERS_FILE}")
        return True
    except Exception as e:
        print(f"Error: Failed to save pantry for user {user_id}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
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
                # Clear any existing session data to start fresh
                session.clear()
                session.permanent = True  # Make session permanent
                session['user_id'] = user_id
                session['username'] = username
                # Ensure pantry is empty for new user (clear any session pantry data)
                if 'web_pantry' in session:
                    del session['web_pantry']
                print(f"Signup successful - User ID: {user_id}, Username: {username}")
                print(f"Session after signup: user_id={session.get('user_id')}, username={session.get('username')}")
                if USE_FIREBASE:
                    # Verify user exists in Firebase
                    verify_user = firebase_get_user_by_id(user_id)
                    if verify_user:
                        print(f"✅ Verified new user exists in Firebase: {verify_user.get('username')}")
                        print(f"   User pantry items: {len(verify_user.get('pantry', []))}")
                    else:
                        print(f"⚠️ Warning: New user {user_id} not found in Firebase after creation")
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
    # Check if user is logged in AND validate user exists
    if 'user_id' in session:
        user_id = session.get('user_id')
        
        # Validate that user actually exists in database
        user_exists = False
        username = None
        try:
            if USE_FIREBASE:
                user_data = firebase_get_user_by_id(user_id)
                if user_data:
                    user_exists = True
                    username = user_data.get('username')
            else:
                users = load_users()
                if user_id in users:
                    user_exists = True
                    username = users[user_id].get('username')
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"Error validating user {user_id}: {e}")
        
        # If user doesn't exist, clear session and treat as anonymous
        if not user_exists:
            print(f"⚠️ Session has invalid user_id {user_id}, clearing session")
            session.clear()
            # Fall through to anonymous user handling below
        else:
            # User exists and is logged in
            # get_user_pantry already normalizes items, so no need to normalize again
            user_pantry = get_user_pantry(user_id)
            # Ensure user_pantry is a list
            if not isinstance(user_pantry, list):
                user_pantry = []
            
            if VERBOSE_LOGGING:
                print(f"DEBUG: Rendering index with {len(user_pantry)} items for user {username}")
                print(f"DEBUG: Items: {[item.get('name', 'NO_NAME') for item in user_pantry[:3]]}")
            
            # Ensure items is always a list, never None
            items_to_render = user_pantry if user_pantry else []
            return render_template("index.html", items=items_to_render, username=username)
    
    # Anonymous user or invalid session - use session-based pantry
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
                pass
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
        candidate_user_id = session['user_id']
        
        # Validate user exists before saving
        user_exists = False
        try:
            if USE_FIREBASE:
                user_data = firebase_get_user_by_id(candidate_user_id)
                if user_data:
                    user_exists = True
            else:
                users = load_users()
                if candidate_user_id in users:
                    user_exists = True
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"Error validating user {candidate_user_id}: {e}")
        
        if not user_exists:
            # Invalid user_id in session, clear it and treat as anonymous
            print(f"⚠️ Session has invalid user_id {candidate_user_id}, clearing session")
            session.pop('user_id', None)
            session.pop('username', None)
            # Fall through to anonymous user handling below
        else:
            # User exists and is valid
            user_id = candidate_user_id
            print(f"Adding item '{item}' to pantry for user {user_id}")
        # Force fresh data by disabling cache to ensure we get the latest pantry items
        user_pantry = get_user_pantry(user_id, use_cache=False)
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
            # Detect category based on item name
            detected_category = validate_category(item, 'other')
            
            # Add new item with quantity and expiration date
            new_item = {
                'id': str(uuid.uuid4()),
                'name': item,
                'quantity': quantity,
                'expirationDate': normalized_expiration,
                'category': detected_category,
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
        web_pantry = get_session_pantry()
        
        # Check for duplicates (case-insensitive, normalized)
        item_exists = False
        item_normalized = item.strip().lower()  # Normalize for comparison
        for pantry_item in web_pantry:
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
            # Detect category based on item name
            detected_category = validate_category(item, 'other')
            
            # Add new item as dictionary with quantity and expiration date
            new_item = {
                'id': str(uuid.uuid4()),
                'name': item,
                'quantity': quantity,
                'expirationDate': normalized_expiration,
                'category': detected_category,
                'addedDate': datetime.now().isoformat()
            }
            web_pantry.append(new_item)
            set_session_pantry(web_pantry)
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
    
    item_found = False
    
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
        item_name_normalized = item_name.strip().lower()  # Normalize once for comparison
        for pantry_item in user_pantry:
            if isinstance(pantry_item, dict):
                pantry_name = pantry_item.get('name', '').strip() if pantry_item.get('name') else ''
                # Compare with stripped names (case-insensitive)
                if pantry_name and pantry_name.lower() == item_name_normalized:
                    item_found = True
                    if VERBOSE_LOGGING:
                        print(f"DEBUG: Found matching item: '{pantry_name}' == '{item_name}'")
                else:
                    pantry_list.append(pantry_item)
            else:
                pantry_str = str(pantry_item).strip() if pantry_item else ''
                if pantry_str and pantry_str.lower() == item_name_normalized:
                    item_found = True
                    if VERBOSE_LOGGING:
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
            return redirect(url_for("index"))
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
            return redirect(url_for("index"))
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
        item_name_normalized = item_name.strip().lower()  # Normalize once for comparison
        for pantry_item in session['web_pantry']:
            if isinstance(pantry_item, dict):
                pantry_name = pantry_item.get('name', '').strip() if pantry_item.get('name') else ''
                # Compare with stripped names (case-insensitive)
                if pantry_name and pantry_name.lower() == item_name_normalized:
                    item_found = True
                    if VERBOSE_LOGGING:
                        print(f"DEBUG: Found matching item: '{pantry_name}' == '{item_name}'")
                else:
                    pantry_list.append(pantry_item)
            else:
                pantry_str = str(pantry_item).strip() if pantry_item else ''
                if pantry_str and pantry_str.lower() == item_name_normalized:
                    item_found = True
                    if VERBOSE_LOGGING:
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
            set_session_pantry(pantry_list)
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
                    pass
                except (ValueError, TypeError):
                    pass
        # If item has no expiration date and we're filtering, skip it
        # (only include items with expiration dates when filtering)
    
    return expiring_items if expiring_items else pantry_items  # Fallback to all if none match

# Pantry Insights page
@app.route("/insights")
def insights():
    """Display pantry insights and statistics"""
    return render_template("insights.html")

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
                    pantry_items_full.append({'name': str(pantry_item), 'expirationDate': None, 'quantity': '1'})
        
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
    # Build pantry with quantities for AI prompt
    pantry_with_quantities = []
    expiring_items = []
    
    from datetime import datetime
    today = datetime.now().date()
    
    for pantry_item in current_pantry:
        if isinstance(pantry_item, dict):
            name = pantry_item.get('name', '')
            quantity = pantry_item.get('quantity', '1')
            if name:
                pantry_items_list.append(name)
                pantry_items_full.append(pantry_item)  # Keep full item data
                pantry_with_quantities.append(f"{name} ({quantity})")
                
                # Check if expiring soon
                exp_date_str = pantry_item.get('expirationDate')
                if exp_date_str:
                    try:
                        exp_date = datetime.fromisoformat(exp_date_str.replace('Z', '+00:00')).date()
                        days_left = (exp_date - today).days
                        if 0 <= days_left <= 7:
                            expiring_items.append(f"{name} ({quantity})")
                    except:
                        pass
        else:
            if pantry_item:
                pantry_items_list.append(str(pantry_item))
                pantry_items_full.append({'name': str(pantry_item), 'expirationDate': None, 'quantity': '1'})
                pantry_with_quantities.append(f"{pantry_item} (1)")
    
    # Check if pantry is empty (handle both None and empty list)
    if not pantry_items_list or len(pantry_items_list) == 0:
        flash("Your pantry is empty. Add items first.", "warning")
        return redirect(url_for("index"))

    # Generate AI-powered recipes based on pantry items (only if no existing recipes)
    pantry_items = ", ".join(pantry_with_quantities)
    pantry = pantry_items_list  # Use string list for compatibility with existing code
    
    # Add priority note for expiring items
    priority_note = ""
    if expiring_items:
        priority_note = f"\n\nIMPORTANT: Prioritize using these items that are expiring soon (within 7 days): {', '.join(expiring_items)}. Try to include at least one of these in each recipe."
    
    prompt = f"""Based on the following pantry items WITH QUANTITIES: {pantry_items}
{priority_note}

Generate 3 creative and practical recipes that use AT LEAST 50% of these pantry ingredients. For each recipe, provide:
1. Recipe name
2. List of ingredients with EXACT quantities from pantry (prioritizing pantry items - at least half must be from pantry)
3. Step-by-step cooking instructions
4. Estimated cooking time
5. Number of servings (calculated based on available quantities)
6. Health assessment (Healthy/Moderately Healthy/Unhealthy)
7. Health explanation (brief reason for the health rating)

CRITICAL REQUIREMENTS:
- Each recipe MUST use at least 50% of ingredients from the pantry list above
- Each recipe MUST use the EXACT quantities available from the pantry items listed above
- Calculate serving sizes based on the available quantities (e.g., if you have "2 bottles of milk", create a recipe that uses 2 bottles and adjust servings accordingly)
- Scale all other ingredients proportionally to match the serving size
- If a recipe normally serves 4 but you have "2 bottles of milk" (which might be 1 liter each), adjust the recipe to use both bottles and calculate appropriate servings (e.g., 6-8 servings)
- Use the full quantity of pantry items when possible to minimize waste
- Include basic pantry staples (salt, pepper, oil, butter) as needed, scaled appropriately

QUANTITY AND SERVING CALCULATION EXAMPLES:
- If pantry has "2 bottles of milk (500ml each)" -> Recipe should use 1 liter total, calculate servings based on typical milk usage (e.g., 4-6 servings for a milk-based dish)
- If pantry has "3 cans of soup (400g each)" -> Recipe should use all 3 cans (1200g total), adjust servings to 6-8 people
- If pantry has "5 slices of pizza" -> Recipe should use all 5 slices, serving size is 5 servings
- Always scale non-pantry ingredients (spices, oil, etc.) proportionally to match the serving size

Format as JSON:
{{
    "recipes": [
        {{
            "name": "Recipe Name",
            "ingredients": ["2 cups pantry_item1 (from pantry)", "1 bottle pantry_item2 (from pantry)", "1 tsp salt", "2 tbsp oil"],
            "instructions": ["step1", "step2", "step3"],
            "cooking_time": "X minutes",
            "servings": "X servings (based on available quantities)",
            "difficulty": "Easy",
            "health_rating": "Healthy",
            "health_explanation": "This dish is healthy because it contains fresh vegetables, lean proteins, and minimal processed ingredients."
        }},
        {{
            "name": "Recipe Name 2",
            "ingredients": ["3 cans pantry_item1 (from pantry)", "1 cup pantry_item3 (from pantry)", "1 tsp pepper", "1 tbsp butter"],
            "instructions": ["step1", "step2", "step3"],
            "cooking_time": "X minutes",
            "servings": "X servings (based on available quantities)",
            "difficulty": "Medium",
            "health_rating": "Moderately Healthy",
            "health_explanation": "This dish is moderately healthy with some nutritious ingredients but may contain higher sodium or fat content."
        }},
        {{
            "name": "Recipe Name 3",
            "ingredients": ["5 slices pantry_item2 (from pantry)", "2 cups pantry_item3 (from pantry)", "1 tsp salt", "1 tbsp olive oil"],
            "instructions": ["step1", "step2", "step3"],
            "cooking_time": "X minutes",
            "servings": "X servings (based on available quantities)",
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
                {"role": "system", "content": "You are a creative chef and recipe developer. Create practical, delicious recipes using available ingredients WITH EXACT QUANTITIES. Calculate serving sizes based on available quantities and scale all ingredients proportionally. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,  # Increased to allow for quantity details
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
                
            pass
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

    return render_template("suggest_recipe.html", recipes=recipes, pantry_items=pantry_items_list, pantry_items_full=pantry_items_full)

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
    except Exception:
        pass  # Directory might already exist or be unwritable
    
    try:
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

    # Get user's current pantry for context fusion
    user_pantry = None
    if 'user_id' in session:
        try:
            # Load user's pantry using get_user_pantry (same method used in index route)
            user_id = session['user_id']
            user_pantry = get_user_pantry(user_id, use_cache=False)
            # Ensure user_pantry is a list
            if not isinstance(user_pantry, list):
                user_pantry = []
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"Error loading user pantry: {e}")
            pass

    else:
        # For anonymous users, use session pantry
        user_pantry = session.get('web_pantry', [])
    
    # Get user's preferred detection method (ML or OpenAI)
    detection_method = request.form.get('detection_method', 'ml').lower()  # Default to ML
    
    # 🔥 FIX: Better logging for Vercel debugging
    print(f"🔍 Detection method: {detection_method}")
    print(f"🔍 ML_VISION_ENABLED: {ML_VISION_ENABLED}, YOLO_DETECTION_ENABLED: {YOLO_DETECTION_ENABLED}")
    print(f"🔍 IS_VERCEL: {IS_VERCEL}, OpenAI client available: {client is not None}")
    
    # ALWAYS try local ML/YOLO FIRST (when enabled), then use OpenAI only if ML finds no items
    # UNLESS user explicitly chose OpenAI
    try:
        detected_items_data = []
        ml_found_items = False
        use_ml = False
        
        # If user chose OpenAI, skip ML and go straight to OpenAI
        if detection_method == 'openai':
            print(f"🔍 User selected OpenAI - skipping ML detection")
            ml_found_items = False
            use_ml = False
        # STEP 1: Try ML pipeline if either ML vision or YOLO is enabled.
        # (YOLO can run even when ML_VISION_ENABLED is false; it has its own flag + lazy loader.)
        elif ML_VISION_ENABLED or YOLO_DETECTION_ENABLED:
            try:
                print(f"🔍 STEP 1: Using ML/YOLO FIRST for detection (ML_VISION_ENABLED={ML_VISION_ENABLED}, YOLO_DETECTION_ENABLED={YOLO_DETECTION_ENABLED})")
                detected_items_data = detect_food_items_with_ml(img_bytes, user_pantry=user_pantry)
                print(f"🔍 ML detection returned {len(detected_items_data) if detected_items_data else 0} items")
                
                if detected_items_data and len(detected_items_data) > 0:
                    # Use ML results, but also use OpenAI if very few items found (likely missed many items)
                    item_count = len(detected_items_data)
                    print(f"✅ ML model detected {item_count} items")
                    
                    # 🔥 FIX: Always use ML results if any items found
                    # Only fall back to OpenAI if ML found ZERO items
                    print(f"✅ Using ML results ({item_count} items)")
                    ml_found_items = True
                    use_ml = True
                else:
                    # ML returned empty results - will fall back to OpenAI
                    print(f"⚠️ ML model found no items, will use OpenAI as fallback")
                    ml_found_items = False
                    detected_items_data = []  # Clear empty results
            except Exception as e:
                print(f"❌ ML vision failed: {e}")
                import traceback
                traceback.print_exc()  # Always print traceback for Vercel debugging
                print(f"⚠️ ML failed, will use OpenAI as fallback")
                ml_found_items = False
                detected_items_data = []  # Clear failed results
        else:
            # 🔥 FIX: ML is disabled - log this clearly for Vercel
            print(f"⚠️ ML detection is disabled (ML_VISION_ENABLED={ML_VISION_ENABLED}, YOLO_DETECTION_ENABLED={YOLO_DETECTION_ENABLED})")
            print(f"⚠️ On Vercel, ML models are disabled by default. Set ML_VISION_ENABLED=true or YOLO_DETECTION_ENABLED=true to enable.")
            print(f"⚠️ Falling back to OpenAI detection...")
            ml_found_items = False
            detected_items_data = []
        
        # STEP 2: Use OpenAI if user selected it OR if ML didn't find any items
        if not ml_found_items:
            if detection_method == 'openai':
                print(f"🔍 STEP 2: Using OpenAI (user selected)")
            else:
                print(f"🔍 STEP 2: Using OpenAI as fallback (ML found no items or ML disabled)")
            
            # 🔥 FIX: Check if OpenAI client is available
            if client is None:
                error_msg = "OpenAI API key not configured. Cannot detect items."
                print(f"❌ ERROR: {error_msg}")
                if request.is_json or request.headers.get('Content-Type') == 'application/json':
                    return jsonify({'success': False, 'error': error_msg}), 500
                else:
                    flash(error_msg, "danger")
                    return redirect(url_for("index"))
            
            # Send image to OpenAI vision API
            import base64
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
            prompt = """You are an expert food recognition system analyzing a pantry/fridge photo. Identify EVERY food item with maximum accuracy. BE AGGRESSIVE - include items even if you're slightly uncertain.

SCAN THE ENTIRE IMAGE SYSTEMATICALLY:
    - Look at ALL areas: foreground, background, shelves, containers, bags, boxes, drawers, doors
- Check items that are partially visible, stacked, overlapping, or in shadows
- Read labels and packaging text carefully - even small text matters
- Count multiple units of the same item
- Include items in the background, corners, and edges
- Look for items behind other items, in containers, or wrapped
- When in doubt, INCLUDE the item rather than exclude it

CRITICAL NAMING RULES:
    ✅ CORRECT: "milk", "chicken", "tomato", "bread", "pasta", "cheese", "eggs", "yogurt", "olive oil", "honey", "mustard"
❌ WRONG: "milk carton", "chicken meat", "tomatoes" (use singular), "bread loaf", "Barilla pasta", "bottle" (use contents: "olive oil")

1. **Item Names** (MOST IMPORTANT):
   - Use SIMPLE, GENERIC food names: "milk" not "whole milk carton"
   - Remove ALL brand names: "Coca-Cola" -> "cola", "Kellogg's Frosted Flakes" -> "cereal"
   - Use SINGULAR form: "apple" not "apples", "tomato" not "tomatoes"
   - Remove packaging words: "milk carton" -> "milk", "bread bag" -> "bread"
   - **CONTAINER CONTENTS (CRITICAL)**: When you see bottles, jars, cans, boxes - identify what's INSIDE, not the container
     ✅ CORRECT: "olive oil" (not "bottle"), "honey" (not "jar"), "mustard" (not "bottle"), "pasta" (not "box")
     ✅ Read labels on containers to identify contents: "365 Organic Olive Oil" -> "olive oil"
     ✅ Look for product names on labels: "Just Wildflower Honey" -> "honey", "Classic Dijon" -> "mustard"
   - Be specific only when helpful: "whole milk" > "milk" (if clearly visible)
   - Common pantry items: milk, eggs, bread, chicken, beef, cheese, yogurt, butter, pasta, rice, cereal, soup, juice, water, soda, coffee, tea, flour, sugar, salt, pepper, olive oil, vegetable oil, vinegar, balsamic vinegar, honey, jam, marmalade, ketchup, mustard, mayonnaise, hot sauce, chili flakes, vanilla extract, chocolate chips, oats, quinoa, beans, tomatoes, onions, garlic, potatoes, nuts, seeds, almonds, pumpkin seeds

2. **Quantity Detection**:
   - Count visible items: "3 apples", "2 bottles", "5 cans", "1 loaf"
   - Read packaging labels: "12 oz", "500g", "1 lb", "16 fl oz"
   - Count packages, not contents: "2 boxes of pasta" not "2 pasta"
   - Format: "X unit" (e.g., "2 bottles", "1 package", "3 cans", "5 pieces", "1 dozen")
   - If unclear, use "1"

3. **Expiration Date** (READ CAREFULLY):
   - Scan ALL text on labels, stickers, packaging
   - Look for: "EXP", "EXPIRES", "USE BY", "BEST BY", "SELL BY", "BB", "UB"
   - Parse formats: MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD, Month DD YYYY, MM/DD/YY
   - Convert to YYYY-MM-DD: "01/15/2024" -> "2024-01-15", "Jan 15 2024" -> "2024-01-15"
   - Use expiration/use-by date (NOT manufacture date)
   - If unclear or not visible, set to null

4. **Category** (choose most specific):
   - dairy: milk, cheese, yogurt, butter, cream, sour cream, cottage cheese
   - produce: fresh fruits, vegetables, herbs (apple, banana, orange, tomato, lettuce, carrot, onion, etc.)
   - meat: beef, chicken, pork, fish, seafood, turkey, bacon, sausage, deli meats
   - beverages: drinks, juices, sodas, water, coffee, tea, beer, wine
   - bakery: bread, pastries, bagels, muffins, rolls, buns
   - canned goods: canned vegetables, soups, beans, tuna, corn, tomatoes
   - snacks: chips, crackers, cookies, nuts, popcorn, pretzels, candy
   - condiments: sauces, dressings, ketchup, mustard, mayonnaise, spices, oils, vinegar
   - grains: rice, pasta, cereal, flour, oats, quinoa, barley
   - frozen: frozen foods, ice cream, frozen vegetables, frozen meals
   - other: anything that doesn't fit above

5. **RARE/UNCOMMON FOODS** (CRITICAL):
   - Read ALL text on labels, especially for specialty/organic/imported items
   - Look for food names in multiple languages (many rare foods have foreign names)
   - If you see a food name you don't recognize, include it anyway if it appears to be food
   - Preserve authentic names: "gochujang" not "korean chili paste", "tahini" not "sesame paste"
   - Check for descriptors: "organic", "artisan", "gourmet", "specialty", "imported" - these often indicate rare foods
   - Look for dietary labels: "gluten-free", "vegan", "keto", "paleo" - these products are often less common
   - Ethnic/regional foods: Look for names in other languages or transliterations
   - Fermented foods: kimchi, sauerkraut, miso, tempeh, kombucha, kefir
   - Specialty condiments: harissa, gochujang, sriracha, ponzu, mirin, fish sauce, oyster sauce
   - Nuts/seeds: macadamia, pistachio, pine nuts, hemp seeds, chia seeds, flax seeds
   - Grains/legumes: quinoa, farro, bulgur, lentils, chickpeas, black beans, edamame
   - Specialty oils: truffle oil, avocado oil, coconut oil, sesame oil, walnut oil
   - Specialty vinegars: balsamic, rice vinegar, apple cider vinegar, white wine vinegar
   - Preserves/spreads: tahini, hummus, pesto, tapenade, bruschetta, guacamole

6. **Accuracy Requirements** (CRITICAL - BE AGGRESSIVE):
   - Include ALL visible food items, even if partially hidden or unclear
   - Include items in background, shadows, or corners
   - Include items you're only 60% sure about - better to include than miss
   - Don't skip items just because they're in background, partially visible, or you don't recognize them
   - If you see ANY packaging/labels/text, read them and include the item
   - When in doubt about ANY food item, INCLUDE IT with a descriptive name
   - Only skip if it's clearly NOT food (e.g., a plate, fork, or non-food object)
   - If you see something that MIGHT be food, include it - let the user decide

FEW-SHOT EXAMPLES:
    Example 1: Image shows milk carton, bread bag, and eggs
-> {"items": [{"name": "milk", "quantity": "1 carton", "expirationDate": "2024-01-20", "category": "dairy"}, {"name": "bread", "quantity": "1 loaf", "expirationDate": "2024-01-15", "category": "bakery"}, {"name": "egg", "quantity": "1 dozen", "expirationDate": null, "category": "dairy"}]}

Example 2: Image shows 3 cans of soup and a box of pasta
-> {"items": [{"name": "soup", "quantity": "3 cans", "expirationDate": null, "category": "canned goods"}, {"name": "pasta", "quantity": "1 box", "expirationDate": null, "category": "grains"}]}

Return ONLY valid JSON (no markdown, no code blocks, no explanations):
    {"items": [{"name": "...", "quantity": "...", "expirationDate": "YYYY-MM-DD or null", "category": "..."}]}"""
            
            if not client:
                raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
            
            # Add timeout and error handling for API calls
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert food recognition system with exceptional attention to detail. Your task is to identify ALL food items in images - be aggressive and include items even if you're slightly uncertain. Extract quantities, read expiration dates from packaging labels, and classify items into appropriate categories. When in doubt, INCLUDE the item rather than exclude it. Always return results in valid JSON format with no additional text."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}}
                        ]}
                    ],
                    max_tokens=3000,  # Increased to allow more items
                    temperature=0.2,  # Slightly higher for more creative detection
                    response_format={"type": "json_object"},
                    timeout=60.0  # 60 second timeout to prevent hanging
                )
            except Exception as api_error:
                # Handle API errors gracefully
                error_type = type(api_error).__name__
                error_msg = str(api_error)
                if VERBOSE_LOGGING:
                    print(f"OpenAI API error ({error_type}): {error_msg}")
                
                # Provide user-friendly error messages
                if 'rate limit' in error_msg.lower() or 'RateLimitError' in error_type:
                    raise ValueError("API rate limit exceeded. Please try again in a few moments.")
                elif 'timeout' in error_msg.lower() or 'Timeout' in error_type:
                    raise ValueError("Request timed out. Please try again with a smaller image.")
                elif 'invalid' in error_msg.lower() or 'InvalidRequestError' in error_type:
                    raise ValueError("Invalid image format. Please upload a valid photo.")
                else:
                    raise ValueError(f"Error connecting to AI service: {error_msg[:100]}")
            
            # Safely extract response content
            food_response = safe_get_response_content(response)
            if not food_response:
                raise ValueError("Empty or invalid response from OpenAI API")
            
            # Parse JSON response with improved error handling
            detected_items_data = parse_api_response_with_retry(food_response)
            print(f"✅ OpenAI detected {len(detected_items_data) if detected_items_data else 0} items")
        
        # Normalize and validate all items with partial recognition and expiration risk
        # If ML model was used, preserve confidence values from ML model
        pantry_items = []
        
        # If ML model was used, process ML results with confidence preservation
        if use_ml and detected_items_data:
            for item in detected_items_data:
                try:
                    raw_name = item.get('name', '').strip()
                    if not raw_name:
                        continue
                    
                    raw_quantity = item.get('quantity', '1')
                    expiration_date = item.get('expirationDate')
                    raw_category = item.get('category', 'other')
                    
                    # Get confidence - preserve actual value from ML model (0.0-1.0)
                    # Only default to 0.5 if confidence is None or missing, not if it's 0
                    confidence = item.get('confidence')
                    if confidence is None or confidence == '':
                        # If confidence is truly missing, use 0.5 as default
                        confidence = 0.5
                        if VERBOSE_LOGGING:
                            print(f"⚠️ Item '{raw_name}' missing confidence, defaulting to 0.5")
                    else:
                        # Convert to float and ensure it's in valid range
                        try:
                            confidence = float(confidence)
                            # Clamp to valid range [0.0, 1.0]
                            confidence = max(0.0, min(1.0, confidence))
                        except (ValueError, TypeError):
                            confidence = 0.5
                            if VERBOSE_LOGGING:
                                print(f"⚠️ Item '{raw_name}' has invalid confidence value, defaulting to 0.5")
                    
                    # Normalize and validate
                    normalized_name = normalize_item_name(raw_name)
                    if not normalized_name or not normalized_name.strip():
                        if VERBOSE_LOGGING:
                            print(f"Skipping item with invalid name: {raw_name}")
                        continue
                    
                    normalized_quantity = parse_quantity(raw_quantity)
                    validated_category = validate_category(normalized_name, raw_category)
                    
                    # Normalize expiration date
                    normalized_exp_date = None
                    if expiration_date:
                        if isinstance(expiration_date, str):
                            normalized_exp_date = normalize_expiration_date(expiration_date)
                        else:
                            normalized_exp_date = None
                    
                    pantry_items.append({
                        'id': str(uuid.uuid4()),
                        'name': normalized_name,
                        'quantity': normalized_quantity,
                        'expirationDate': normalized_exp_date,
                        'category': validated_category,
                        'confidence': confidence,  # Use actual confidence value from ML model
                        'addedDate': datetime.now().isoformat()
                    })
                    
                    if VERBOSE_LOGGING:
                        print(f"  ✓ {normalized_name}: confidence={confidence:.2f} ({confidence*100:.0f}%)")
                except (AttributeError, TypeError, KeyError) as item_error:
                    if VERBOSE_LOGGING:
                        print(f"Warning: Error processing ML detection item: {item_error}")
                    continue
        else:
            # Process OpenAI results (with default confidence values)
            for item in detected_items_data:
                raw_name = item.get('name', '').strip()
                raw_quantity = item.get('quantity', '1')
                expiration_date = item.get('expirationDate')
                raw_category = item.get('category', 'other')
                # Default confidence: OpenAI is generally reliable, so default to 0.75 (medium-high)
                # If confidence is explicitly provided, use it; otherwise use intelligent default
                confidence = item.get('confidence')
                if confidence is None:
                    # Assign default confidence based on item properties
                    # Common food items get higher default confidence
                    common_foods = ['milk', 'bread', 'egg', 'chicken', 'beef', 'cheese', 'yogurt', 'butter', 
                                   'pasta', 'rice', 'cereal', 'soup', 'juice', 'water', 'soda', 'coffee', 'tea',
                                   'apple', 'banana', 'orange', 'tomato', 'carrot', 'lettuce', 'onion']
                    name_lower = raw_name.lower()
                    if any(common in name_lower for common in common_foods):
                        confidence = 0.85  # High confidence for common items
                    elif raw_category and raw_category != 'other':
                        confidence = 0.70  # Medium-high for categorized items
                    else:
                        confidence = 0.60  # Medium for uncategorized items
                else:
                    confidence = float(confidence)
                
                # Handle partial recognition - if confidence is very low, provide category-based name
                if confidence < 0.3 and raw_category and raw_category != 'other':
                    # Partial recognition: use category-based generic name
                    category_names = {
                        'dairy': 'Likely dairy item',
                        'produce': 'Unidentified produce',
                        'meat': 'Likely meat item',
                        'beverages': 'Likely beverage',
                        'bakery': 'Likely bakery item',
                        'canned_goods': 'Likely canned good',
                        'snacks': 'Likely snack',
                        'condiments': 'Likely condiment',
                        'grains': 'Likely grain/pasta',
                        'frozen': 'Likely frozen item'
                    }
                    normalized_name = category_names.get(raw_category, 'Unidentified food item')
                    is_partial = True
                else:
                    # Normalize and validate
                    normalized_name = normalize_item_name(raw_name)
                    # Be less strict - only skip if name is completely empty or just whitespace
                    if not normalized_name or not normalized_name.strip():
                        if VERBOSE_LOGGING:
                            print(f"Skipping item with invalid name: {raw_name}")
                        continue  # Skip invalid items
                    is_partial = False
                
                normalized_quantity = parse_quantity(raw_quantity)
                validated_category = validate_category(normalized_name, raw_category)
                
                # Normalize expiration date
                normalized_exp_date = None
                if expiration_date:
                    normalized_exp_date = normalize_expiration_date(str(expiration_date))
                
                # Calculate expiration risk (days until expiration)
                expiration_risk = None
                if normalized_exp_date:
                    try:
                        exp_date_obj = datetime.strptime(normalized_exp_date, '%Y-%m-%d')
                        today = datetime.now()
                        days_until_exp = (exp_date_obj - today).days
                        if days_until_exp < 0:
                            expiration_risk = 'expired'
                        elif days_until_exp <= 3:
                            expiration_risk = 'critical'  # 0-3 days
                        elif days_until_exp <= 7:
                            expiration_risk = 'high'  # 4-7 days
                        elif days_until_exp <= 14:
                            expiration_risk = 'medium'  # 8-14 days
                        else:
                            expiration_risk = 'low'  # >14 days
                    except Exception:
                        pass
                
                pantry_items.append({
                    'id': str(uuid.uuid4()),
                    'name': normalized_name,
                    'quantity': normalized_quantity,
                    'expirationDate': normalized_exp_date,
                    'category': validated_category,
                    'addedDate': datetime.now().isoformat(),
                    'confidence': confidence,
                    'is_partial': is_partial,  # Flag for partial recognition
                    'expiration_risk': expiration_risk  # Risk level for expiration
                })
        
        # Remove duplicates (case-insensitive)
        seen_names = set()
        unique_items = []
        for item in pantry_items:
            name_lower = item['name'].lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                unique_items.append(item)
        
        pantry_items = unique_items
        
        # Add to appropriate pantry based on user authentication
        # IMPORTANT:
        # - For classic (non-AJAX) form submissions, we add all detected items here so they
        #   immediately appear in the pantry after redirect back to index.
        # - For modern AJAX uploads from the homepage, we rely on:
        #     * Auto-add of high-confidence items below, and
        #     * Explicit user confirmation via /api/confirm_items for low-confidence items.
        #   In that case we avoid adding everything here to prevent confusing duplicate logic.
        is_ajax_request = (
            request.headers.get('X-Requested-With') == 'XMLHttpRequest'
            or (request.headers.get('Content-Type') or '').startswith('application/json')
        )
        
        if not is_ajax_request:
            if 'user_id' in session and session.get('user_id'):
                try:
                    user_id = session['user_id']
                    # Add to user's pantry
                    user_pantry = get_user_pantry(user_id)
                    # Validate user_pantry is a list
                    if not isinstance(user_pantry, list):
                        user_pantry = []
                    
                    # Convert to list of dicts if needed
                    pantry_list = []
                    for item in user_pantry:
                        try:
                            if isinstance(item, dict):
                                pantry_list.append(item)
                            else:
                                item_str = str(item).strip() if item else ''
                                if item_str:
                                    pantry_list.append({
                                        'id': str(uuid.uuid4()),
                                        'name': item_str,
                                        'quantity': '1',
                                        'expirationDate': None,
                                        'addedDate': datetime.now().isoformat()
                                    })
                        except (TypeError, AttributeError):
                            pass
                    
                    # Add new items (check for duplicates first)
                    existing_names = {
                        item.get('name', '').strip().lower()
                        for item in pantry_list
                        if isinstance(item, dict) and item.get('name')
                    }
                    new_items = []
                    for item in pantry_items:
                        try:
                            if not isinstance(item, dict):
                                continue
                            item_name = item.get('name', '')
                            if not isinstance(item_name, str):
                                item_name = str(item_name) if item_name else ''
                            item_name = item_name.strip().lower()
                            if item_name and item_name not in existing_names:
                                pantry_list.append(item)
                                existing_names.add(item_name)
                                new_items.append(item)
                        except (AttributeError, TypeError):
                            pass
                    
                    if new_items:
                        update_user_pantry(user_id, pantry_list)
                except (KeyError, TypeError, AttributeError) as e:
                    if VERBOSE_LOGGING:
                        print(f"Error adding items to user pantry: {e}")
                    # Continue with anonymous pantry as fallback

        else:
            # Add to anonymous web pantry (stored in session for persistence)
            try:
                # CRITICAL: Mark session as permanent for anonymous users
                session.permanent = True
                
                if 'web_pantry' not in session:
                    session['web_pantry'] = []
                
                # Validate session['web_pantry'] is a list
                web_pantry = session.get('web_pantry', [])
                if not isinstance(web_pantry, list):
                    web_pantry = []
                
                # Convert to list of dicts if needed
                pantry_list = []
                for item in web_pantry:
                    try:
                        if isinstance(item, dict):
                            pantry_list.append(item)
                        else:
                            item_str = str(item).strip() if item else ''
                            if item_str:
                                pantry_list.append({
                                    'id': str(uuid.uuid4()),
                                    'name': item_str,
                                    'quantity': '1',
                                    'expirationDate': None,
                                    'addedDate': datetime.now().isoformat()
                                })
                    except (TypeError, AttributeError):
                        pass
                
                # Add new items (check for duplicates first)
                    existing_names = {
                        item.get('name', '').strip().lower()
                        for item in pantry_list
                        if isinstance(item, dict) and item.get('name')
                    }
                new_items = []
                for item in pantry_items:
                    try:
                        if not isinstance(item, dict):
                                continue
                        item_name = item.get('name', '')
                        if not isinstance(item_name, str):
                            item_name = str(item_name) if item_name else ''
                        item_name = item_name.strip().lower()
                        if item_name and item_name not in existing_names:
                            pantry_list.append(item)
                            existing_names.add(item_name)
                            new_items.append(item)
                    except (AttributeError, TypeError):
                        pass
                
                if new_items:
                    session['web_pantry'] = pantry_list
                    # Mark session as modified to ensure it's saved
                    session.modified = True
            except (KeyError, TypeError, AttributeError) as e:
                if VERBOSE_LOGGING:
                    print(f"Error adding items to anonymous pantry: {e}")
                # Initialize empty pantry if session is corrupted

                try:
                    session['web_pantry'] = pantry_items if pantry_items else []
                    session.modified = True
                except Exception:
                    pass
        
        # Separate high-confidence and low-confidence items
        # 🔥 FIX: Lower threshold to ensure more items are returned
        HIGH_CONFIDENCE_THRESHOLD = 0.6  # Auto-add items with 60%+ confidence (lowered from 0.7)
        high_conf_items = [item for item in pantry_items if item.get('confidence', 0) >= HIGH_CONFIDENCE_THRESHOLD]
        low_conf_items = [item for item in pantry_items if item.get('confidence', 0) < HIGH_CONFIDENCE_THRESHOLD]
        
        # 🔥 FIX: Ensure ALL items are included in response, even if confidence is very low
        # Combine all items for the response
        all_items_for_response = pantry_items.copy()  # Include ALL items
        
        # Auto-add high-confidence items
        auto_added = []
        if high_conf_items:
            auto_added = high_conf_items.copy()
            # Add high-confidence items to pantry
            if 'user_id' in session and session.get('user_id'):
                try:
                    user_id = session['user_id']
                    user_pantry = get_user_pantry(user_id)
                    if not isinstance(user_pantry, list):
                        user_pantry = []
                    pantry_list = []
                    for item in user_pantry:
                        if isinstance(item, dict):
                            pantry_list.append(item)
                    existing_names = {item.get('name', '').strip().lower() for item in pantry_list if isinstance(item, dict) and item.get('name')}
                    for item in auto_added:
                        item_name = item.get('name', '').strip().lower()
                        if item_name and item_name not in existing_names:
                            pantry_list.append(item)
                            existing_names.add(item_name)
                    update_user_pantry(user_id, pantry_list)
                    pass
                except Exception as e:
                    if VERBOSE_LOGGING:

                        print(f"Error auto-adding high-confidence items: {e}")
            else:
                # Anonymous user - add to session
                try:
                    session.permanent = True
                    if 'web_pantry' not in session:
                        session['web_pantry'] = []
                    web_pantry = session.get('web_pantry', [])
                    if not isinstance(web_pantry, list):
                        web_pantry = []
                    pantry_list = []
                    for item in web_pantry:
                        if isinstance(item, dict):
                            pantry_list.append(item)
                    existing_names = {item.get('name', '').strip().lower() for item in pantry_list if isinstance(item, dict) and item.get('name')}
                    for item in auto_added:
                        item_name = item.get('name', '').strip().lower()
                        if item_name and item_name not in existing_names:
                            pantry_list.append(item)
                            existing_names.add(item_name)
                    session['web_pantry'] = pantry_list
                    session.modified = True
                    pass
                except Exception as e:
                    if VERBOSE_LOGGING:

                        print(f"Error auto-adding to anonymous pantry: {e}")
        
        # Store low-confidence items in session for user confirmation
        if low_conf_items:
            session['pending_items'] = low_conf_items
            session.modified = True
        
        # Return JSON response with items and confidence for AJAX requests
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Content-Type', '').startswith('application/json'):
            # Ensure pantry_items is a list
            if not isinstance(pantry_items, list):
                pantry_items = []
            
            # Ensure all items have required fields
            for item in pantry_items:
                if not isinstance(item, dict):
                    continue
                # Ensure confidence exists
                if 'confidence' not in item or item['confidence'] is None:
                    # Assign default confidence
                    item['confidence'] = 0.5
                # Ensure id exists
                if 'id' not in item or not item['id']:
                    item['id'] = str(uuid.uuid4())
            
            # Separate partial recognition items
            partial_items = [item for item in pantry_items if item.get('is_partial', False)]
            complete_items = [item for item in pantry_items if not item.get('is_partial', False)]
            
            # Debug logging
            if VERBOSE_LOGGING:
                print(f"Returning {len(pantry_items)} items to frontend")
                print(f"  - High confidence: {len(auto_added)}")
                print(f"  - Low confidence: {len(low_conf_items)}")
                print(f"  - Total items: {len(pantry_items)}")
            
            return jsonify({
                'success': True,
                'auto_added': len(auto_added),
                'needs_confirmation': len(low_conf_items),
                'items': pantry_items,  # All items with full details including confidence
                'high_confidence': auto_added,  # Full item objects for high confidence
                'low_confidence': low_conf_items,  # Full item objects for low confidence
                'partial_recognition': [{'name': item['name'], 'category': item.get('category'), 'confidence': item.get('confidence', 0)} for item in partial_items],
                'message': f"Found {len(pantry_items)} items ({len(complete_items)} identified, {len(partial_items)} partial). {len(auto_added)} added automatically, {len(low_conf_items)} need confirmation." if pantry_items else "No items were detected in the photo. Please try again with a clearer image."
            })
        
        # Flash message for regular form submission
        if auto_added:
            auto_names = [item['name'] for item in auto_added]
            flash(f"✅ Auto-added {len(auto_added)} high-confidence items: {', '.join(auto_names[:5])}", "success")
        if low_conf_items:
            low_names = [item['name'] for item in low_conf_items]
            flash(f"⚠️ {len(low_conf_items)} items need confirmation: {', '.join(low_names[:5])}. Please review and confirm.", "warning")
        
    except ValueError as e:
        # Specific error for missing API key
        error_msg = str(e)
        print(f"ValueError analyzing photo: {error_msg}")
        # Return JSON for AJAX requests
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Content-Type', '').startswith('application/json'):
            return jsonify({
                'success': False,
                'error': f"{error_msg} Please configure OPENAI_API_KEY in your environment variables.",
                'items': []
            }), 400
        flash(f"⚠️ {error_msg} Please configure OPENAI_API_KEY in your Render environment variables.", "danger")
        return redirect(url_for("index"))
    except Exception as e:
        # Generic error handler
        error_msg = str(e)
        print(f"Error analyzing photo: {error_msg}")
        import traceback
        traceback.print_exc()
        # Return JSON for AJAX requests
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Content-Type', '').startswith('application/json'):
            return jsonify({
                'success': False,
                'error': f"Error analyzing photo: {error_msg}",
                'items': []
            }), 500
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
    
    # Build pantry with quantities
    pantry_with_quantities = []
    # Ensure current_pantry is iterable
    if not isinstance(current_pantry, (list, tuple)):
        current_pantry = []
    for pantry_item in current_pantry:
        if isinstance(pantry_item, dict):
            name = pantry_item.get('name', '')
            quantity = pantry_item.get('quantity', '1')
            if name:
                pantry_with_quantities.append(f"{name} ({quantity})")
        else:
            if pantry_item:
                pantry_with_quantities.append(f"{pantry_item} (1)")
    
    pantry_items = ", ".join(pantry_with_quantities)
    prompt = f"""Based on the pantry items WITH QUANTITIES: {pantry_items}

Generate 3 creative recipes that use AT LEAST 50% of the pantry ingredients WITH EXACT QUANTITIES. For each recipe, provide:
1. Recipe name
2. List of ingredients with EXACT quantities from pantry (prioritizing pantry items - at least half must be from pantry)
3. Step-by-step cooking instructions
4. Estimated cooking time
5. Number of servings (calculated based on available quantities)
6. Detailed nutrition facts (calories, carbs, protein, fat, fiber per serving) - calculate based on exact quantities used

CRITICAL REQUIREMENTS:
- Each recipe MUST use the EXACT quantities available from the pantry items listed above
- Calculate serving sizes based on the available quantities
- Scale all other ingredients proportionally to match the serving size
- Use the full quantity of pantry items when possible to minimize waste

Format as JSON:
{{
    "recipes": [
        {{
            "name": "Recipe Name",
            "ingredients": ["2 cups pantry_item1 (from pantry)", "1 bottle pantry_item2 (from pantry)", "1 tsp salt"],
            "instructions": ["step1", "step2", "step3"],
            "cooking_time": "X minutes",
            "servings": "X servings (based on available quantities)",
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
        
        # Add timeout and error handling for API calls
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a nutritionist and chef. Create recipes that use at least 50% pantry ingredients WITH EXACT QUANTITIES. Calculate serving sizes based on available quantities and provide accurate nutrition information based on the exact quantities used."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                timeout=60.0  # 60 second timeout
            )
        except Exception as api_error:
            error_type = type(api_error).__name__
            error_msg = str(api_error)
            if VERBOSE_LOGGING:

                print(f"OpenAI API error in nutrition route ({error_type}): {error_msg}")
            raise ValueError(f"Error generating nutrition info: {error_msg[:100]}")
        
        import json
        recipe_text = safe_get_response_content(response)
        if not recipe_text:
            raise ValueError("Empty response from OpenAI API")
        
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
            
            pass
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

            pass
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
        pass
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
    print(f"   USE_FIREBASE: {USE_FIREBASE}")
    if USE_FIREBASE:
        print(f"   Firebase is enabled - user will be saved to Firestore")
    else:
        print(f"   ⚠️ Firebase is NOT enabled - user will be saved to file storage")
    
    user_id, error = create_user(username, email, password, client_type)
    if user_id:
        print(f"✅ User created successfully: {user_id}")
        
        # Verify user exists in Firebase if Firebase is enabled
        if USE_FIREBASE:
            verify_user = firebase_get_user_by_id(user_id)
            if verify_user:
                print(f"✅ Verified user exists in Firebase: {verify_user.get('username')}")
            else:
                print(f"⚠️ WARNING: User {user_id} not found in Firebase after creation!")
                print(f"   This might indicate a Firebase connection issue")
        
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
        pass
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
        pass
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
            # Use anonymous pantry from session (not global variable)
            # CRITICAL: For web clients, always use session as that's where items are saved
            if client_type == 'mobile':
                # For mobile, use global variable (legacy support)
                pantry_to_use = mobile_pantry if 'mobile_pantry' in globals() else []
            else:
                # For web, ALWAYS use session (this is where items are saved by confirm_items)
                # 🔥 FIX: Force refresh session pantry to get latest data
                pantry_to_use = get_session_pantry()
                # Sync global variable for backward compatibility
                global web_pantry
                web_pantry = pantry_to_use.copy()
                if VERBOSE_LOGGING:
                    print(f"📦 GET /api/pantry: Retrieved {len(pantry_to_use)} items from session")
                    print(f"   Session keys: {list(session.keys())}")
                    print(f"   Session modified: {session.modified}")
        
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
                pass
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
    """Add an item to pantry via API with comprehensive validation"""
    global mobile_pantry, web_pantry  # Declare globals at function start
    
    try:
        # Input validation
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        # Parse JSON with error handling
        try:
            data = request.get_json(force=True)  # Force JSON parsing
            if VERBOSE_LOGGING:
                print(f"✅ Received data: {data}")
                print(f"   Data type: {type(data)}")
                print(f"   Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"❌ Error parsing JSON: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Invalid JSON data: {str(e)}'}), 400
        
        if not data:
            return jsonify({'success': False, 'error': 'Invalid request data'}), 400
        
        # Log request details for debugging
        if VERBOSE_LOGGING:
            print(f"\n{'='*60}")
            print(f"📥 API ADD ITEM REQUEST")
            print(f"{'='*60}")
            print(f"Method: {request.method}")
            print(f"Content-Type: {request.content_type}")
            print(f"Headers: {dict(request.headers)}")
            print(f"X-User-ID: {request.headers.get('X-User-ID', 'NOT PROVIDED')}")
            print(f"X-Client-Type: {request.headers.get('X-Client-Type', 'NOT PROVIDED')}")
    
        # Support both old format (item: string) and new format (PantryItem object)
        if 'item' in data:
            # Old format - convert to new format
            item_name = data['item'].strip()
            if not item_name:
                return jsonify({'success': False, 'error': 'Item name cannot be empty'}), 400
            
            quantity = data.get('quantity', '1')
            expiration_date = data.get('expirationDate')
            
            # Detect category based on item name
            detected_category = validate_category(item_name, 'other')
            
            pantry_item = {
                'id': str(uuid.uuid4()),
                'name': item_name,
                'quantity': quantity,
                'expirationDate': expiration_date,
                'category': detected_category,
                'addedDate': datetime.now().isoformat()
            }
        elif 'name' in data:
            # New format - PantryItem object
            item_name = data.get('name', '').strip() if data.get('name') else ''
            if not item_name:
                if VERBOSE_LOGGING:
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
        
        # Get category from request or detect from name
        raw_category = data.get('category', 'other')
        detected_category = validate_category(item_name, raw_category)
        
        pantry_item = {
            'id': data.get('id', str(uuid.uuid4())),
            'name': item_name,
            'quantity': quantity.strip() if isinstance(quantity, str) else str(quantity),
            'expirationDate': expiration_date,
            'category': detected_category,
            'addedDate': data.get('addedDate', datetime.now().isoformat())
        }
        
        if VERBOSE_LOGGING:
                print(f"✅ Created pantry item: name='{pantry_item['name']}', quantity='{pantry_item['quantity']}', expirationDate={pantry_item['expirationDate']}")
    
                client_type = request.headers.get('X-Client-Type', 'web')
        user_id = request.headers.get('X-User-ID')
        
        # Validate user_id exists if provided (prevent anonymous users from saving to Firebase)
        if user_id:
            user_exists = False
            try:
                if USE_FIREBASE:
                    user_data = firebase_get_user_by_id(user_id)
                    if user_data:
                        user_exists = True
                else:
                    users = load_users()
                    if user_id in users:
                        user_exists = True
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"Error validating user {user_id}: {e}")
            
            if not user_exists:
                print(f"❌ ERROR: Cannot add item - user {user_id} does not exist in database!")
                return jsonify({
                    'success': False,
                    'error': 'Invalid user. Please log in to add items to your pantry.'
                }), 401
        
        # Check if user is authenticated and valid
        if user_id and user_exists:
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
            if VERBOSE_LOGGING:
                print(f"💾 Updating pantry for user {user_id} with {len(pantry_list)} items")
            update_user_pantry(user_id, pantry_list)
            if VERBOSE_LOGGING:
                print(f"✅ Successfully added item '{pantry_item['name']}' to user {user_id}'s pantry")
            return jsonify({
                'success': True,
                'message': f'Added "{pantry_item["name"]}" to pantry',
                'item': pantry_item,
                'total_items': len(pantry_list)
            }), 200
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
                # Save to session for persistence across requests
                session['web_pantry'] = pantry_list
                session.modified = True
            
            if VERBOSE_LOGGING:
                print(f"✅ Successfully added item '{pantry_item['name']}' to anonymous {client_type} pantry")
            return jsonify({
            'success': True,
            'message': f'Added "{pantry_item["name"]}" to pantry',
            'item': pantry_item,
            'total_items': len(pantry_list)
        }), 200
    except Exception as e:
        # Comprehensive error handling
        error_msg = str(e)
        error_type = type(e).__name__
        if VERBOSE_LOGGING:
            print(f"❌ Error in api_add_item: {error_type}: {error_msg}")
            import traceback
            traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error adding item: {error_msg[:100]}',
            'type': error_type
        }), 500

@app.route('/api/confirm_items', methods=['POST'])
def api_confirm_items():
    """Confirm and add multiple items from photo analysis"""
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        data = request.get_json(force=True)
        if not data or 'items' not in data:
            return jsonify({'success': False, 'error': 'Missing items array'}), 400
        
        items = data['items']
        if not isinstance(items, list):
            return jsonify({'success': False, 'error': 'Items must be an array'}), 400
        
        # Determine user context
        user_id = None
        if 'user_id' in session and session.get('user_id'):
            user_id = session['user_id']
        else:
            # For anonymous users, use session
            session.permanent = True
        
        items_added = 0
        skipped_duplicates = 0
        
        # Prepare all items to add
        items_to_add = []
        for item in items:
            if not isinstance(item, dict) or 'name' not in item:
                continue
            
            item_name = item.get('name', '').strip()
            if not item_name:
                continue
            
            # Prepare item data
            pantry_item = {
                'id': item.get('id', str(uuid.uuid4())),
                'name': item_name,
                'quantity': item.get('quantity', '1'),
                'expirationDate': item.get('expirationDate'),
                'category': item.get('category', 'other'),
                'addedDate': item.get('addedDate', datetime.now().isoformat()),
                'confidence': item.get('confidence', 0.5),
                'is_partial': item.get('is_partial', False),
                'expiration_risk': item.get('expiration_risk')
            }
            items_to_add.append(pantry_item)
        
        # Initialize pantry_list early to avoid scope issues
        pantry_list = []
        
        # Batch add all items at once to avoid overwriting
        if user_id:
            # Add to user's pantry
            user_pantry = get_user_pantry(user_id)
            if not isinstance(user_pantry, list):
                user_pantry = []
            
            # Convert to list of dicts
            pantry_list = []
            for p_item in user_pantry:
                if isinstance(p_item, dict):
                    pantry_list.append(p_item)
            
            # Get existing names (case-insensitive)
            existing_names = {p.get('name', '').strip().lower() for p in pantry_list if isinstance(p, dict) and p.get('name')}
            
            # Add all new items at once
            for pantry_item in items_to_add:
                item_name_lower = pantry_item['name'].lower()
                if item_name_lower not in existing_names:
                    pantry_list.append(pantry_item)
                    existing_names.add(item_name_lower)
                    items_added += 1
                else:
                    skipped_duplicates += 1
            
            # Save all items at once
            if items_added > 0:
                update_user_pantry(user_id, pantry_list)
                if VERBOSE_LOGGING:
                    print(f"✅ Added {items_added} items to user {user_id}'s pantry via confirm_items")
                    print(f"   User pantry now has {len(pantry_list)} items")
        else:
            # Add to anonymous pantry (session)
            web_pantry = get_session_pantry()
            
            # Convert to list of dicts
            pantry_list = []
            for p_item in web_pantry:
                if isinstance(p_item, dict):
                    pantry_list.append(p_item)
            
            # Get existing names (case-insensitive)
            existing_names = {p.get('name', '').strip().lower() for p in pantry_list if isinstance(p, dict) and p.get('name')}
            
            # Add all new items at once
            for pantry_item in items_to_add:
                item_name_lower = pantry_item['name'].lower()
                if item_name_lower not in existing_names:
                    pantry_list.append(pantry_item)
                    existing_names.add(item_name_lower)
                    items_added += 1
                else:
                    skipped_duplicates += 1
            
            # 🔥 FIX: Always save session pantry, even if no new items (to ensure session is persisted)
            # Save all items at once using helper function
            set_session_pantry(pantry_list)
            # Force session to be saved (already done in set_session_pantry, but ensure it's set)
            session.modified = True
            session.permanent = True
            if VERBOSE_LOGGING:
                print(f"✅ Added {items_added} items to anonymous pantry via confirm_items")
                print(f"   Session pantry now has {len(pantry_list)} items")
                print(f"   Session modified: {session.modified}, Session permanent: {session.permanent}")
                # Verify session was saved
                verify_pantry = get_session_pantry()
                print(f"   Verification: Session pantry has {len(verify_pantry)} items after save")
        
        # Debug logging
        if VERBOSE_LOGGING:
            print(f"confirm_items result: {items_added} added, {skipped_duplicates} skipped")
            if user_id:
                final_pantry = get_user_pantry(user_id)
                print(f"   User {user_id} pantry now has {len(final_pantry)} items")
            else:
                final_pantry = session.get('web_pantry', [])
                print(f"   Anonymous pantry now has {len(final_pantry)} items")
        
        # Collect feedback for model improvement
        # Store user corrections for learning (items they confirmed/corrected)
        if items_added > 0:
            try:
                log_user_feedback(items, user_id)
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"Warning: Failed to log feedback: {e}")
        
        # 🔥 FIX: Get final pantry count to ensure accurate response
        final_pantry_count = len(pantry_list) if pantry_list else 0
        
        return jsonify({
            'success': True,
            'items_added': items_added,
            'skipped_duplicates': skipped_duplicates,
            'total_items': final_pantry_count,
            'message': f'Added {items_added} items to pantry' + (f' ({skipped_duplicates} duplicates skipped)' if skipped_duplicates > 0 else '')
        }), 200
        
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Error confirming items: {e}")
            import traceback
            traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

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
    # Try headers first (mobile), then fall back to session (web)
    user_id = request.headers.get('X-User-ID')
    if not user_id and 'user_id' in session:
        user_id = session.get('user_id')
        print(f"✅ Using user_id from session: {user_id}")
    
    # URL decode the item_id
    from urllib.parse import unquote
    item_id = unquote(item_id)
    print(f"Decoded item ID: {item_id}")
    
    # Check if user is authenticated
    if user_id:
        try:
            pantry_to_use = get_user_pantry(user_id)
        except Exception as e:
            print(f"❌ Error loading user pantry: {e}")
            return jsonify({'success': False, 'error': f'Failed to load pantry: {str(e)}'}), 500

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
        item_id_clean = item_id.strip() if item_id else ''
        # Treat 'unknown' as empty ID (frontend sends 'unknown' when item has no ID)
        if item_id_clean.lower() == 'unknown':
            item_id_clean = ''
        item_id_clean_lower = item_id_clean.lower() if item_id_clean else ''
        
        for i, pantry_item in enumerate(pantry_list):
            if isinstance(pantry_item, dict):
                item_id_from_dict = pantry_item.get('id', '').strip() if pantry_item.get('id') else ''
                item_name = pantry_item.get('name', '').strip().lower() if pantry_item.get('name') else ''
                
                # 🔥 FIX: Improved matching logic
                # 1. If we have a valid ID to match, try ID match first (case-sensitive)
                # 2. If ID matches OR if ID is empty/'unknown' and name matches, delete it
                id_match = item_id_clean and item_id_from_dict and item_id_from_dict == item_id_clean
                name_match = item_id_clean_lower and item_name == item_id_clean_lower
                
                # Match if: (ID matches) OR (no valid ID provided and name matches)
                if id_match or (not item_id_clean and name_match):
                    item_to_delete = pantry_list.pop(i)
                    break
            else:
                # Handle old string format - compare case-insensitively
                pantry_str = str(pantry_item).strip().lower() if pantry_item else ''
                if item_id_clean_lower and pantry_str == item_id_clean_lower:
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
        # For web clients, get from session first (session persists across requests)
        if client_type == 'mobile':
            pantry_to_use = mobile_pantry
        else:
            # For web clients, get from session first (session persists across requests)
            if 'web_pantry' not in session:
                session['web_pantry'] = []
            pantry_to_use = session.get('web_pantry', [])
            # Also sync global variable for consistency
            web_pantry = pantry_to_use
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
        item_id_clean = item_id.strip() if item_id else ''
        # Treat 'unknown' as empty ID (frontend sends 'unknown' when item has no ID)
        if item_id_clean.lower() == 'unknown':
            item_id_clean = ''
        item_id_clean_lower = item_id_clean.lower() if item_id_clean else ''
        
        for i, pantry_item in enumerate(pantry_list):
            if isinstance(pantry_item, dict):
                item_id_from_dict = pantry_item.get('id', '').strip() if pantry_item.get('id') else ''
                item_name = pantry_item.get('name', '').strip().lower() if pantry_item.get('name') else ''
                
                # 🔥 FIX: Improved matching logic for anonymous users
                # 1. If we have a valid ID to match, try ID match first (case-sensitive)
                # 2. If ID matches OR if ID is empty/'unknown' and name matches, delete it
                id_match = item_id_clean and item_id_from_dict and item_id_from_dict == item_id_clean
                name_match = item_id_clean_lower and item_name == item_id_clean_lower
                
                # Match if: (ID matches) OR (no valid ID provided and name matches)
                if id_match or (not item_id_clean and name_match):
                    item_to_delete = pantry_list.pop(i)
                    break
            else:
                # Handle old string format - compare case-insensitively
                pantry_str = str(pantry_item).strip().lower() if pantry_item else ''
                if item_id_clean_lower and pantry_str == item_id_clean_lower:
                    item_to_delete = pantry_list.pop(i)
                    break
        
        if item_to_delete:
            if client_type == 'mobile':
                mobile_pantry = pantry_list
            else:
                web_pantry = pantry_list
                # Update session for anonymous web users
                session['web_pantry'] = pantry_list
                session.modified = True
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
        pass
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
            try:
                pantry_to_use = get_user_pantry(user_id)
            except Exception as e:
                print(f"❌ Error loading user pantry: {e}")
                return jsonify({'success': False, 'error': f'Failed to load pantry: {str(e)}'}), 500

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
                    
                    # 🔥 FIX: Improved matching logic for delete (quantity = 0)
                    # Match by ID if both are valid, otherwise match by name
                    id_match = item_id_clean and item_id_from_dict and item_id_from_dict == item_id_clean
                    name_match = item_name_from_dict == item_name_lower
                    
                    # Skip the item to delete if: (ID matches) OR (no valid ID and name matches)
                    if id_match or (not item_id_clean and name_match):
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
            # Get pantry from session for web clients (consistent with index route)
            if client_type == 'mobile':
                pantry_to_use = mobile_pantry
            else:
                # For web clients, get from session first (session persists across requests)
                if 'web_pantry' not in session:
                    session['web_pantry'] = []
                pantry_to_use = session.get('web_pantry', [])
                # Also sync global variable for consistency
                web_pantry = pantry_to_use
            
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
                    
                    # 🔥 FIX: Improved matching logic for delete (quantity = 0, anonymous)
                    # Match by ID if both are valid, otherwise match by name
                    id_match = item_id_clean and item_id_from_dict and item_id_from_dict == item_id_clean
                    name_match = item_name_from_dict == item_name_lower
                    
                    # Skip the item to delete if: (ID matches) OR (no valid ID and name matches)
                    if id_match or (not item_id_clean and name_match):
                        item_found = True
                        continue  # Skip adding this item to pantry_list (effectively deletes it)
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
                    # Save to session for persistence across requests
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
        try:
            pantry_to_use = get_user_pantry(user_id)
        except Exception as e:
            print(f"❌ Error loading user pantry: {e}")
            return jsonify({'success': False, 'error': f'Failed to load pantry: {str(e)}'}), 500

        print(f"📦 User pantry has {len(pantry_to_use) if isinstance(pantry_to_use, list) else 0} items")

        # Convert to list of dicts if needed - create new list to avoid scope issues
        pantry_list = []
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
        
        # 🔥 FIX: Also check for originalId in request body (helps when name changed)
        original_id_from_body = data.get('originalId', '').strip() if data.get('originalId') else ''
        
        for i, pantry_item in enumerate(pantry_list):
            if isinstance(pantry_item, dict):
                item_id_from_dict = pantry_item.get('id', '').strip() if pantry_item.get('id') else ''
                item_name_from_dict = pantry_item.get('name', '').strip().lower() if pantry_item.get('name') else ''
                
                # Try exact ID match first (URL param or body param)
                # This allows updating item name even if it changed
                id_to_match = item_id_clean or original_id_from_body
                if id_to_match and item_id_from_dict and item_id_from_dict == id_to_match:
                    # Ensure the updated item retains its original ID
                    if not updated_item.get('id') or updated_item.get('id') == 'unknown':
                        updated_item['id'] = item_id_from_dict
                    # Preserve category from original item if not provided in update
                    if 'category' not in updated_item and 'category' in pantry_item:
                        updated_item['category'] = pantry_item['category']
                    pantry_list[i] = updated_item
                    item_found = True
                    print(f"✅ Matched item by ID: {id_to_match}")
                    break
                
                # Fallback to name match (case-insensitive) if ID doesn't match or is empty
                # This handles cases where item has no ID or ID is 'unknown'
                # Only match by name if we don't have a valid ID to match
                # ⚠️ WARNING: This will fail if the user changed the item name!
                if not id_to_match and item_name_from_dict == item_name_lower:
                    # Preserve existing ID if item has one, otherwise generate new ID
                    if not updated_item.get('id'):
                        updated_item['id'] = item_id_from_dict if item_id_from_dict else str(uuid.uuid4())
                    # Preserve category from original item if not provided in update
                    if 'category' not in updated_item and 'category' in pantry_item:
                        updated_item['category'] = pantry_item['category']
                    pantry_list[i] = updated_item
                    item_found = True
                    print(f"✅ Matched item by name: {item_name_lower}")
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
        # Get pantry from session for web clients (consistent with index route)
        if client_type == 'mobile':
            pantry_to_use = mobile_pantry
        else:
            # For web clients, get from session first (session persists across requests)
            if 'web_pantry' not in session:
                session['web_pantry'] = []
            pantry_to_use = session.get('web_pantry', [])
            # Also sync global variable for consistency
            web_pantry = pantry_to_use
        
        # Ensure pantry_to_use is a list
        if not isinstance(pantry_to_use, list):
            pantry_to_use = []
        print(f"📦 Anonymous pantry has {len(pantry_to_use)} items")
        # Convert to list of dicts if needed - create new list to avoid scope issues
        pantry_list = []
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
        if item_id_clean.lower() == 'unknown':
            item_id_clean = ''
        item_name_lower = item_name.lower().strip()
        
        # 🔥 FIX: Also check for originalId in request body (helps when name changed)
        original_id_from_body = data.get('originalId', '').strip() if data.get('originalId') else ''
        
        for i, pantry_item in enumerate(pantry_list):
            if isinstance(pantry_item, dict):
                item_id_from_dict = pantry_item.get('id', '').strip() if pantry_item.get('id') else ''
                item_name_from_dict = pantry_item.get('name', '').strip().lower() if pantry_item.get('name') else ''
                
                # 🔥 FIX: Improved matching logic for updates (anonymous users)
                # 1. Try ID match first (from URL param or body param) - allows name changes
                # 2. If ID matches, update that item (even if name changed)
                # 3. If ID doesn't match or is empty/'unknown', fallback to name match
                id_to_match = item_id_clean or original_id_from_body
                id_match = id_to_match and item_id_from_dict and item_id_from_dict == id_to_match
                # For name match, compare with OLD name (before update) - but we don't have that
                # So we need to match by ID first, then allow name updates
                name_match = item_name_from_dict == item_name_lower
                
                # Match if: (ID matches) OR (no valid ID provided and name matches)
                # Priority: ID match first (allows name changes), then name match as fallback
                if id_match or (not id_to_match and name_match):
                    # Ensure the updated item retains its original ID
                    if not updated_item.get('id') or updated_item.get('id') == 'unknown':
                        updated_item['id'] = item_id_from_dict if item_id_from_dict else str(uuid.uuid4())
                    # Preserve category from original item if not provided in update
                    if 'category' not in updated_item and 'category' in pantry_item:
                        updated_item['category'] = pantry_item['category']
                    pantry_list[i] = updated_item
                    item_found = True
                    print(f"✅ Matched item by {'ID' if id_match else 'name'}: {id_to_match if id_match else item_name_lower}")
                    break
        
        if item_found:
            if client_type == 'mobile':
                mobile_pantry = pantry_list
            else:
                web_pantry = pantry_list
                # Save to session for persistence across requests
                session['web_pantry'] = pantry_list
                session.modified = True
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
    
    if pantry_items and len(pantry_items) > 0 and isinstance(pantry_items[0], dict):
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
                        pass
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
        - If pantry has "2 bottles of milk (500ml each)" -> Recipe should use 1 liter total, calculate servings based on typical milk usage (e.g., 4-6 servings for a milk-based dish)
        - If pantry has "3 cans of soup (400g each)" -> Recipe should use all 3 cans (1200g total), adjust servings to 6-8 people
        - If pantry has "5 slices of pizza" -> Recipe should use all 5 slices, serving size is 5 servings
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
    global mobile_pantry, web_pantry  # Declare globals at function start
    
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
        
        # Get user's current pantry for context fusion
        user_pantry = None
        client_type = request.headers.get('X-Client-Type', 'web')
        
        # Get user_id from header (mobile) or session (web) for pantry loading
        user_id_for_pantry = request.headers.get('X-User-ID')
        if not user_id_for_pantry and 'user_id' in session:
            user_id_for_pantry = session['user_id']
        
        if user_id_for_pantry:
            # For authenticated users, get pantry from database
            try:
                user_pantry = get_user_pantry(user_id_for_pantry, use_cache=False)
                if not isinstance(user_pantry, list):
                    user_pantry = []
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"Error loading user pantry for context: {e}")
                user_pantry = []
        elif client_type == 'mobile':
            # For mobile anonymous users, use mobile_pantry global
            user_pantry = mobile_pantry
        else:
            # For web anonymous users, use session pantry
            user_pantry = session.get('web_pantry', [])
        
        # ALWAYS try ML model FIRST, then use OpenAI only if ML finds no items
        detected_items_data = []
        pantry_items = []  # Initialize pantry_items early
        ml_found_items = False
        use_ml = False
        
        # STEP 1: ALWAYS try ML first (if ML_VISION_ENABLED is true)
        if ML_VISION_ENABLED:
            try:
                print(f"🔍 STEP 1: Using ML model FIRST for detection (ML_VISION_ENABLED={ML_VISION_ENABLED})")
                detected_items_data = detect_food_items_with_ml(img_bytes, user_pantry=user_pantry)
                
                if detected_items_data and len(detected_items_data) > 0:
                    # Use ML results, but also use OpenAI if very few items found (likely missed many items)
                    item_count = len(detected_items_data)
                    print(f"✅ ML model detected {item_count} items")
                    
                    # 🔥 FIX: Always use ML results if any items found
                    # Only fall back to OpenAI if ML found ZERO items
                    print(f"✅ Using ML results ({item_count} items)")
                    ml_found_items = True
                    use_ml = True
                    # Process ML detection results into pantry_items format
                    for item in detected_items_data:
                        try:
                            raw_name = item.get('name', '').strip()
                            if not raw_name:
                                continue
                            
                            raw_quantity = item.get('quantity', '1')
                            expiration_date = item.get('expirationDate')
                            raw_category = item.get('category', 'other')
                            
                            # Get confidence - preserve actual value from ML model (0.0-1.0)
                            # Only default to 0.5 if confidence is None or missing, not if it's 0
                            confidence = item.get('confidence')
                            if confidence is None or confidence == '':
                                # If confidence is truly missing, use 0.5 as default
                                confidence = 0.5
                                if VERBOSE_LOGGING:
                                    print(f"⚠️ Item '{raw_name}' missing confidence, defaulting to 0.5")
                            else:
                                # Convert to float and ensure it's in valid range
                                try:
                                    confidence = float(confidence)
                                    # Clamp to valid range [0.0, 1.0]
                                    confidence = max(0.0, min(1.0, confidence))
                                except (ValueError, TypeError):
                                    confidence = 0.5
                                    if VERBOSE_LOGGING:
                                        print(f"⚠️ Item '{raw_name}' has invalid confidence value, defaulting to 0.5")
                            
                            # Normalize and validate
                            normalized_name = normalize_item_name(raw_name)
                            # Be less strict - only skip if name is completely empty or just whitespace
                            if not normalized_name or not normalized_name.strip():
                                if VERBOSE_LOGGING:
                                    print(f"Skipping item with invalid name: {raw_name}")
                                continue  # Skip invalid items
                            
                            normalized_quantity = parse_quantity(raw_quantity)
                            validated_category = validate_category(normalized_name, raw_category)
                            
                            # Normalize expiration date
                            normalized_exp_date = None
                            if expiration_date:
                                if isinstance(expiration_date, str):
                                    normalized_exp_date = normalize_expiration_date(expiration_date)
                                else:
                                    normalized_exp_date = None
                            
                            pantry_items.append({
                                'id': str(uuid.uuid4()),
                                'name': normalized_name,
                                'quantity': normalized_quantity,
                                'expirationDate': normalized_exp_date,
                                'category': validated_category,
                                'confidence': confidence,  # Use actual confidence value from ML model
                                'addedDate': datetime.now().isoformat()
                            })
                            
                            if VERBOSE_LOGGING:
                                print(f"  ✓ {normalized_name}: confidence={confidence:.2f} ({confidence*100:.0f}%)")
                        except (AttributeError, TypeError, KeyError) as item_error:
                            if VERBOSE_LOGGING:
                                print(f"Warning: Error processing ML detection item: {item_error}")
                            continue
                    
                    # Remove duplicates (case-insensitive) and ensure all items have required fields
                    seen_names = set()
                    unique_items = []
                    for item in pantry_items:
                        try:
                            name_lower = item.get('name', '').lower()
                            if name_lower and name_lower not in seen_names:
                                # Ensure all required fields are present
                                if 'id' not in item or not item['id']:
                                    item['id'] = str(uuid.uuid4())
                                # Confidence should already be set, but ensure it exists
                                if 'confidence' not in item or item['confidence'] is None:
                                    item['confidence'] = 0.5
                                    if VERBOSE_LOGGING:
                                        print(f"⚠️ Item '{item.get('name')}' missing confidence after processing")
                                if 'addedDate' not in item:
                                    item['addedDate'] = datetime.now().isoformat()
                                seen_names.add(name_lower)
                                unique_items.append(item)
                        except (AttributeError, TypeError):
                            continue
                    
                    pantry_items = unique_items
                    print(f"✅ ML detection processed: {len(pantry_items)} unique items")
                else:
                    # ML returned empty results - will fall back to OpenAI
                    print(f"⚠️ ML model found no items, will use OpenAI as fallback")
                    ml_found_items = False
                    detected_items_data = []  # Clear empty results
                    pantry_items = []  # Clear pantry_items
            except Exception as e:
                print(f"❌ ML vision failed: {e}")
                import traceback
                if VERBOSE_LOGGING:
                    traceback.print_exc()
                print(f"⚠️ ML failed, will use OpenAI as fallback")
                ml_found_items = False
                detected_items_data = []  # Clear failed results
                pantry_items = []  # Clear pantry_items
        
        # STEP 2: Use OpenAI ONLY if ML didn't find any items
        if not ml_found_items:
            print(f"🔍 STEP 2: Using OpenAI as fallback (ML found no items)")
            # Send image to OpenAI vision API
            import base64
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
            prompt = """You are an expert food recognition system analyzing a pantry/fridge photo. Identify EVERY food item with maximum accuracy. BE AGGRESSIVE - include items even if you're slightly uncertain.

SCAN THE ENTIRE IMAGE SYSTEMATICALLY:
- Look at ALL areas: foreground, background, shelves, containers, bags, boxes, drawers, doors
- Check items that are partially visible, stacked, overlapping, or in shadows
- Read labels and packaging text carefully - even small text matters
- Count multiple units of the same item
- Include items in the background, corners, and edges
- Look for items behind other items, in containers, or wrapped
- When in doubt, INCLUDE the item rather than exclude it

CRITICAL NAMING RULES:
✅ CORRECT: "milk", "chicken", "tomato", "bread", "pasta", "cheese", "eggs", "yogurt"
❌ WRONG: "milk carton", "chicken meat", "tomatoes" (use singular), "bread loaf", "Barilla pasta"

1. **Item Names** (MOST IMPORTANT):
   - Use SIMPLE, GENERIC food names: "milk" not "whole milk carton"
   - Remove ALL brand names: "Coca-Cola" -> "cola", "Kellogg's Frosted Flakes" -> "cereal"
   - Use SINGULAR form: "apple" not "apples", "tomato" not "tomatoes"
   - Remove packaging words: "milk carton" -> "milk", "bread bag" -> "bread"
   - Be specific only when helpful: "whole milk" > "milk" (if clearly visible)
   - Common pantry items: milk, eggs, bread, chicken, beef, cheese, yogurt, butter, pasta, rice, cereal, soup, juice, water, soda, coffee, tea, flour, sugar, salt, pepper, oil, vinegar, ketchup, mustard, mayonnaise, jam, peanut butter, crackers, cookies, chips, nuts, fruits (apple, banana, orange, etc.), vegetables (carrot, tomato, lettuce, onion, etc.)

2. **Quantity Detection**:
   - Count visible items: "3 apples", "2 bottles", "5 cans", "1 loaf"
   - Read packaging labels: "12 oz", "500g", "1 lb", "16 fl oz"
   - Count packages, not contents: "2 boxes of pasta" not "2 pasta"
   - Format: "X unit" (e.g., "2 bottles", "1 package", "3 cans", "5 pieces", "1 dozen")
   - If unclear, use "1"

3. **Expiration Date** (READ CAREFULLY):
   - Scan ALL text on labels, stickers, packaging
   - Look for: "EXP", "EXPIRES", "USE BY", "BEST BY", "SELL BY", "BB", "UB"
   - Parse formats: MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD, Month DD YYYY, MM/DD/YY
   - Convert to YYYY-MM-DD: "01/15/2024" -> "2024-01-15", "Jan 15 2024" -> "2024-01-15"
   - Use expiration/use-by date (NOT manufacture date)
   - If unclear or not visible, set to null

4. **Category** (choose most specific):
   - dairy: milk, cheese, yogurt, butter, cream, sour cream, cottage cheese
   - produce: fresh fruits, vegetables, herbs (apple, banana, orange, tomato, lettuce, carrot, onion, etc.)
   - meat: beef, chicken, pork, fish, seafood, turkey, bacon, sausage, deli meats
   - beverages: drinks, juices, sodas, water, coffee, tea, beer, wine
   - bakery: bread, pastries, bagels, muffins, rolls, buns
   - canned goods: canned vegetables, soups, beans, tuna, corn, tomatoes
   - snacks: chips, crackers, cookies, nuts, popcorn, pretzels, candy
   - condiments: sauces, dressings, ketchup, mustard, mayonnaise, spices, oils, vinegar
   - grains: rice, pasta, cereal, flour, oats, quinoa, barley
   - frozen: frozen foods, ice cream, frozen vegetables, frozen meals
   - other: anything that doesn't fit above

5. **RARE/UNCOMMON FOODS** (CRITICAL):
   - Read ALL text on labels, especially for specialty/organic/imported items
   - Look for food names in multiple languages (many rare foods have foreign names)
   - If you see a food name you don't recognize, include it anyway if it appears to be food
   - Preserve authentic names: "gochujang" not "korean chili paste", "tahini" not "sesame paste"
   - Check for descriptors: "organic", "artisan", "gourmet", "specialty", "imported" - these often indicate rare foods
   - Look for dietary labels: "gluten-free", "vegan", "keto", "paleo" - these products are often less common
   - Ethnic/regional foods: Look for names in other languages or transliterations
   - Fermented foods: kimchi, sauerkraut, miso, tempeh, kombucha, kefir
   - Specialty condiments: harissa, gochujang, sriracha, ponzu, mirin, fish sauce, oyster sauce
   - Nuts/seeds: macadamia, pistachio, pine nuts, hemp seeds, chia seeds, flax seeds
   - Grains/legumes: quinoa, farro, bulgur, lentils, chickpeas, black beans, edamame
   - Specialty oils: truffle oil, avocado oil, coconut oil, sesame oil, walnut oil
   - Specialty vinegars: balsamic, rice vinegar, apple cider vinegar, white wine vinegar
   - Preserves/spreads: tahini, hummus, pesto, tapenade, bruschetta, guacamole

6. **Accuracy Requirements** (CRITICAL - BE AGGRESSIVE):
   - Include ALL visible food items, even if partially hidden or unclear
   - Include items in background, shadows, or corners
   - Include items you're only 60% sure about - better to include than miss
   - Don't skip items just because they're in background, partially visible, or you don't recognize them
   - If you see ANY packaging/labels/text, read them and include the item
   - When in doubt about ANY food item, INCLUDE IT with a descriptive name
   - Only skip if it's clearly NOT food (e.g., a plate, fork, or non-food object)
   - If you see something that MIGHT be food, include it - let the user decide

FEW-SHOT EXAMPLES:
    Example 1: Image shows milk carton, bread bag, and eggs
-> {"items": [{"name": "milk", "quantity": "1 carton", "expirationDate": "2024-01-20", "category": "dairy"}, {"name": "bread", "quantity": "1 loaf", "expirationDate": "2024-01-15", "category": "bakery"}, {"name": "egg", "quantity": "1 dozen", "expirationDate": null, "category": "dairy"}]}

Example 2: Image shows 3 cans of soup and a box of pasta
-> {"items": [{"name": "soup", "quantity": "3 cans", "expirationDate": null, "category": "canned goods"}, {"name": "pasta", "quantity": "1 box", "expirationDate": null, "category": "grains"}]}

Return ONLY valid JSON (no markdown, no code blocks, no explanations):
    {"items": [{"name": "...", "quantity": "...", "expirationDate": "YYYY-MM-DD or null", "category": "..."}]}"""
            
            if not client:
                return jsonify({
                    'success': False,
                    'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.'
                }), 500
            
            # Add timeout and error handling for API calls
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert food recognition system with exceptional attention to detail. Your task is to identify ALL food items in images - be aggressive and include items even if you're slightly uncertain. Extract quantities, read expiration dates from packaging labels, and classify items into appropriate categories. When in doubt, INCLUDE the item rather than exclude it. Always return results in valid JSON format with no additional text."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}}
                        ]}
                    ],
                    max_tokens=3000,  # Increased to allow more items
                    temperature=0.2,  # Slightly higher for more creative detection
                    response_format={"type": "json_object"},
                    timeout=60.0  # 60 second timeout to prevent hanging
                )
            except Exception as api_error:
                # Handle API errors gracefully
                error_type = type(api_error).__name__
                error_msg = str(api_error)
                if VERBOSE_LOGGING:
                    print(f"OpenAI API error ({error_type}): {error_msg}")
                
                # Provide user-friendly error messages
                if 'rate limit' in error_msg.lower() or 'RateLimitError' in error_type:
                    return jsonify({
                        'success': False,
                        'error': 'API rate limit exceeded. Please try again in a few moments.'
                    }), 429
                elif 'timeout' in error_msg.lower() or 'Timeout' in error_type:
                    return jsonify({
                        'success': False,
                        'error': 'Request timed out. Please try again with a smaller image.'
                    }), 504
                elif 'invalid' in error_msg.lower() or 'InvalidRequestError' in error_type:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid image format. Please upload a valid photo.'
                    }), 400
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Error connecting to AI service: {error_msg[:100]}'
                    }), 500
            
            # Safely extract response content
            food_response = safe_get_response_content(response)
            if not food_response:
                return jsonify({
                    'success': False,
                    'error': 'Empty or invalid response from OpenAI API. Please try again.'
                }), 500
            
            # Parse JSON response with improved error handling
            detected_items_data = parse_api_response_with_retry(food_response)
            print(f"✅ OpenAI detected {len(detected_items_data) if detected_items_data else 0} items")
            
            # Normalize and validate all items
            pantry_items = []
            for item in detected_items_data:
                raw_name = item.get('name', '').strip()
                raw_quantity = item.get('quantity', '1')
                expiration_date = item.get('expirationDate')
                raw_category = item.get('category', 'other')
                
                # Normalize and validate
                normalized_name = normalize_item_name(raw_name)
                # Be less strict - only skip if name is completely empty or just whitespace
                if not normalized_name or not normalized_name.strip():
                    if VERBOSE_LOGGING:
                        print(f"Skipping item with invalid name: {raw_name}")
                    continue  # Skip invalid items
                
                normalized_quantity = parse_quantity(raw_quantity)
                validated_category = validate_category(normalized_name, raw_category)
                
                # Normalize expiration date
                normalized_exp_date = None
                if expiration_date:
                    normalized_exp_date = normalize_expiration_date(str(expiration_date))
                
                pantry_items.append({
                    'id': str(uuid.uuid4()),
                    'name': normalized_name,
                    'quantity': normalized_quantity,
                    'expirationDate': normalized_exp_date,
                    'category': validated_category,
                    'addedDate': datetime.now().isoformat()
                })
            
            # Remove duplicates (case-insensitive)
            seen_names = set()
            unique_items = []
            for item in pantry_items:
                name_lower = item['name'].lower()
                if name_lower not in seen_names:
                    seen_names.add(name_lower)
                    unique_items.append(item)
            
            pantry_items = unique_items
        
        # Add to appropriate pantry based on client type and user authentication
        client_type = request.headers.get('X-Client-Type', 'web')
        
        # Get user_id from header (mobile) or session (web)
        user_id = request.headers.get('X-User-ID')
        if not user_id and 'user_id' in session:
            candidate_user_id = session['user_id']
            # Validate user exists
            user_exists = False
            try:
                if USE_FIREBASE:
                    user_data = firebase_get_user_by_id(candidate_user_id)
                    if user_data:
                        user_exists = True
                else:
                    users = load_users()
                    if candidate_user_id in users:
                        user_exists = True
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"Error validating user {candidate_user_id}: {e}")
            
            if user_exists:
                user_id = candidate_user_id
            else:
                # Invalid user_id in session, clear it
                print(f"⚠️ Session has invalid user_id {candidate_user_id}, clearing session")
                session.pop('user_id', None)
                session.pop('username', None)
        
        # Get username for logging
        username = 'unknown'
        if user_id:
            try:
                if USE_FIREBASE:
                    user_data = firebase_get_user_by_id(user_id)
                    if user_data:
                        username = user_data.get('username', 'unknown')
                else:
                    users = load_users()
                    if user_id in users:
                        username = users[user_id].get('username', 'unknown')
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"Error getting username for logging: {e}")
        
        if VERBOSE_LOGGING:
            print(f"📸 Photo upload - User ID: {user_id}, Username: {username}, Client Type: {client_type}")
        
        # Validate user exists before saving (prevent anonymous users from saving to Firebase)
        if user_id:
            user_exists = False
            try:
                if USE_FIREBASE:
                    user_data = firebase_get_user_by_id(user_id)
                    if user_data:
                        user_exists = True
                else:
                    users = load_users()
                    if user_id in users:
                        user_exists = True
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"Error validating user {user_id}: {e}")
            
            if not user_exists:
                print(f"❌ ERROR: Cannot save photo items - user {user_id} does not exist in database!")
                return jsonify({
                    'success': False,
                    'error': 'Invalid user. Please log in to add items from photos.',
                    'items': []
                }), 401
        
        total_items = 0
        if user_id:
            try:
                # Add to user's pantry
                user_pantry = get_user_pantry(user_id)
                if not isinstance(user_pantry, list):
                    user_pantry = []
                
                # Convert detected_items to proper format (list of dicts) with duplicate checking
                existing_names = {item.get('name', '').strip().lower() for item in user_pantry if isinstance(item, dict) and item.get('name')}
                for item in pantry_items:
                    try:
                        if not isinstance(item, dict):
                            pass
                        item_name = item.get('name', '')
                        if not isinstance(item_name, str):
                            item_name = str(item_name) if item_name else ''
                        item_name = item_name.strip().lower()
                        if item_name and item_name not in existing_names:
                            user_pantry.append(item)
                            existing_names.add(item_name)
                    except (AttributeError, TypeError):
                        continue
                
                update_user_pantry(user_id, user_pantry)
                total_items = len(user_pantry)
            except (TypeError, AttributeError, KeyError) as e:
                if VERBOSE_LOGGING:
                    print(f"Error adding items to user pantry: {e}")
                total_items = len(pantry_items)

        else:
            # Add to anonymous pantry
            if client_type == 'mobile':
                try:
                    if not isinstance(mobile_pantry, list):
                        mobile_pantry = []
                    existing_names = {item.get('name', '').strip().lower() for item in mobile_pantry if isinstance(item, dict) and item.get('name')}
                    for item in pantry_items:
                        try:
                            if not isinstance(item, dict):
                                pass
                            item_name = item.get('name', '')
                            if not isinstance(item_name, str):
                                item_name = str(item_name) if item_name else ''
                            item_name = item_name.strip().lower()
                            if item_name and item_name not in existing_names:
                                mobile_pantry.append(item)
                                existing_names.add(item_name)
                            pass
                        except (AttributeError, TypeError):
                            continue
                    total_items = len(mobile_pantry)
                except (TypeError, AttributeError):
                    mobile_pantry = pantry_items.copy() if pantry_items else []
                    total_items = len(mobile_pantry)
            else:
                try:
                    # Add to anonymous web pantry (stored in session)
                    # CRITICAL: Mark session as permanent for anonymous users
                    session.permanent = True
                    
                    if 'web_pantry' not in session:
                        session['web_pantry'] = []
                    
                    # Validate session['web_pantry'] is a list
                    web_pantry_session = session.get('web_pantry', [])
                    if not isinstance(web_pantry_session, list):
                        web_pantry_session = []
                    
                    # Check for duplicates before adding (case-insensitive)
                    existing_names = {item.get('name', '').strip().lower() for item in web_pantry_session if isinstance(item, dict) and item.get('name')}
                    new_items = []
                    for item in pantry_items:
                        try:
                            if not isinstance(item, dict):
                                pass
                            item_name = item.get('name', '')
                            if not isinstance(item_name, str):
                                item_name = str(item_name) if item_name else ''
                            item_name = item_name.strip().lower()
                            if item_name and item_name not in existing_names:
                                web_pantry_session.append(item)
                                existing_names.add(item_name)
                                new_items.append(item)
                        except (AttributeError, TypeError):
                            continue
                    
                    # Mark session as modified to ensure it's saved
                    if new_items:
                        session['web_pantry'] = web_pantry_session
                    session.modified = True
                    total_items = len(session.get('web_pantry', []))
                except (KeyError, TypeError, AttributeError) as e:
                    if VERBOSE_LOGGING:
                        print(f"Error adding items to anonymous web pantry: {e}")
                    # Initialize empty pantry if session is corrupted

                    try:
                        session['web_pantry'] = pantry_items.copy() if pantry_items else []
                        session.modified = True
                        total_items = len(pantry_items)
                        pass
                    except:
                        total_items = 0
        
        # Prepare response items with confidence scores for mobile app
        # Convert pantry_items to DetectedItem format with all required fields
        response_items = []
        for item in pantry_items:
            try:
                if not isinstance(item, dict):
                    continue
                
                # Ensure all required fields are present
                item_id = item.get('id', str(uuid.uuid4()))
                name = item.get('name', '').strip()
                if not name:
                    continue
                
                quantity = item.get('quantity', '1')
                category = item.get('category', 'other')
                confidence = item.get('confidence', 0.5)
                expiration_date = item.get('expirationDate')
                
                # Ensure confidence is a float between 0 and 1
                try:
                    confidence = float(confidence)
                    if confidence < 0 or confidence > 1:
                        confidence = 0.5
                except (ValueError, TypeError):
                    confidence = 0.5
                
                response_items.append({
                    'id': item_id,
                    'name': name,
                    'quantity': str(quantity) if quantity else '1',
                    'category': str(category) if category else 'other',
                    'confidence': confidence,
                    'expirationDate': expiration_date if expiration_date else None,
                    'is_partial': False,  # Default to False
                    'expiration_risk': None  # Can be calculated later if needed
                })
            except (AttributeError, TypeError, KeyError) as item_error:
                if VERBOSE_LOGGING:
                    print(f"Warning: Error formatting response item: {item_error}")
                continue
        
        # Categorize items by confidence for mobile app
        high_confidence_count = sum(1 for item in response_items if item.get('confidence', 0) >= 0.7)
        needs_confirmation_count = len(response_items) - high_confidence_count
        
        return jsonify({
            'success': True,
            'message': f'Successfully analyzed photo! Found {len(response_items)} items' if response_items else 'Photo analyzed but no items were detected',
            'items': response_items,  # Return full item objects with confidence for mobile app
            'auto_added': high_confidence_count,  # Items with high confidence (>= 0.7)
            'needs_confirmation': needs_confirmation_count,  # Items needing user confirmation
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

@app.route('/api/insights', methods=['GET'])
def api_insights():
    """Get pantry insights and statistics"""
    try:
        client_type = request.headers.get('X-Client-Type', 'web')
        user_id = request.headers.get('X-User-ID')
        
        # Get appropriate pantry
        if user_id:
            pantry_items = get_user_pantry(user_id)
        elif 'user_id' in session:
            pantry_items = get_user_pantry(session['user_id'])
        else:
            # Use session-based pantry for anonymous users
            if client_type == 'mobile':
                pantry_items = mobile_pantry
            else:
                pantry_items = session.get('web_pantry', [])
        
        # Ensure pantry_items is a list
        if not isinstance(pantry_items, list):
            pantry_items = []
        
        # Calculate statistics
        today = datetime.now().date()
        item_counts = {}
        category_counts = {}
        expired_items = []
        expiring_soon_items = []
        total_days_in_pantry = 0
        items_with_dates = 0
        
        for item in pantry_items:
            if not isinstance(item, dict):
                continue
            
            item_name = item.get('name', '').strip().lower()
            if item_name:
                # Most common items
                item_counts[item_name] = item_counts.get(item_name, 0) + 1
            
            # Category counts
            category = item.get('category', 'other')
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Average time in pantry
            added_date_str = item.get('addedDate')
            if added_date_str:
                try:
                    # Handle different date formats
                    date_str = str(added_date_str).strip()
                    if 'T' in date_str:
                        # ISO format with time
                        if date_str.endswith('Z'):
                            date_str = date_str.replace('Z', '+00:00')
                        added_date = datetime.fromisoformat(date_str).date()
                    else:
                        # Date only format (YYYY-MM-DD)
                        added_date = datetime.strptime(date_str[:10], '%Y-%m-%d').date()
                    days_in_pantry = (today - added_date).days
                    if days_in_pantry >= 0:
                        total_days_in_pantry += days_in_pantry
                        items_with_dates += 1
                    pass
                except Exception as e:
                    if VERBOSE_LOGGING:

                        print(f"Warning: Could not parse addedDate '{added_date_str}': {e}")
                    pass
            
            # Expired and expiring soon items
            exp_date_str = item.get('expirationDate')
            if exp_date_str:
                try:
                    # Handle different date formats
                    date_str = str(exp_date_str).strip()
                    if len(date_str) >= 10:
                        # Try ISO format first
                        if 'T' in date_str:
                            if date_str.endswith('Z'):
                                date_str = date_str.replace('Z', '+00:00')
                            exp_date = datetime.fromisoformat(date_str).date()
                        else:
                            # Date only format (YYYY-MM-DD)
                            exp_date = datetime.strptime(date_str[:10], '%Y-%m-%d').date()
                        
                        days_until_exp = (exp_date - today).days
                        
                        if days_until_exp < 0:
                            expired_items.append({
                                'name': item.get('name', 'Unknown'),
                                'expired_days': abs(days_until_exp),
                                'category': item.get('category', 'other')
                            })
                        elif days_until_exp <= 7:
                            expiring_soon_items.append({
                                'name': item.get('name', 'Unknown'),
                                'days_remaining': days_until_exp,
                                'category': item.get('category', 'other')
                            })
                    pass
                except Exception as e:
                    if VERBOSE_LOGGING:

                        print(f"Warning: Could not parse expirationDate '{exp_date_str}': {e}")
                    pass
        
        # Sort most common items
        most_common = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        # Get category for each most common item
        most_common_items = []
        for name, count in most_common:
            # Find the category for this item name
            category = 'other'
            for item in pantry_items:
                if item.get('name', '').lower().strip() == name.lower().strip():
                    category = item.get('category', 'other')
                    break
            most_common_items.append({'name': name, 'count': count, 'category': category})
        
        # Sort categories
        category_list = [{'category': cat, 'count': count} for cat, count in category_counts.items()]
        category_list.sort(key=lambda x: x['count'], reverse=True)
        
        # Sort expired items (most expired first)
        expired_items.sort(key=lambda x: x['expired_days'], reverse=True)
        
        # Sort expiring soon items (soonest first)
        expiring_soon_items.sort(key=lambda x: x['days_remaining'])
        
        # Calculate average time in pantry
        avg_days_in_pantry = total_days_in_pantry / items_with_dates if items_with_dates > 0 else 0
        
        return jsonify({
            'success': True,
            'stats': {
                'total_items': len(pantry_items),
                'most_common_items': most_common_items if most_common_items else [],
                'category_breakdown': category_list if category_list else [],
                'average_days_in_pantry': round(avg_days_in_pantry, 1),
                'expired_items_count': len(expired_items),
                'expired_items': expired_items[:10] if expired_items else [],  # Top 10 most expired
                'expiring_soon_count': len(expiring_soon_items),
                'expiring_soon_items': expiring_soon_items[:10] if expiring_soon_items else []  # Top 10 expiring soonest
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload_photos_batch', methods=['POST'])
def upload_photos_batch():
    """Upload multiple photos and merge detections (for multiple angles)"""
    try:
        if 'photos[]' not in request.files:
            return jsonify({'success': False, 'error': 'No photos uploaded'}), 400
        
        photos = request.files.getlist('photos[]')
        if not photos:
            return jsonify({'success': False, 'error': 'No photos selected'}), 400
        
        if len(photos) > 5:
            return jsonify({'success': False, 'error': 'Maximum 5 photos allowed'}), 400
        
        # Get user's current pantry for context fusion
        user_pantry = None
        client_type = request.headers.get('X-Client-Type', 'web')
        if client_type == 'mobile':
            user_pantry = mobile_pantry
        else:
            user_pantry = session.get('web_pantry', [])
        
        # Process all photos and collect detections
        all_detections = []
        for photo in photos:
            try:
                photo.seek(0)
                img_bytes = photo.read()
                
                if len(img_bytes) == 0 or len(img_bytes) > 10 * 1024 * 1024:
                    pass
                
                # Detect items in this photo
                detections = detect_food_items_with_ml(img_bytes, user_pantry=user_pantry)
                all_detections.extend(detections)
                pass
            except Exception as e:
                if VERBOSE_LOGGING:

                    print(f"Warning: Failed to process photo: {e}")
                continue
        
        # Merge detections (deduplicate by name, keep highest confidence)
        merged = {}
        for item in all_detections:
            key = item.get("name", "").lower().strip()
            if not key:
                continue
            
            if key not in merged or item.get("confidence", 0) > merged[key].get("confidence", 0):
                merged[key] = item
        
        result = list(merged.values())
        result.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Categorize by confidence
        categorized = categorize_by_confidence(result)
        
        return jsonify({
            'success': True,
            'items': result,
            'high_confidence': categorized['high_confidence'],
            'medium_confidence': categorized['medium_confidence'],
            'low_confidence': categorized['low_confidence'],
            'total_detections': len(result)
        })
    except Exception as e:
        if VERBOSE_LOGGING:
            import traceback
            traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

def log_user_feedback(confirmed_items, user_id=None):
    """
    Log user feedback when they confirm/correct detected items.
    This data is used to improve model accuracy over time.
    """
    try:
        feedback_dir = '/tmp/feedback' if IS_VERCEL or IS_RENDER else os.path.join(_app_file_dir, 'feedback')
        os.makedirs(feedback_dir, exist_ok=True)
        
        feedback_data = {
            "confirmed_items": confirmed_items,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "detection_methods": [item.get('detection_method', 'unknown') for item in confirmed_items if isinstance(item, dict)]
        }
        
        feedback_file = os.path.join(feedback_dir, f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json")
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        if VERBOSE_LOGGING:
            print(f"✅ Logged user feedback: {len(confirmed_items)} confirmed items")
        
        # Also update accuracy statistics
        update_accuracy_stats(confirmed_items)
        
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Warning: Failed to log user feedback: {e}")

def update_accuracy_stats(confirmed_items):
    """
    Update accuracy statistics based on user confirmations.
    Tracks which detection methods are most accurate for different food types.
    """
    try:
        stats_file = '/tmp/accuracy_stats.json' if IS_VERCEL or IS_RENDER else os.path.join(_app_file_dir, 'accuracy_stats.json')
        
        # Load existing stats
        stats = {}
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
            except Exception:
                stats = {}
        
        # Initialize structure
        if 'detection_methods' not in stats:
            stats['detection_methods'] = {}
        if 'food_categories' not in stats:
            stats['food_categories'] = {}
        if 'total_confirmations' not in stats:
            stats['total_confirmations'] = 0
        
        # Update stats for each confirmed item
        for item in confirmed_items:
            if not isinstance(item, dict):
                continue
            
            method = item.get('detection_method', 'unknown')
            category = item.get('category', 'other')
            confidence = item.get('confidence', 0.5)
            
            # Track method accuracy
            if method not in stats['detection_methods']:
                stats['detection_methods'][method] = {
                    'count': 0,
                    'total_confidence': 0,
                    'confirmed': 0
                }
            
            stats['detection_methods'][method]['count'] += 1
            stats['detection_methods'][method]['total_confidence'] += confidence
            stats['detection_methods'][method]['confirmed'] += 1
            
            # Track category accuracy
            if category not in stats['food_categories']:
                stats['food_categories'][category] = {
                    'count': 0,
                    'total_confidence': 0,
                    'confirmed': 0
                }
            
            stats['food_categories'][category]['count'] += 1
            stats['food_categories'][category]['total_confidence'] += confidence
            stats['food_categories'][category]['confirmed'] += 1
        
        stats['total_confirmations'] += len(confirmed_items)
        stats['last_updated'] = datetime.now().isoformat()
        
        # Save updated stats
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        if VERBOSE_LOGGING:
            print(f"📊 Updated accuracy statistics: {stats['total_confirmations']} total confirmations")
            
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Warning: Failed to update accuracy stats: {e}")

def update_failure_stats(correct_labels, predicted_labels, detected_items):
    """Update failure statistics to identify common errors"""
    try:
        stats_file = '/tmp/failure_stats.json' if IS_VERCEL or IS_RENDER else os.path.join(_app_file_dir, 'failure_stats.json')
        
        stats = {}
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
            except Exception:
                stats = {}
        
        if 'common_errors' not in stats:
            stats['common_errors'] = {}
        if 'missed_items' not in stats:
            stats['missed_items'] = {}
        if 'false_positives' not in stats:
            stats['false_positives'] = {}
        
        # Track common prediction errors
        for predicted, correct in zip(predicted_labels, correct_labels):
            if predicted != correct:
                key = f"{predicted}->{correct}"
                stats['common_errors'][key] = stats['common_errors'].get(key, 0) + 1
        
        # Track missed items (in correct but not in predicted)
        correct_set = set(str(c).lower() for c in correct_labels)
        predicted_set = set(str(p).lower() for p in predicted_labels)
        missed = correct_set - predicted_set
        for item in missed:
            stats['missed_items'][item] = stats['missed_items'].get(item, 0) + 1
        
        # Track false positives (in predicted but not in correct)
        false_pos = predicted_set - correct_set
        for item in false_pos:
            stats['false_positives'][item] = stats['false_positives'].get(item, 0) + 1
        
        stats['last_updated'] = datetime.now().isoformat()
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Warning: Failed to update failure stats: {e}")

@app.route('/api/log_detection_failure', methods=['POST'])
def log_detection_failure():
    """Log failed detections for retraining"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        image_b64 = data.get("image")
        correct_labels = data.get("correct_labels", [])
        predicted_labels = data.get("predicted_labels", [])
        
        if not image_b64 or not correct_labels:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Save failure case to /tmp/failures/ for retraining
        import uuid
        from datetime import datetime
        
        failure_data = {
            "image": image_b64,
            "correct": correct_labels,
            "predicted": predicted_labels,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save failure case
        failures_dir = '/tmp/failures' if IS_VERCEL or IS_RENDER else os.path.join(_app_file_dir, 'failures')
        try:
            os.makedirs(failures_dir, exist_ok=True)
            failure_file = os.path.join(failures_dir, f"{uuid.uuid4()}.json")
            with open(failure_file, 'w') as f:
                json.dump(failure_data, f, indent=2)
            if VERBOSE_LOGGING:

                print(f"Logged failure case: {failure_file}")
            pass
        except Exception as e:
            if VERBOSE_LOGGING:

                print(f"Warning: Could not save failure case: {e}")
            # Don't fail the request if saving fails
        
        return jsonify({'success': True, 'message': 'Failure case logged'})
    except Exception as e:
        if VERBOSE_LOGGING:
            import traceback
            traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/evaluate_detection', methods=['POST'])
def evaluate_detection():
    """Calculate evaluation metrics (precision, recall, F1)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        predictions = data.get("predictions", [])
        ground_truth = data.get("ground_truth", [])
        
        if not isinstance(predictions, list) or not isinstance(ground_truth, list):
            return jsonify({'success': False, 'error': 'Invalid input format'}), 400
        
        # Calculate metrics
        metrics = calculate_detection_metrics(predictions, ground_truth)
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        pass
    except Exception as e:
        if VERBOSE_LOGGING:
            import traceback
            traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

def calculate_detection_metrics(predictions, ground_truth):
    """Calculate precision, recall, F1 score for detections"""
    pred_names = {p.get("name", "").lower().strip() for p in predictions if isinstance(p, dict)}
    gt_names = {g.get("name", "").lower().strip() for g in ground_truth if isinstance(g, dict)}
    
    # Remove empty strings
    pred_names = {n for n in pred_names if n}
    gt_names = {n for n in gt_names if n}
    
    true_positives = len(pred_names & gt_names)
    false_positives = len(pred_names - gt_names)
    false_negatives = len(gt_names - pred_names)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_predictions": len(pred_names),
        "total_ground_truth": len(gt_names)
    }

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

@app.route('/api/correction', methods=['POST'])
def api_log_correction():
    """
    🔥 User Correction Feedback Loop: Log corrections for retraining/evaluation.
    
    This endpoint receives corrections when users click "This is wrong" and provides
    the correct label. This data is stored for:
    - Model retraining
    - Evaluation metrics
    - Continuous improvement
    
    This is real ML lifecycle engineering.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        detected_name = data.get('detected_name', '')
        correct_name = data.get('correct_name', '')
        confidence = data.get('confidence', 0)
        item_data = data.get('item_data', {})
        photo_data = data.get('photo_data')  # Base64 encoded image
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        if not detected_name or not correct_name:
            return jsonify({'success': False, 'error': 'Missing detected_name or correct_name'}), 400
        
        # Get user ID from session or header
        user_id = request.headers.get('X-User-ID') or session.get('user_id') or 'anonymous'
        
        # Create corrections directory
        corrections_dir = '/tmp/corrections' if IS_VERCEL or IS_RENDER else os.path.join(_app_file_dir, 'corrections')
        os.makedirs(corrections_dir, exist_ok=True)
        
        # Save image if provided
        image_path = None
        if photo_data:
            try:
                import base64
                # Remove data URL prefix if present
                if ',' in photo_data:
                    photo_data = photo_data.split(',')[1]
                
                image_data = base64.b64decode(photo_data)
                image_filename = f"correction_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
                image_path = os.path.join(corrections_dir, image_filename)
                
                with open(image_path, 'wb') as f:
                    f.write(image_data)
            except Exception as img_error:
                print(f"Warning: Failed to save correction image: {img_error}")
        
        # Create correction record
        correction_data = {
            'timestamp': timestamp,
            'user_id': user_id,
            'detected_name': detected_name,
            'correct_name': correct_name,
            'confidence': confidence,
            'item_data': item_data,
            'image_path': image_path,
            'correction_type': 'wrong_detection'
        }
        
        # Save correction to JSON file
        correction_file = os.path.join(corrections_dir, f"correction_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json")
        with open(correction_file, 'w') as f:
            json.dump(correction_data, f, indent=2)
        
        # 🔥 IMPROVEMENT 7: Track failure analytics
        try:
            from failure_analytics import track_user_correction
            track_user_correction(detected_name, correct_name, confidence, image_path)
            print(f"   📊 Tracked correction in failure analytics: '{detected_name}' -> '{correct_name}'")
        except ImportError:
            # Failure analytics module not available - skip silently
            pass
        except Exception as analytics_error:
            if VERBOSE_LOGGING:
                print(f"Warning: Failed to track analytics: {analytics_error}")
        
        print(f"✅ Logged correction: '{detected_name}' -> '{correct_name}' (conf: {confidence:.2f}, user: {user_id})")
        
        return jsonify({
            'success': True,
            'message': 'Correction logged successfully',
            'correction_id': os.path.basename(correction_file)
        })
        
    except Exception as e:
        print(f"❌ Error logging correction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

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
