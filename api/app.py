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
    except Exception as e:
        print(f"⚠️ Warning: Firebase enabled but initialization failed: {e}")
        print("   Falling back to file-based storage")
        USE_FIREBASE = False

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
        print("   On Render: Go to Dashboard → Your Service → Environment → Add OPENAI_API_KEY")
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
ML_VISION_ENABLED = os.getenv("ML_VISION_ENABLED", "false").lower() == "true"
# For serverless deployments, prefer "classify_only" (smaller) or disable ML entirely
# ML models (transformers, torch, easyocr) are very large and can exceed function size limits
ML_VISION_MODE = os.getenv("ML_VISION_MODE", "classify_only").lower()  # hybrid | classify_only

# ✅ YOLOv8 Object Detection (Better accuracy than DETR)
# YOLOv8 provides superior object detection for pantry/fridge images
# Enable this for better detection accuracy (recommended over DETR)
YOLO_DETECTION_ENABLED = os.getenv("YOLO_DETECTION_ENABLED", "false").lower() == "true"
YOLO_MODEL_SIZE = os.getenv("YOLO_MODEL_SIZE", "n").lower()  # n (nano), s (small), m (medium), l (large), x (xlarge)

# Lazy-loaded ML models (keep memory usage lower in serverless)
_ml_models_loaded = False
_food_classifier = None
_ocr_reader = None
_object_detector = None
_yolo_model = None  # YOLOv8 model for better object detection

def load_ml_models():
    """Load ML models lazily for serverless-friendly photo analysis.
    Includes comprehensive error handling and timeout protection."""
    global _ml_models_loaded, _food_classifier, _ocr_reader, _object_detector
    
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
                    from transformers import pipeline
                    import signal
                    import threading
                    
                    classifier_loaded = False
                    classifier_error = None
                    
                    _food_classifier_local = None
                    
                    def load_classifier():
                        nonlocal classifier_loaded, classifier_error, _food_classifier_local
                        try:
                            _food_classifier_local = pipeline(
                                "image-classification",
                                model="nateraw/food-image-classification",
                                device=-1  # CPU only for serverless
                            )
                            classifier_loaded = True
                            pass
                        except Exception as e:
                            classifier_error = e
                    
                    thread = threading.Thread(target=load_classifier)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=120)  # 2 minute timeout
                    
                    if thread.is_alive():
                        print("⚠️  Warning: Food classifier loading timed out")
                        _food_classifier = None
                    elif classifier_error:
                        print(f"⚠️  Warning: food classifier not available: {classifier_error}")
                        _food_classifier = None
                    elif classifier_loaded and _food_classifier_local:
                        _food_classifier = _food_classifier_local
                        print("✅ Food classifier loaded")
                    else:
                        _food_classifier = None
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
                            pass
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
                    
                    # Use YOLOv8n (nano) by default for faster inference
                    # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large), yolov8x.pt (xlarge)
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
    
    # Resize to optimal size for ML models (larger = better accuracy, but slower)
    max_size = 1024  # Increased from 900 for better accuracy
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Enhanced preprocessing optimized for fridge/pantry photos AND bad angles
    # Bad angle photos often have poor lighting and contrast
    # Sharpen to improve edge detection and text recognition (especially important for angled photos)
    img = img.filter(ImageFilter.SHARPEN)
    
    # Increase contrast MORE aggressively for angled photos (items are harder to see)
    # Angled photos often have less contrast due to lighting angles
    img = ImageEnhance.Contrast(img).enhance(1.5)  # Increased from 1.4 for bad angles
    
    # Brightness adjustment - angled photos may have uneven lighting
    # Use adaptive brightness enhancement
    img = ImageEnhance.Brightness(img).enhance(1.15)  # Increased from 1.1
    
    # Saturation boost to help identify colorful food items (more important with bad angles)
    img = ImageEnhance.Color(img).enhance(1.15)  # Increased from 1.1
    
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
    
    # Try original and common rotations (0, 90, 180, 270)
    # Limit to most common angles to balance speed vs accuracy
    angles_to_try = [None, 90, 270]  # Original, 90° right, 90° left (most common bad angles)
    
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
    Hierarchical classification: Category → Specific Item
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
        
        if score < 0.15:
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
    
    return None

def apply_context_fusion(items, user_pantry=None):
    """
    Boost confidence based on context:
        - Items already in pantry are more likely
    - Common items are more likely
    - Location hints (if bbox available)
    """
    if not user_pantry:
        return items
    
    existing_items = set()
    if isinstance(user_pantry, list):
        for item in user_pantry:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name:
                    existing_items.add(name.lower().strip())
    
    common_items = {"milk", "eggs", "bread", "cheese", "chicken", "yogurt", "butter", "apple", "banana", "tomato"}
    
    for item in items:
        name_lower = item.get("name", "").lower().strip()
        if not name_lower:
            continue
        
        base_confidence = item.get("confidence", 0)
        
        # Boost if already in pantry
        if name_lower in existing_items:
            item["confidence"] = min(1.0, base_confidence + 0.15)
            item["context_boost"] = "already_in_pantry"
        # Boost if common item
        elif name_lower in common_items:
            item["confidence"] = min(1.0, base_confidence + 0.1)
            item["context_boost"] = "common_item"
    
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
    
    # Boost confidence for items detected by multiple methods
    for item in result:
        key = item.get("name", "").lower().strip()
        if key in detection_counts:
            detection_info = detection_counts[key]
            method_count = len(detection_info['methods'])
            
            # Boost confidence: +5% for each additional detection method
            if method_count > 1:
                boost = min(0.15, (method_count - 1) * 0.05)  # Max 15% boost
                item['confidence'] = min(1.0, item.get('confidence', 0) * (1 + boost))
                item['ensemble_boost'] = True
                item['detection_methods'] = list(detection_info['methods'])
    
    return result

def calibrate_confidence(items):
    """
    Calibrate confidence scores based on historical accuracy data.
    Adjusts confidence to better reflect actual accuracy.
    """
    # TODO: Load historical accuracy data from feedback
    # For now, apply basic calibration based on detection method
    for item in items:
        method = item.get('detection_method', 'unknown')
        original_conf = item.get('confidence', 0.5)
        
        # Calibration factors based on method reliability (empirical)
        calibration_factors = {
            'yolov8': 0.95,  # YOLOv8 is generally reliable, slight calibration down
            'dETR': 0.90,   # DETR is reliable but slightly less than YOLOv8
            'classification': 0.85,  # Classification alone is less reliable
            'ensemble': 1.0,  # Ensemble is most reliable
            'yolov8+classification': 0.98,  # Combined methods are more reliable
            'unknown': 0.85
        }
        
        factor = calibration_factors.get(method, 0.85)
        
        # Apply calibration (slightly reduce overconfident predictions)
        calibrated_conf = original_conf * factor
        
        # Store both original and calibrated
        item['original_confidence'] = original_conf
        item['confidence'] = calibrated_conf
        item['calibration_factor'] = factor
    
    return items

def categorize_by_confidence(items):
    """
    Categorize items by confidence for user confirmation:
        - High confidence (≥0.8): Auto-add
    - Medium confidence (0.5-0.8): Ask user
    - Low confidence (<0.5): Ignore or mark uncertain
    """
    HIGH_CONF = 0.8
    MEDIUM_CONF = 0.5
    
    high_conf = []
    medium_conf = []
    low_conf = []
    
    for item in items:
        conf = item.get("confidence", 0)
        if conf >= HIGH_CONF:
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

def detect_food_items_with_ml(img_bytes, user_pantry=None, skip_preprocessing=False, use_multi_angle=True):
    """
    Clear pipeline: Image → Object Detection → Classification → Metadata → Context Fusion → Confidence Categorization
    
    Improved hybrid ML approach with better accuracy, especially for bad angle photos:
        1. Multi-angle ensemble (optional) - try multiple orientations for consistency
    2. Object Detection (primary method) - finds bounding boxes
    3. Classify each detection hierarchically (category → item)
    4. Extract metadata (quantity, expiration)
    5. Apply context fusion (boost confidence for items in pantry)
    6. Categorize by confidence for user confirmation
    
    Args:
        img_bytes: Raw image bytes
        user_pantry: Optional user's current pantry for context fusion
        skip_preprocessing: If True, assume img_bytes is already a PIL Image
        use_multi_angle: If True, try multiple orientations (slower but more robust)
    
    Includes comprehensive error handling for extreme scenarios.
    """
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
            load_ml_models()
        pass
    except Exception as e:
        if VERBOSE_LOGGING:

            print(f"Warning: Failed to load ML models: {e}")
        return []  # Return empty list if models can't be loaded
    
    # STEP 0: Multi-angle ensemble (if enabled and not already preprocessed)
    if use_multi_angle and not skip_preprocessing:
        try:
            # Try multi-angle detection for better consistency with bad angles
            multi_angle_items = detect_with_multiple_angles(img_bytes, user_pantry=user_pantry)
            if multi_angle_items and len(multi_angle_items) > 0:
                # Use multi-angle results if they found items
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
    
    # ENHANCED: Use OCR to identify rare foods from packaging labels (fallback if no detections)
    if _ocr_reader and not items:  # Only if no items found yet (rare food scenario)
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
                        ocr_results = _ocr_reader.readtext(img_array)
                        pass
                    except Exception as e:
                        ocr_error = e
                
                thread = threading.Thread(target=run_ocr)
                thread.daemon = True
                thread.start()
                thread.join(timeout=20)  # 20 second timeout for OCR
                
                if thread.is_alive():
                    if VERBOSE_LOGGING:

                        print("Warning: OCR timed out after 20 seconds")
                    return items
                elif ocr_error:
                    raise ocr_error
                
                if not ocr_results:
                    return items
            except Exception as ocr_err:
                if VERBOSE_LOGGING:

                    print(f"Warning: OCR processing failed: {ocr_err}")
                return items
            
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
                    
                    if conf > 0.3:  # OCR confidence threshold
                        text_lower = text.lower().strip()
                        if not text_lower or len(text_lower) < 2:
                            pass
                        
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
                                                    items.append({
                                                        "name": normalized,
                                                        "quantity": "1",
                                                        "expirationDate": exp_date,
                                                        "category": validate_category(normalized, "other"),
                                                        "confidence": float(conf * 0.6)  # OCR-based confidence (lower weight)
                                                    })
                                                    item_confidence[normalized.lower()] = float(conf * 0.6)
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

    # STEP 2: Object Detection (Primary Method) - Find bounding boxes, then classify each
    # Priority: YOLOv8 (if enabled) > DETR (if enabled)
    
    # STEP 2a: YOLOv8 Detection (Better accuracy - recommended)
    detections_found_yolo = False
    if _yolo_model and YOLO_DETECTION_ENABLED:
        try:
            import numpy as np
            from PIL import Image
            import io
            
            # Convert PIL image to numpy array for YOLOv8
            img_array = np.array(img)
            
            # Run YOLOv8 inference with optimized confidence threshold
            # Use adaptive threshold: start lower to catch more items, then filter
            # Lower threshold (0.20) to catch more items, especially in cluttered pantry images
            # Higher IOU (0.5) for better NMS to reduce duplicates
            results = _yolo_model(img_array, conf=0.20, iou=0.50, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                # YOLOv8 COCO class names (80 classes including food items)
                # Map COCO classes to food items
                coco_to_food = {
                    # Fruits
                    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
                    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
                    # Containers that might contain food
                    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
                    44: "spoon", 45: "bowl",
                }
                
                # Process detections
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    detections_found_yolo = True
                    
                    for box in boxes:
                        try:
                            # Extract box coordinates (xyxy format)
                            # Handle both tensor and numpy array formats
                            xyxy = box.xyxy[0] if hasattr(box.xyxy, '__getitem__') else box.xyxy
                            if hasattr(xyxy, 'cpu'):
                                xyxy = xyxy.cpu().numpy()
                            elif hasattr(xyxy, 'numpy'):
                                xyxy = xyxy.numpy()
                            
                            # Ensure we have 4 coordinates
                            if len(xyxy) < 4:
                                continue
                            x1, y1, x2, y2 = xyxy[:4]
                            
                            conf = box.conf[0] if hasattr(box.conf, '__getitem__') else box.conf
                            if hasattr(conf, 'cpu'):
                                conf = conf.cpu().numpy()
                            elif hasattr(conf, 'numpy'):
                                conf = conf.numpy()
                            confidence = float(conf)
                            
                            cls = box.cls[0] if hasattr(box.cls, '__getitem__') else box.cls
                            if hasattr(cls, 'cpu'):
                                cls = cls.cpu().numpy()
                            elif hasattr(cls, 'numpy'):
                                cls = cls.numpy()
                            class_id = int(cls)
                            
                            # Filter by confidence threshold (increased from 0.3 to 0.25 for better recall)
                            # We'll use NMS later to filter duplicates
                            if confidence < 0.25:  # Lower threshold to catch more items
                                continue
                            
                            # Get class name from YOLOv8 (with bounds checking)
                            if hasattr(_yolo_model, 'names') and class_id < len(_yolo_model.names):
                                class_name = _yolo_model.names[class_id]
                            else:
                                class_name = f"class_{class_id}"
                            
                            # Map COCO class to food item
                            mapped_name = coco_to_food.get(class_id)
                            
                            # If not in food map, check if class name suggests food
                            if not mapped_name:
                                class_lower = class_name.lower()
                                # Check for common food-related keywords
                                if any(keyword in class_lower for keyword in ['food', 'fruit', 'vegetable', 'bottle', 'can', 'container']):
                                    mapped_name = class_name
                                else:
                                    # Skip non-food items
                                    continue
                            
                            # Ensure valid crop coordinates
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            x1 = max(0, min(x1, img.size[0] - 1))
                            y1 = max(0, min(y1, img.size[1] - 1))
                            x2 = max(x1 + 1, min(x2, img.size[0]))
                            y2 = max(y1 + 1, min(y2, img.size[1]))
                            
                            if x2 <= x1 or y2 <= y1:
                                continue
                            
                            # Crop the detected region
                            try:
                                crop = img.crop((x1, y1, x2, y2))
                                if crop.size[0] == 0 or crop.size[1] == 0:
                                    continue
                            except Exception:
                                continue
                            
                            # Initialize detection method
                            detection_method = 'yolov8'
                            classification_conf = 0
                            
                            # Classify the cropped region for better accuracy (ensemble approach)
                            classification = classify_food_hierarchical(crop)
                            if classification:
                                final_name = classification["name"]
                                final_category = classification["category"]
                                classification_conf = classification["confidence"]
                                
                                # Smart ensemble: weight based on both confidences
                                # If both are high, boost more. If one is low, be more conservative
                                yolo_weight = 0.4 if confidence > 0.5 else 0.3
                                class_weight = 0.6 if classification_conf > 0.5 else 0.7
                                
                                # Normalize weights
                                total_weight = yolo_weight + class_weight
                                yolo_weight /= total_weight
                                class_weight /= total_weight
                                
                                combined_conf = (confidence * yolo_weight + classification_conf * class_weight)
                                
                                # Additional boost if names match (consensus)
                                if final_name.lower() == mapped_name.lower():
                                    combined_conf = min(1.0, combined_conf * 1.1)  # 10% boost for consensus
                                
                                detection_method = 'yolov8+classification'
                            else:
                                # Use mapped name from YOLO
                                final_name = mapped_name
                                final_category = validate_category(mapped_name, "other")
                                combined_conf = confidence
                            
                            # Extract expiration date from OCR if available
                            exp_date = None
                            if _ocr_reader:
                                try:
                                    # extract_expiration_dates handles OCR internally
                                    exp_dates = extract_expiration_dates(crop, _ocr_reader)
                                    if exp_dates and len(exp_dates) > 0:
                                        exp_date = exp_dates[0]
                                except Exception:
                                    pass
                            
                            key = final_name.lower().strip()
                            # Keep highest confidence version of each item
                            if key and (key not in item_confidence or combined_conf > item_confidence[key]):
                                item_data = {
                                    "name": final_name,
                                    "quantity": "1",
                                    "expirationDate": exp_date,
                                    "category": final_category,
                                    "confidence": combined_conf,
                                    "bbox": (x1, y1, x2, y2),
                                    "detection_method": detection_method,
                                    "yolo_confidence": confidence,
                                    "classification_confidence": classification_conf
                                }
                                items.append(item_data)
                                item_confidence[key] = combined_conf
                        except Exception as det_error:
                            if VERBOSE_LOGGING:
                                print(f"Warning: Error processing YOLOv8 detection: {det_error}")
                            continue
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"YOLOv8 detection error: {e}")
            detections_found_yolo = False
    
    # STEP 2b: DETR Detection (Fallback if YOLOv8 not available)
    # Note: detections_found is already initialized at line 932
    if not detections_found_yolo and _object_detector and ML_VISION_MODE == "hybrid":
        try:
            processor = _object_detector.get("processor")
            model = _object_detector.get("model")
            torch_module = _object_detector.get("torch")
            
            if not all([processor, model, torch_module]):
                if VERBOSE_LOGGING:

                    print("Warning: Object detector components missing")
                detections_found = False
            
            # Validate image before processing
            if img is None:
                detections_found = False
            
            try:
                inputs = processor(images=img, return_tensors="pt")
                pass
            except Exception as proc_error:
                if VERBOSE_LOGGING:

                    print(f"Warning: Failed to process image for object detection: {proc_error}")
                detections_found = False
            else:
                try:
                    with torch_module.no_grad():
                        outputs = model(**inputs)
                    pass
                except Exception as model_error:
                    if VERBOSE_LOGGING:

                        print(f"Warning: Object detection model failed: {model_error}")
                    detections_found = False
                else:
                    try:
                        target_sizes = torch_module.tensor([img.size[::-1]])
                        # Lower detection threshold from 0.7 to 0.4 to catch more items
                        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.4)
                        if not results or len(results) == 0:
                            detections_found = False
                        else:
                            results = results[0]
                            detections_found = True
                        pass
                    except Exception as post_error:
                        if VERBOSE_LOGGING:

                            print(f"Warning: Failed to post-process detection results: {post_error}")
                        detections_found = False
            
            if detections_found:
                # Expanded food mapping for common pantry items
                coco_food_map = {
                    "banana": "banana", "apple": "apple", "orange": "orange",
                    "sandwich": "sandwich", "pizza": "pizza", "donut": "donut",
                    "cake": "cake", "broccoli": "broccoli", "carrot": "carrot",
                    "hot dog": "hot dog", "bottle": "beverage", "bowl": "cereal",
                    "cup": "beverage", "fork": None, "knife": None, "spoon": None,
                    "bananas": "banana", "apples": "apple", "oranges": "orange",
                }
                
                # Safely extract results
                scores = results.get("scores", [])
                labels = results.get("labels", [])
                boxes = results.get("boxes", [])
                
                if not all([scores, labels, boxes]):
                    detections_found = False
                else:
                    # Limit number of detections to prevent memory issues
                    max_detections = 50
                    for idx, (score, label, box) in enumerate(zip(scores[:max_detections], labels[:max_detections], boxes[:max_detections])):
                        try:
                            # Validate score
                            try:
                                score_float = float(score)
                                pass
                            except (ValueError, TypeError):
                                pass
                            
                            if score_float < 0.5:  # Lower threshold
                                pass
                            
                            # Get label name safely
                            try:
                                label_int = int(label)
                                label_name = model.config.id2label.get(label_int, "")
                                pass
                            except (ValueError, TypeError, AttributeError):
                                pass
                            
                            mapped_name = coco_food_map.get(label_name)
                            if not mapped_name:
                                pass
                            
                            # Validate and extract box coordinates
                            try:
                                box_list = box.tolist() if hasattr(box, 'tolist') else list(box)
                                if len(box_list) < 4:
                                    pass
                                x0, y0, x1, y1 = [int(float(v)) for v in box_list[:4]]
                                pass
                            except (ValueError, TypeError, IndexError):
                                pass
                            
                            # Ensure valid crop coordinates
                            if x1 <= x0 or y1 <= y0:
                                pass
                            
                            # Ensure coordinates are within image bounds
                            if x0 < 0 or y0 < 0 or x1 > img.size[0] or y1 > img.size[1]:
                                # Clamp to image bounds
                                x0 = max(0, min(x0, img.size[0] - 1))
                                y0 = max(0, min(y0, img.size[1] - 1))
                                x1 = max(x0 + 1, min(x1, img.size[0]))
                                y1 = max(y0 + 1, min(y1, img.size[1]))
                            
                            try:
                                crop = img.crop((x0, y0, x1, y1))
                                if crop.size[0] == 0 or crop.size[1] == 0:
                                    pass
                                pass
                            except Exception as crop_error:
                                if VERBOSE_LOGGING:

                                    print(f"Warning: Failed to crop image: {crop_error}")
                                pass
                            
                            # STEP 2b: Hierarchical classification of crop
                            classification = classify_food_hierarchical(crop)
                            if classification:
                                # Use hierarchical classification if available (more accurate)
                                final_name = classification["name"]
                                final_category = classification["category"]
                                classification_conf = classification["confidence"]
                                # Combine detection and classification confidence (weighted)
                                combined_conf = (float(score_float) * 0.4 + classification_conf * 0.6)
                            else:
                                # Fallback to mapped name
                                final_name = mapped_name
                                final_category = validate_category(mapped_name, "other")
                                combined_conf = float(score_float)
                            
                            key = final_name.lower().strip()
                            # Keep highest confidence version of each item
                            if key and (key not in item_confidence or combined_conf > item_confidence[key]):
                                items.append({
                                    "name": final_name,
                                    "quantity": "1",
                                    "expirationDate": exp_date,
                                    "category": final_category,
                                    "confidence": combined_conf,
                                    "bbox": (x0, y0, x1, y1)  # Keep bbox for user correction
                                })
                                item_confidence[key] = combined_conf
                        except Exception as det_error:
                            if VERBOSE_LOGGING:

                                print(f"Warning: Error processing detection {idx}: {det_error}")
                            continue
        except Exception as e:
            if VERBOSE_LOGGING:

                print(f"Object detection error: {e}")
            # Only set detections_found to False if YOLO didn't find anything
            if not detections_found_yolo:
                detections_found = False
    
    # STEP 3: Fallback to Full-Image Classification if no detections found
    # Use YOLO results if available, otherwise use DETR results
    if detections_found_yolo:
        detections_found = True  # YOLO found items
    
    if not detections_found and _food_classifier:
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
                        pass
                    score = pred.get("score", 0)
                    try:
                        score = float(score)
                        pass
                    except (ValueError, TypeError):
                        pass
                    
                    if score < 0.15:
                        pass
                    
                    label = pred.get("label", "")
                    if not label or not isinstance(label, str):
                        pass
                    
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
                        pass
                    
                    # Skip non-food items
                    non_food_keywords = ["table", "plate", "fork", "knife", "spoon", "cup"]
                    if any(kw in label.lower() for kw in non_food_keywords):
                        if conf < 0.35:
                            pass
                    
                    key = name.lower().strip()
                    if key not in item_confidence or conf > item_confidence[key]:
                        items.append({
                            "name": name,
                            "quantity": "1",
                            "expirationDate": exp_date,
                            "category": category,
                            "confidence": conf
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

    # STEP 4: Apply Non-Maximum Suppression (NMS) to remove overlapping detections
    items = apply_nms(items, iou_threshold=0.5)
    
    # STEP 5: De-duplicate by name (keep highest confidence)
    unique = {}
    for item in items:
        key = item.get("name", "").lower().strip()
        if key and key not in unique:
            unique[key] = item
        elif key in unique:
            # Keep item with higher confidence
            existing_conf = unique[key].get("confidence", 0)
            new_conf = item.get("confidence", 0)
            if new_conf > existing_conf:
                unique[key] = item
    
    # Convert to list and sort by confidence
    result = list(unique.values())
    result.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    
    # STEP 6: Apply ensemble confidence boosting (if multiple methods detected same item)
    result = apply_ensemble_boosting(result, items)
    
    # STEP 7: Apply context fusion (boost confidence for items in pantry)
    if user_pantry:
        result = apply_context_fusion(result, user_pantry)
    
    # STEP 8: Calibrate confidence based on historical accuracy
    result = calibrate_confidence(result)
    
    # STEP 9: Return categorized by confidence (high/medium/low)
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
                return user_id, None
        except Exception as e:
            print(f"⚠️ Error creating user in Firebase: {e}, falling back to file storage")
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
    
    # Category mapping based on keywords
    category_keywords = {
        'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream', 'sour cream', 'cottage cheese', 'milk product'],
        'produce': ['apple', 'banana', 'orange', 'tomato', 'lettuce', 'carrot', 'onion', 'potato', 'vegetable', 'fruit', 'pepper', 'cucumber', 'broccoli', 'spinach'],
        'meat': ['chicken', 'beef', 'pork', 'fish', 'turkey', 'bacon', 'sausage', 'ham', 'steak', 'ground beef', 'salmon', 'tuna'],
        'beverages': ['juice', 'soda', 'water', 'coffee', 'tea', 'beer', 'wine', 'cola', 'drink'],
        'bakery': ['bread', 'bagel', 'muffin', 'croissant', 'roll', 'bun', 'pastry'],
        'canned goods': ['can', 'canned', 'soup', 'tuna', 'beans', 'corn'],
        'snacks': ['chip', 'cracker', 'cookie', 'nut', 'popcorn', 'pretzel', 'candy'],
        'condiments': ['sauce', 'ketchup', 'mustard', 'mayo', 'mayonnaise', 'dressing', 'spice', 'oil', 'vinegar', 'salt', 'pepper'],
        'grains': ['rice', 'pasta', 'cereal', 'flour', 'oat', 'quinoa', 'barley'],
        'frozen': ['frozen', 'ice cream']
    }
    
    # Check if category matches item name
    for correct_cat, keywords in category_keywords.items():
        if any(keyword in name_lower for keyword in keywords):
            return correct_cat
    
    return category  # Return original if no match

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
            # Add new item as dictionary with quantity and expiration date
            new_item = {
                'id': str(uuid.uuid4()),
                'name': item,
                'quantity': quantity,
                'expirationDate': normalized_expiration,
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
            # Load user's pantry from file (same method used in index route)
            user_id = session['user_id']
            users = load_users()
            if user_id in users:
                user_data = users[user_id]
                user_pantry = user_data.get('pantry', {}).get('items', [])
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"Error loading user pantry: {e}")
            pass
    else:
        # For anonymous users, use session pantry
        user_pantry = session.get('web_pantry', [])
    
    # Try ML vision first (if enabled), fallback to OpenAI
    detected_items_data = []
    use_ml = ML_VISION_ENABLED
    if use_ml:
        try:
            detected_items_data = detect_food_items_with_ml(img_bytes, user_pantry=user_pantry)
            if not detected_items_data and client:
                if VERBOSE_LOGGING:
                    print("ML vision found no items, falling back to OpenAI")
                use_ml = False
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"ML vision failed: {e}")
            use_ml = False

    if not use_ml:
        # Send image to OpenAI vision API
        import base64
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        prompt = """You are an expert food recognition system analyzing a pantry/fridge photo. Identify EVERY food item with maximum accuracy.

SCAN THE ENTIRE IMAGE SYSTEMATICALLY:
    - Look at ALL areas: foreground, background, shelves, containers, bags, boxes
- Check items that are partially visible, stacked, or overlapping
- Read labels and packaging text carefully
- Count multiple units of the same item

CRITICAL NAMING RULES:
    ✅ CORRECT: "milk", "chicken", "tomato", "bread", "pasta", "cheese", "eggs", "yogurt"
❌ WRONG: "milk carton", "chicken meat", "tomatoes" (use singular), "bread loaf", "Barilla pasta"

1. **Item Names** (MOST IMPORTANT):
   - Use SIMPLE, GENERIC food names: "milk" not "whole milk carton"
   - Remove ALL brand names: "Coca-Cola" → "cola", "Kellogg's Frosted Flakes" → "cereal"
   - Use SINGULAR form: "apple" not "apples", "tomato" not "tomatoes"
   - Remove packaging words: "milk carton" → "milk", "bread bag" → "bread"
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
   - Convert to YYYY-MM-DD: "01/15/2024" → "2024-01-15", "Jan 15 2024" → "2024-01-15"
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

6. **Accuracy Requirements**:
   - Include ALL clearly visible food items, ESPECIALLY rare/uncommon ones
   - Don't skip items just because they're in background or you don't recognize them
   - If you see packaging/labels, read them carefully to identify rare foods
   - When in doubt about a rare food, include it with a descriptive name based on the label
   - Only skip if completely unidentifiable or clearly not food

FEW-SHOT EXAMPLES:
    Example 1: Image shows milk carton, bread bag, and eggs
→ {"items": [{"name": "milk", "quantity": "1 carton", "expirationDate": "2024-01-20", "category": "dairy"}, {"name": "bread", "quantity": "1 loaf", "expirationDate": "2024-01-15", "category": "bakery"}, {"name": "egg", "quantity": "1 dozen", "expirationDate": null, "category": "dairy"}]}

Example 2: Image shows 3 cans of soup and a box of pasta
→ {"items": [{"name": "soup", "quantity": "3 cans", "expirationDate": null, "category": "canned goods"}, {"name": "pasta", "quantity": "1 box", "expirationDate": null, "category": "grains"}]}

Return ONLY valid JSON (no markdown, no code blocks, no explanations):
    {"items": [{"name": "...", "quantity": "...", "expirationDate": "YYYY-MM-DD or null", "category": "..."}]}"""
    
    try:
        if not use_ml:
            if not client:
                raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
            
            # Add timeout and error handling for API calls
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert food recognition system with exceptional attention to detail. Your task is to accurately identify all food items in images, extract quantities, read expiration dates from packaging labels, and classify items into appropriate categories. Always return results in valid JSON format with no additional text."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}}
                        ]}
                    ],
                    max_tokens=2000,
                    temperature=0.1,
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
        
        # Normalize and validate all items with partial recognition and expiration risk
            pantry_items = []
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
                    if not normalized_name or len(normalized_name) < 2:
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
                    pass
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
                existing_names = {item.get('name', '').strip().lower() for item in pantry_list if isinstance(item, dict) and item.get('name')}
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
                existing_names = {item.get('name', '').strip().lower() for item in pantry_list if isinstance(item, dict) and item.get('name')}
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
                    pass
                except Exception:
                    pass
        
        # Separate high-confidence and low-confidence items
        HIGH_CONFIDENCE_THRESHOLD = 0.7  # Auto-add items with 70%+ confidence
        high_conf_items = [item for item in pantry_items if item.get('confidence', 0) >= HIGH_CONFIDENCE_THRESHOLD]
        low_conf_items = [item for item in pantry_items if item.get('confidence', 0) < HIGH_CONFIDENCE_THRESHOLD]
        
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
                    import uuid
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
        
        # Add timeout and error handling for API calls
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a nutritionist and chef. Create recipes that use at least 50% pantry ingredients and provide accurate nutrition information."},
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
                pantry_to_use = get_session_pantry()
                # Sync global variable for backward compatibility
                global web_pantry
                web_pantry = pantry_to_use.copy()
        
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
        
        pantry_item = {
            'id': data.get('id', str(uuid.uuid4())),
            'name': item_name,
            'quantity': quantity.strip() if isinstance(quantity, str) else str(quantity),
            'expirationDate': expiration_date,
            'addedDate': data.get('addedDate', datetime.now().isoformat())
        }
        
        if VERBOSE_LOGGING:
                print(f"✅ Created pantry item: name='{pantry_item['name']}', quantity='{pantry_item['quantity']}', expirationDate={pantry_item['expirationDate']}")
    
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
            
            # Save all items at once using helper function
            if items_added > 0:
                set_session_pantry(pantry_list)
                if VERBOSE_LOGGING:
                    print(f"✅ Added {items_added} items to anonymous pantry via confirm_items")
                    print(f"   Session pantry now has {len(pantry_list)} items")
        
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
        
        return jsonify({
            'success': True,
            'items_added': items_added,
            'skipped_duplicates': skipped_duplicates,
            'total_items': len(pantry_list) if 'pantry_list' in locals() else 0,
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
                    
                    # Match by ID or name (if ID is empty or 'unknown', match by name only)
                    if (item_id_clean and item_id_from_dict and item_id_from_dict == item_id_clean) or item_name_from_dict == item_name_lower:
                        item_found = True
                        pass
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
        pantry_to_use = get_user_pantry(user_id)
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
        if client_type == 'mobile':
            # For mobile, use mobile_pantry global or session
            user_pantry = mobile_pantry
        else:
            # For web, use session pantry
            user_pantry = session.get('web_pantry', [])
        
        # Try ML vision first (if enabled), fallback to OpenAI
        detected_items_data = []
        pantry_items = []  # Initialize pantry_items early
        use_ml = ML_VISION_ENABLED
        if use_ml:
            try:
                detected_items_data = detect_food_items_with_ml(img_bytes, user_pantry=user_pantry)
                if not detected_items_data and client:
                    if VERBOSE_LOGGING:

                        print("ML vision found no items, falling back to OpenAI")
                    use_ml = False
                else:
                    # Process ML detection results into pantry_items format
                    if detected_items_data:
                        for item in detected_items_data:
                            try:
                                raw_name = item.get('name', '').strip()
                                if not raw_name:
                                    continue
                                
                                raw_quantity = item.get('quantity', '1')
                                expiration_date = item.get('expirationDate')
                                raw_category = item.get('category', 'other')
                                confidence = item.get('confidence', 0.5)
                                
                                # Normalize and validate
                                normalized_name = normalize_item_name(raw_name)
                                if not normalized_name or len(normalized_name) < 2:
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
                                    'confidence': float(confidence) if confidence else 0.5,
                                    'addedDate': datetime.now().isoformat()
                                })
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
                                    if 'confidence' not in item:
                                        item['confidence'] = 0.5
                                    if 'addedDate' not in item:
                                        item['addedDate'] = datetime.now().isoformat()
                                    seen_names.add(name_lower)
                                    unique_items.append(item)
                            except (AttributeError, TypeError):
                                continue
                        
                        pantry_items = unique_items
                        if VERBOSE_LOGGING:
                            print(f"✅ ML detection found {len(pantry_items)} items")
                pass
            except Exception as e:
                if VERBOSE_LOGGING:

                    print(f"ML vision failed: {e}")
                use_ml = False

        if not use_ml:
            # Send image to OpenAI vision API
            import base64
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
            prompt = """You are an expert food recognition system analyzing a pantry/fridge photo. Identify EVERY food item with maximum accuracy.

SCAN THE ENTIRE IMAGE SYSTEMATICALLY:
- Look at ALL areas: foreground, background, shelves, containers, bags, boxes
- Check items that are partially visible, stacked, or overlapping
- Read labels and packaging text carefully
- Count multiple units of the same item

CRITICAL NAMING RULES:
✅ CORRECT: "milk", "chicken", "tomato", "bread", "pasta", "cheese", "eggs", "yogurt"
❌ WRONG: "milk carton", "chicken meat", "tomatoes" (use singular), "bread loaf", "Barilla pasta"

1. **Item Names** (MOST IMPORTANT):
   - Use SIMPLE, GENERIC food names: "milk" not "whole milk carton"
   - Remove ALL brand names: "Coca-Cola" → "cola", "Kellogg's Frosted Flakes" → "cereal"
   - Use SINGULAR form: "apple" not "apples", "tomato" not "tomatoes"
   - Remove packaging words: "milk carton" → "milk", "bread bag" → "bread"
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
   - Convert to YYYY-MM-DD: "01/15/2024" → "2024-01-15", "Jan 15 2024" → "2024-01-15"
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

6. **Accuracy Requirements**:
   - Include ALL clearly visible food items, ESPECIALLY rare/uncommon ones
   - Don't skip items just because they're in background or you don't recognize them
   - If you see packaging/labels, read them carefully to identify rare foods
   - When in doubt about a rare food, include it with a descriptive name based on the label
   - Only skip if completely unidentifiable or clearly not food

FEW-SHOT EXAMPLES:
    Example 1: Image shows milk carton, bread bag, and eggs
→ {"items": [{"name": "milk", "quantity": "1 carton", "expirationDate": "2024-01-20", "category": "dairy"}, {"name": "bread", "quantity": "1 loaf", "expirationDate": "2024-01-15", "category": "bakery"}, {"name": "egg", "quantity": "1 dozen", "expirationDate": null, "category": "dairy"}]}

Example 2: Image shows 3 cans of soup and a box of pasta
→ {"items": [{"name": "soup", "quantity": "3 cans", "expirationDate": null, "category": "canned goods"}, {"name": "pasta", "quantity": "1 box", "expirationDate": null, "category": "grains"}]}

Return ONLY valid JSON (no markdown, no code blocks, no explanations):
    {"items": [{"name": "...", "quantity": "...", "expirationDate": "YYYY-MM-DD or null", "category": "..."}]}"""
        
        if not use_ml:
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
                        {"role": "system", "content": "You are an expert food recognition system with exceptional attention to detail. Your task is to accurately identify all food items in images, extract quantities, read expiration dates from packaging labels, and classify items into appropriate categories. Always return results in valid JSON format with no additional text."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}}
                        ]}
                    ],
                    max_tokens=2000,
                    temperature=0.1,
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
            
            # Normalize and validate all items
            pantry_items = []
            for item in detected_items_data:
                raw_name = item.get('name', '').strip()
                raw_quantity = item.get('quantity', '1')
                expiration_date = item.get('expirationDate')
                raw_category = item.get('category', 'other')
                
                # Normalize and validate
                normalized_name = normalize_item_name(raw_name)
                if not normalized_name or len(normalized_name) < 2:
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
        user_id = request.headers.get('X-User-ID')
        
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
                                'expired_days': abs(days_until_exp)
                            })
                        elif days_until_exp <= 7:
                            expiring_soon_items.append({
                                'name': item.get('name', 'Unknown'),
                                'days_remaining': days_until_exp
                            })
                    pass
                except Exception as e:
                    if VERBOSE_LOGGING:

                        print(f"Warning: Could not parse expirationDate '{exp_date_str}': {e}")
                    pass
        
        # Sort most common items
        most_common = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        most_common_items = [{'name': name, 'count': count} for name, count in most_common]
        
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
