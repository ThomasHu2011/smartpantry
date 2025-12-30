# Vercel serverless function entry point
# This file imports the Flask app from the parent directory

import sys
import os
import traceback

# Debug: Print environment info
print("=" * 60)
print("Vercel Serverless Function Starting")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"__file__: {__file__}")
print(f"Python path (first 5): {sys.path[:5]}")

# Get the SmartPantryWeb directory (parent of api/)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)

print(f"Current dir (api/): {_current_dir}")
print(f"Parent dir (SmartPantryWeb/): {_parent_dir}")

# Add parent directory to Python path
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
    print(f"Added {_parent_dir} to Python path")

# Try multiple import strategies
# IMPORTANT: Vercel copies app.py into the api/ directory during build
# So we should try importing from the current directory first
app = None
import_error = None

# Strategy 1: Import from current directory (Vercel copies app.py here)
try:
    print("Attempting import: from app import app (from current api/ directory)")
    # Import directly from current directory - Vercel includes app.py here
    from app import app
    print("✅ Successfully imported app from current directory")
except ImportError as e:
    import_error = e
    print(f"❌ Import from current directory failed: {e}")
    traceback.print_exc()
    
    # Strategy 2: Try importing from parent directory (fallback for local dev)
    try:
        print("Attempting import: from parent directory")
        # Add parent to path if not already there
        if _parent_dir not in sys.path:
            sys.path.insert(0, _parent_dir)
        from app import app
        print("✅ Successfully imported app from parent directory")
    except ImportError as e2:
        print(f"❌ Import from parent directory also failed: {e2}")
        traceback.print_exc()
        
        # Strategy 3: Try direct file import
        try:
            print("Attempting import: direct file import")
            import importlib.util
            app_path = os.path.join(_current_dir, 'app.py')
            if os.path.exists(app_path):
                spec = importlib.util.spec_from_file_location("app_module", app_path)
                app_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(app_module)
                app = app_module.app
                print("✅ Successfully imported app via direct file import")
            else:
                print(f"❌ app.py not found at {app_path}")
        except Exception as e3:
            print(f"❌ Direct file import also failed: {e3}")
            traceback.print_exc()

# If still no app, raise error with detailed info
if app is None:
    print("=" * 60)
    print("FATAL ERROR: Could not import Flask app")
    print("=" * 60)
    print(f"Import error: {import_error}")
    print(f"Files in parent directory: {os.listdir(_parent_dir) if os.path.exists(_parent_dir) else 'Directory does not exist'}")
    print(f"Files in current directory: {os.listdir(_current_dir) if os.path.exists(_current_dir) else 'Directory does not exist'}")
    print("=" * 60)
    raise ImportError(f"Failed to import Flask app. Original error: {import_error}")

# Verify app exists and is a Flask app
if not app:
    raise ValueError("Flask app is None after import")

# Verify it's actually a Flask app
try:
    from flask import Flask
    if not isinstance(app, Flask):
        raise ValueError(f"Imported app is not a Flask instance, got {type(app)}")
except ImportError:
    print("Warning: Could not verify Flask type (flask module not available)")

print("=" * 60)
print("✅ Flask app imported successfully")
print(f"App type: {type(app)}")
print(f"App is callable: {callable(app)}")
print("=" * 60)

# Export handler for Vercel - this is what Vercel looks for
# Vercel's Python runtime expects a WSGI application (Flask app is WSGI-compatible)
# The Flask app already has a global error handler, so uncaught errors will be handled
# 
# IMPORTANT: Export the Flask app instance directly as 'handler'
# Vercel will detect it as a WSGI application and call it with (environ, start_response)
#
# The error "issubclass() arg 1 must be a class" in Vercel's internal code suggests
# that Vercel's handler detection is getting confused. We ensure the handler is
# a clean WSGI callable by exporting the Flask app directly.
handler = app

# Verify handler is properly set up
if handler is None:
    raise ValueError("Handler is None - Flask app was not imported correctly")
if not callable(handler):
    raise ValueError(f"Handler is not callable - got {type(handler)}")

# Ensure handler has __call__ method (WSGI requirement)
if not hasattr(handler, '__call__'):
    raise ValueError("Handler does not have __call__ method - not a valid WSGI app")

print("=" * 60)
print("✅ Handler exported successfully")
print(f"Handler type: {type(handler)}")
print(f"Handler class: {handler.__class__}")
print(f"Handler MRO: {handler.__class__.__mro__}")
print("=" * 60)

