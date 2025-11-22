# Vercel serverless function entry point
# This file imports the Flask app from SmartPantryWeb directory

import sys
import os

# Get the project root directory (parent of api/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add SmartPantryWeb to Python path so we can import app
smartpantry_web_path = os.path.join(project_root, 'SmartPantryWeb')
if smartpantry_web_path not in sys.path:
    sys.path.insert(0, smartpantry_web_path)

# Debug: Print paths to help diagnose
print(f"Project root: {project_root}")
print(f"SmartPantryWeb path: {smartpantry_web_path}")
print(f"Python path: {sys.path[:3]}")  # First 3 entries
print(f"Current directory: {os.getcwd()}")
print(f"SmartPantryWeb exists: {os.path.exists(smartpantry_web_path)}")

if os.path.exists(smartpantry_web_path):
    try:
        files = os.listdir(smartpantry_web_path)
        print(f"Files in SmartPantryWeb: {files[:10]}")  # First 10 files
    except Exception as e:
        print(f"Error listing files: {e}")

# Import the app from SmartPantryWeb/app.py
try:
    from app import app
    print("✅ Successfully imported app from SmartPantryWeb/app.py")
except ImportError as e:
    # Enhanced error handling for debugging
    import traceback
    print(f"❌ ERROR: Failed to import app from SmartPantryWeb: {e}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")
    print(f"SmartPantryWeb path: {smartpantry_web_path}")
    print(f"Current working directory: {os.getcwd()}")
    if os.path.exists(smartpantry_web_path):
        print(f"Files in SmartPantryWeb: {os.listdir(smartpantry_web_path)}")
    else:
        print("❌ SmartPantryWeb directory does not exist!")
    traceback.print_exc()
    
    # Fallback: try importing directly from the module path
    try:
        import importlib.util
        app_path = os.path.join(smartpantry_web_path, 'app.py')
        print(f"Trying fallback import from: {app_path}")
        if os.path.exists(app_path):
            spec = importlib.util.spec_from_file_location("app", app_path)
            app_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(app_module)
            app = app_module.app
            print("✅ Fallback import successful")
        else:
            raise ImportError(f"app.py not found at {app_path}")
    except Exception as fallback_error:
        print(f"❌ ERROR: Fallback import also failed: {fallback_error}")
        traceback.print_exc()
        raise

# Verify app exists
if not app:
    raise ValueError("Flask app not found or is None")

print("✅ App verified, exporting handler...")

# Export handler for Vercel - this is what Vercel looks for
handler = app

print("✅ Handler exported successfully")
