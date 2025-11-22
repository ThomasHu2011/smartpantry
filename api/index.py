# Vercel serverless function entry point
# This file imports the Flask app from the same directory (api/app.py)

import sys
import os
import traceback

# Print debug info immediately
print("=" * 60)
print("DEBUG: Starting api/index.py")
print("=" * 60)

# Ensure current directory (api/) is in Python path
_current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"DEBUG: Current directory: {_current_dir}")
print(f"DEBUG: __file__: {__file__}")

if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
    print(f"DEBUG: Added {_current_dir} to sys.path")

print(f"DEBUG: sys.path (first 5): {sys.path[:5]}")
print(f"DEBUG: Working directory: {os.getcwd()}")

# List files in current directory
try:
    files_in_dir = os.listdir(_current_dir)
    print(f"DEBUG: Files in api/ directory: {files_in_dir}")
    print(f"DEBUG: app.py exists: {os.path.exists(os.path.join(_current_dir, 'app.py'))}")
except Exception as e:
    print(f"DEBUG: Error listing directory: {e}")

print("=" * 60)
print("DEBUG: Attempting to import app...")
print("=" * 60)

# Import the app from api/app.py (same directory - Vercel includes this)
try:
    from app import app
    print("DEBUG: ✅ Successfully imported app!")
    print(f"DEBUG: app type: {type(app)}")
    print(f"DEBUG: app is None: {app is None}")
except ImportError as e:
    # Enhanced error handling for debugging
    print("=" * 60)
    print("ERROR: Failed to import app")
    print("=" * 60)
    print(f"ERROR: ImportError: {e}")
    print(f"ERROR: Current directory: {_current_dir}")
    print(f"ERROR: Python path: {sys.path[:10]}")
    try:
        print(f"ERROR: Files in api/: {os.listdir(_current_dir)}")
    except:
        print("ERROR: Could not list directory")
    print(f"ERROR: app.py exists: {os.path.exists(os.path.join(_current_dir, 'app.py'))}")
    print("=" * 60)
    print("FULL TRACEBACK:")
    print("=" * 60)
    traceback.print_exc()
    print("=" * 60)
    raise
except Exception as e:
    # Catch any other errors during import
    print("=" * 60)
    print(f"ERROR: Unexpected error during import: {type(e).__name__}")
    print(f"ERROR: {e}")
    print("=" * 60)
    traceback.print_exc()
    print("=" * 60)
    raise

# Verify app exists
if not app:
    raise ValueError("Flask app not found or is None")

print("=" * 60)
print("DEBUG: ✅ App imported successfully, exporting handler")
print("=" * 60)

# Export handler for Vercel - this is what Vercel looks for
handler = app
