# Vercel serverless function entry point
# This file imports the Flask app from the same directory (api/app.py)

import sys
import os

# Ensure current directory (api/) is in Python path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Import the app from api/app.py (same directory - Vercel includes this)
try:
    from app import app
except ImportError as e:
    # Enhanced error handling for debugging
    import traceback
    print(f"ERROR: Failed to import app: {e}")
    print(f"Current directory: {_current_dir}")
    print(f"Python path: {sys.path[:5]}")
    print(f"Files in api/: {os.listdir(_current_dir) if os.path.exists(_current_dir) else 'Directory not found'}")
    print(f"app.py exists: {os.path.exists(os.path.join(_current_dir, 'app.py'))}")
    traceback.print_exc()
    raise

# Verify app exists
if not app:
    raise ValueError("Flask app not found or is None")

# Export handler for Vercel - this is what Vercel looks for
handler = app
