# Vercel serverless function entry point
# This file imports the Flask app from the same directory (api/app.py)

# Import the app from api/app.py (same directory - Vercel includes this)
try:
    from app import app
except ImportError as e:
    # Enhanced error handling for debugging
    import traceback
    import sys
    import os
    print(f"ERROR: Failed to import app: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in api/: {os.listdir(os.path.dirname(__file__))}")
    traceback.print_exc()
    raise

# Verify app exists
if not app:
    raise ValueError("Flask app not found or is None")

# Export handler for Vercel - this is what Vercel looks for
handler = app
