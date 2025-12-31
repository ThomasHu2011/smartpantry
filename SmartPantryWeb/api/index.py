# Vercel serverless function entry point
# This file imports the Flask app - Vercel copies app.py to api/ during build
# IMPORTANT: Vercel's Python runtime auto-detects Flask apps when exported as 'app'
# Do NOT export as 'handler' - that triggers HTTP handler detection and causes errors

import sys
import os

# Get directory paths
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)

# Add parent directory to Python path for imports
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import Flask app
# Vercel copies app.py to api/ directory during build, so try current dir first
try:
    # Try importing from current directory (where Vercel puts app.py)
    from app import app
except ImportError:
    # Fallback: import from parent directory (for local development)
    # Ensure parent is in path
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from app import app

# Verify app is valid
if not app:
    raise ValueError("Flask app is None")

# Export app directly - Vercel auto-detects Flask WSGI apps when exported as 'app'
# Do NOT use 'handler = app' - that causes Vercel to treat it as an HTTP handler class
# and triggers the issubclass() TypeError

