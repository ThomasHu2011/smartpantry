# Vercel serverless function entry point
# This file imports the Flask app - Vercel copies app.py to api/ during build

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

# Export handler for Vercel
# Vercel expects a WSGI application (Flask app is WSGI-compatible)
# Export the Flask app instance directly - Vercel will detect it as WSGI
handler = app

