# Vercel serverless function entry point
# This file imports the Flask app from the same directory (api/app.py)
# IMPORTANT: Vercel's Python runtime auto-detects Flask apps when exported as 'app'
# Do NOT export as 'handler' - that triggers HTTP handler detection and causes errors

import sys
import os

# Ensure current directory (api/) is in Python path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Import Flask app - app.py should be in the same directory (api/)
from app import app
