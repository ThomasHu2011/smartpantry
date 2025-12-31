# Vercel serverless function entry point
# This file imports the Flask app from the same directory (api/app.py)
# IMPORTANT: Vercel's Python runtime auto-detects Flask apps when exported as 'app'
# Do NOT export as 'handler' - that triggers HTTP handler detection and causes errors

# Import Flask app - Vercel copies app.py to api/ during build
from app import app
