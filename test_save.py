import os
import sys
import json

# Add api directory to path
sys.path.insert(0, 'api')

# Set environment to local
os.environ.pop('VERCEL', None)
os.environ.pop('RENDER', None)

# Import after setting environment
from app import USERS_FILE, save_users, load_users

print(f"Testing save functionality...")
print(f"USERS_FILE: {USERS_FILE}")
print(f"Absolute path: {os.path.abspath(USERS_FILE)}")

# Test save
test_users = {
    'test-user-123': {
        'id': 'test-user-123',
        'username': 'testuser',
        'email': 'test@example.com',
        'password_hash': 'testhash',
        'pantry': []
    }
}

print(f"\nSaving {len(test_users)} test users...")
save_users(test_users)

print(f"\nLoading users...")
loaded = load_users()
print(f"Loaded {len(loaded)} users")
print(f"File exists: {os.path.exists(USERS_FILE)}")
if os.path.exists(USERS_FILE):
    print(f"File size: {os.path.getsize(USERS_FILE)} bytes")
    with open(USERS_FILE, 'r') as f:
        content = f.read()
        print(f"File content preview: {content[:200]}...")
