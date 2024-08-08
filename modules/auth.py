import bcrypt
from modules.database import users_collection

def authenticate_user(username, password):
    """사용자 인증 함수"""
    user = users_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return user
    return None

def register_user(username, password):
    """사용자 등록 함수"""
    existing_user = users_collection.find_one({"username": username})
    if existing_user:
        return False
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({"username": username, "password": hashed_password})
    return True