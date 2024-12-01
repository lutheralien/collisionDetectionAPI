# config.py
from datetime import timedelta
import os

class Config:
    UPLOAD_FOLDER = 'temp_uploads'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    PORT = 5000  # Default port
    MONGODB_URI = "mongodb://localhost:27017/"
    MONGODB_DB = "collision-detection"
    SECRET_KEY = 'your-secret-key-here'  # Change this to a secure secret key
    JWT_SECRET_KEY = 'your-jwt-secret-key-here'  # Change this to a secure secret key
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=1)

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False
    SERVER_NAME = None
    PORT = 5000

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    SERVER_NAME = 'your.domain.com'
    PORT = 80

class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    PORT = 5000

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}