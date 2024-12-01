# app/__init__.py
from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from config import config
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Global MongoDB client
mongo_client = None
jwt = JWTManager()

def init_mongodb(app):
    """
    Initialize MongoDB connection using configuration from app
    Returns the database instance
    """
    global mongo_client
    try:
        # Get MongoDB URI from config
        mongodb_uri = app.config.get('MONGODB_URI')
        database_name = app.config.get('MONGODB_DB')
        
        # Create MongoDB client
        mongo_client = MongoClient(mongodb_uri)
        
        # Test the connection
        mongo_client.admin.command('ping')
        
        print(f"Successfully connected to MongoDB at {mongodb_uri}")
        return mongo_client[database_name]
        
    except ConnectionFailure as e:
        print(f"Failed to connect to MongoDB: {str(e)}")
        raise
    except Exception as e:
        print(f"An error occurred while connecting to MongoDB: {str(e)}")
        raise

def create_app(config_name=None):
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'default')
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize JWT
    jwt.init_app(app)
    
    # Configure CORS
    CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:4200", "http://localhost:4200/"],  # Add both variations
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Include more HTTP methods
        "allow_headers": [
            "Content-Type", 
            "Authorization", 
            "X-Requested-With",  # Often needed for AJAX requests
            "Accept"
        ],
        "supports_credentials": True,
        "max_age": 3600
    }
    })
    
    # Initialize MongoDB connection
    app.db = init_mongodb(app)
    
    # Import and register blueprints from routes package
    from app.routes import main_bp, superadmin_bp,image_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(superadmin_bp)
    app.register_blueprint(image_bp)
    
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Register teardown function to close MongoDB connection
    
    
    return app