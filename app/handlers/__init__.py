from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pymongo.database import Database
from flask import current_app
import logging
import re
import jwt


class SuperAdminHandler:
    """Handles superadmin operations including creation, authentication, and retrieval."""

    def __init__(self, db: Database):
        """
        Initialize the SuperAdminHandler.

        Args:
            db: Database instance from pymongo for MongoDB operations.
        """
        self.db = db
        self.collection = db.superadmins
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))

    @staticmethod
    def _validate_password(password: str) -> bool:
        """
        Validate password strength.
        Requires at least 8 characters, one uppercase, one lowercase, one number
        """
        if len(password) < 8:
            return False
        if not re.search(r"[A-Z]", password):
            return False
        if not re.search(r"[a-z]", password):
            return False
        if not re.search(r"\d", password):
            return False
        return True

    def _check_email_exists(self, email: str) -> bool:
        """Check if email already exists in database."""
        return self.collection.find_one({"email": email}) is not None

    def create_superadmin(self, data: Dict[str, str]) -> Dict[str, Any]:
        """
        Create a new superadmin account.

        Args:
            data: Dictionary containing email, password, first_name, and last_name

        Returns:
            Dict containing success status, message, and created admin ID if successful
        """
        try:
            # Validate required fields
            required_fields = ["email", "password", "first_name", "last_name"]
            missing_fields = [field for field in required_fields if not data.get(field)]
            if missing_fields:
                return {
                    "success": False,
                    "message": f"Missing required fields: {', '.join(missing_fields)}"
                }

            # Validate email format
            if not self._validate_email(data["email"]):
                return {
                    "success": False,
                    "message": "Invalid email format"
                }

            # Validate password strength
            if not self._validate_password(data["password"]):
                return {
                    "success": False,
                    "message": "Password must be at least 8 characters long and contain uppercase, lowercase, and numbers"
                }

            # Check for existing admin
            if self._check_email_exists(data["email"]):
                return {
                    "success": False,
                    "message": "Email already registered"
                }

            # Create new superadmin document
            superadmin = {
                "email": data["email"].lower(),  # Store email in lowercase
                "password": generate_password_hash(data["password"], method='pbkdf2:sha256'),
                "first_name": data["first_name"].strip(),
                "last_name": data["last_name"].strip(),
                "role": "superadmin",
                "status": "active",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "last_login": None,
                "failed_login_attempts": 0
            }

            result = self.collection.insert_one(superadmin)
            return {
                "success": True,
                "message": "Superadmin created successfully",
                "id": str(result.inserted_id)
            }

        except Exception as e:
            self.logger.error(f"Error creating superadmin: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": "Internal server error occurred"
            }

    def login_superadmin(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate a superadmin and generate JWT token.

        Args:
            email: Superadmin email
            password: Superadmin password

        Returns:
            Dict containing success status, token, and user data if successful
        """
        try:
            superadmin = self.collection.find_one({"email": email.lower()})
            if not superadmin:
                return {"success": False, "message": "Invalid email or password"}

            # Check account status
            if superadmin.get("status") == "locked":
                return {"success": False, "message": "Account is locked. Please contact support."}

            # Verify password
            if not check_password_hash(superadmin["password"], password):
                # Increment failed login attempts
                self.collection.update_one(
                    {"_id": superadmin["_id"]},
                    {"$inc": {"failed_login_attempts": 1}}
                )

                # Lock account after 5 failed attempts
                if superadmin.get("failed_login_attempts", 0) >= 4:
                    self.collection.update_one(
                        {"_id": superadmin["_id"]},
                        {"$set": {"status": "locked"}}
                    )
                    return {"success": False, "message": "Account locked due to multiple failed attempts"}

                return {"success": False, "message": "Invalid email or password"}

            # Generate JWT token with appropriate expiration
            token_payload = {
                'user_id': str(superadmin['_id']),
                'email': superadmin['email'],
                'role': superadmin['role'],
                'exp': datetime.utcnow() + timedelta(hours=12)  # 12-hour token
            }

            token = jwt.encode(
                token_payload,
                current_app.config['SECRET_KEY'],
                algorithm="HS256"
            )

            # Reset failed login attempts and update last login
            self.collection.update_one(
                {"_id": superadmin["_id"]},
                {
                    "$set": {
                        "last_login": datetime.utcnow(),
                        "failed_login_attempts": 0,
                        "updated_at": datetime.utcnow()
                    }
                }
            )

            return {
                "success": True,
                "token": token,
                "user": {
                    "id": str(superadmin["_id"]),
                    "email": superadmin["email"],
                    "first_name": superadmin["first_name"],
                    "last_name": superadmin["last_name"],
                    "role": superadmin["role"]
                }
            }

        except Exception as e:
            self.logger.error(f"Error during login: {str(e)}", exc_info=True)
            return {"success": False, "message": "Internal server error occurred"}
  
  
    def get_superadmin_by_id(self, superadmin_id: str) -> Dict[str, Any]:
        """
        Retrieve a superadmin by their ID.
        
        Args:
            superadmin_id: The ID of the superadmin to retrieve
        
        Returns:
            Dict containing success status and superadmin data if found
        """
        try:
            if not ObjectId.is_valid(superadmin_id):
                return {"success": False, "message": "Invalid superadmin ID format"}

            superadmin = self.collection.find_one({"_id": ObjectId(superadmin_id)})
            if not superadmin:
                return {"success": False, "message": "Superadmin not found"}

            # Remove sensitive fields
            superadmin.pop("password", None)
            superadmin.pop("failed_login_attempts", None)
            superadmin["_id"] = str(superadmin["_id"])

            return {"success": True, "data": superadmin, "message": "Superadmin details fetched successfully"}

        except Exception as e:
            self.logger.error(f"Error retrieving superadmin: {str(e)}", exc_info=True)
            return {"success": False, "message": "Internal server error occurred"}