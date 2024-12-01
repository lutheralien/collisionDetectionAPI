from datetime import datetime
from bson import ObjectId

class SuperAdminSchema:
    def __init__(self, email, password, first_name, last_name):
        self.email = email
        self.password = password  # This should be hashed before storage
        self.first_name = first_name
        self.last_name = last_name
        self.created_at = datetime.utcnow()
        self.last_login = None
        self.is_active = True
        self.role = "superadmin",

    def to_dict(self):
        return {
            "email": self.email,
            "password": self.password,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "is_active": self.is_active,
            "role": self.role
        }
