from functools import wraps
from flask import jsonify, request, current_app
from flask_jwt_extended import (
    verify_jwt_in_request, 
    get_jwt_identity, 
    get_jwt, 
    jwt_required
)
import jwt
import logging

def admin_required(fn):
    """
    Decorator to enforce superadmin access with comprehensive error handling
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            # Verify JWT token exists and is valid
            verify_jwt_in_request()
            
            # Get JWT claims
            claims = get_jwt()
            
            # Additional token validation
            if not claims:
                return jsonify({
                    "success": False,
                    "message": "Invalid token claims"
                }), 401
            
            # Check superadmin role
            if claims.get('role') != 'superadmin':
                return jsonify({
                    "success": False,
                    "message": "Superadmin access required"
                }), 403
            
            # Additional optional checks
            if claims.get('status') == 'inactive':
                return jsonify({
                    "success": False,
                    "message": "Account is inactive"
                }), 403
            
            # Proceed with the original function
            return fn(*args, **kwargs)
        
        except jwt.ExpiredSignatureError:
            # Handle expired token
            return jsonify({
                "success": False,
                "message": "Token has expired. Please log in again."
            }), 401
        
        except jwt.InvalidTokenError:
            # Handle invalid token (malformed, tampered, etc.)
            return jsonify({
                "success": False,
                "message": "Invalid authentication token"
            }), 401
        
        except Exception as e:
            # Log unexpected errors
            current_app.logger.error(f"Unexpected auth error: {str(e)}", exc_info=True)
            return jsonify({
                "success": False,
                "message": "Authentication error occurred"
            }), 500
    
    return wrapper

def role_required(allowed_roles):
    """
    Decorator to enforce role-based access control
    
    :param allowed_roles: List of roles allowed to access the endpoint
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                # Verify JWT token exists and is valid
                verify_jwt_in_request()
                
                # Get JWT claims
                claims = get_jwt()
                
                # Validate claims
                if not claims:
                    return jsonify({
                        "success": False,
                        "message": "Invalid token claims"
                    }), 401
                
                # Check if user's role is in allowed roles
                user_role = claims.get('role')
                if user_role not in allowed_roles:
                    return jsonify({
                        "success": False,
                        "message": f"Access denied. Required roles: {', '.join(allowed_roles)}"
                    }), 403
                
                # Proceed with the original function
                return fn(*args, **kwargs)
            
            except jwt.ExpiredSignatureError:
                return jsonify({
                    "success": False,
                    "message": "Token has expired. Please log in again."
                }), 401
            
            except jwt.InvalidTokenError:
                return jsonify({
                    "success": False,
                    "message": "Invalid authentication token"
                }), 401
            
            except Exception as e:
                current_app.logger.error(f"Unexpected auth error: {str(e)}", exc_info=True)
                return jsonify({
                    "success": False,
                    "message": "Authentication error occurred"
                }), 500
        
        return wrapper
    return decorator