# app/routes/superadmin.py
from app.handlers import SuperAdminHandler
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import (
    jwt_required, 
    create_access_token,
    get_jwt_identity
)
from middlewares import admin_required


superadmin_bp = Blueprint('superadmin', __name__, url_prefix='/api/v1')

@superadmin_bp.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        required_fields = ['email', 'password', 'first_name', 'last_name']
        
        # Validate required fields
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "message": f"Missing required field: {field}"
                }), 400

        handler = SuperAdminHandler(current_app.db)
        print('data',data)
        result = handler.create_superadmin(data)
        
        if result["success"]:
            return jsonify(result), 201
        return jsonify(result), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@superadmin_bp.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({
                "success": False,
                "message": "Email and password are required"
            }), 400

        handler = SuperAdminHandler(current_app.db)
        result = handler.login_superadmin(data["email"], data["password"])
        
        if result["success"]:
            # Create access token
            access_token = create_access_token(
                identity=result["user"]["id"],
                additional_claims={"role": "superadmin"}
            )
            result["token"] = access_token
            return jsonify(result), 200
        return jsonify(result), 401

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@superadmin_bp.route('/profile', methods=['GET'])
@jwt_required()
@admin_required
def get_profile():
    try:
        print('in the get profile')
        user_id = get_jwt_identity()
        handler = SuperAdminHandler(current_app.db)
        result =  handler.get_superadmin_by_id(user_id)
        
        if result["success"]:
            return jsonify(result), 200
        return jsonify(result), 404

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500