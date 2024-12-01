# app/routes/__init__.py
from .superadmin import superadmin_bp
from .default import main_bp
from .image import image_bp

# Create a main blueprint for default routes



# Export the blueprints
__all__ = ['main_bp', 'superadmin_bp','image_bp']  