import os
from app.services.object_detection import ObjectDetector
import cv2
import numpy as np
import logging
import base64
import time
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('image_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# Image processing blueprint
image_bp = Blueprint('image_processing', __name__, url_prefix='/api/v1')
detector = ObjectDetector()

# Constants
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'temp_uploads', 'images'))
PROCESSED_FOLDER = os.path.join(UPLOAD_FOLDER, 'processed')
RETENTION_DAYS = 7
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

# Ensure directories exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(file_path):
    """Encode image to base64"""
    try:
        with open(file_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        raise

def get_file_info(filepath):
    """Get file information"""
    stat = os.stat(filepath)
    return {
        'filename': os.path.basename(filepath),
        'created': datetime.fromtimestamp(stat.st_ctime),
        'size': stat.st_size,
        'path': filepath
    }

@image_bp.route('/process-image', methods=['POST'])
def detect_vehicle_collision():
    """
    Detect potential vehicle collisions in uploaded images.
    Supports single or multiple image upload.
    """
    logger.info('Received image collision detection request')
    
    # Validate request
    if 'images' not in request.files:
        return jsonify({"error": "No images uploaded"}), 400
    
    uploaded_files = request.files.getlist('images')
    
    if not uploaded_files or uploaded_files[0].filename == '':
        return jsonify({"error": "No selected files"}), 400
    
    # Validate file types and sizes
    processed_images = []
    
    try:
        # Process each uploaded image
        for file in uploaded_files:
            # Filename and path generation
            timestamp = str(int(time.time()))
            input_filename = secure_filename(f"input_{timestamp}_{file.filename}")
            output_filename = f"processed_{timestamp}_{file.filename}"
            
            input_path = os.path.join(UPLOAD_FOLDER, input_filename)
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)
            
            # Validate file type
            if not allowed_file(file.filename):
                logger.warning(f'Unsupported file type: {file.filename}')
                continue
            
            # Save input file
            file.save(input_path)
            
            try:
                # Read image with OpenCV
                frame = cv2.imread(input_path)
                
                if frame is None:
                    logger.error(f'Failed to read image: {input_path}')
                    continue
                
                # Perform vehicle detection and collision analysis
                detections, collisions = detector.detect_vehicles(frame)
                
                # Draw detections and collision markers
                processed_frame = detector.draw_detections_and_collisions(
                    frame.copy(), detections, collisions
                )
                
                # Save processed image
                cv2.imwrite(output_path, processed_frame)
                
                # Encode processed image
                base64_image = encode_image_to_base64(output_path)
                file_info = get_file_info(output_path)
                
                # Prepare image result
                image_result = {
                    "original_filename": file.filename,
                    "file_info": {
                        "filename": file_info['filename'],
                        "size": file_info['size'],
                        "created": file_info['created'].isoformat()
                    },
                    "detections": {
                        "total_vehicles": len(detections),
                        "potential_collisions": len(collisions)
                    },
                    "image_data": base64_image
                }
                
                processed_images.append(image_result)
                
            except Exception as e:
                logger.error(f"Error processing image {file.filename}: {str(e)}")
                continue
            
            finally:
                # Clean up input file
                if os.path.exists(input_path):
                    os.remove(input_path)
        
        # Return results
        if processed_images:
            return jsonify({
                "status": "success",
                "total_processed": len(processed_images),
                "images": processed_images
            })
        else:
            return jsonify({
                "error": "No valid images could be processed"
            }), 400
    
    except Exception as e:
        logger.error(f'Unexpected error in image processing: {str(e)}')
        return jsonify({"error": str(e)}), 500

@image_bp.route('/processed-images', methods=['GET'])
def list_processed_images():
    """List all processed images"""
    try:
        images = []
        for filename in os.listdir(PROCESSED_FOLDER):
            if filename.startswith("processed_"):
                file_path = os.path.join(PROCESSED_FOLDER, filename)
                images.append(get_file_info(file_path))
        
        return jsonify({
            "images": sorted(images, key=lambda x: x['created'], reverse=True)
        })
    except Exception as e:
        logger.error(f"Error listing processed images: {str(e)}")
        return jsonify({"error": "Error listing processed images"}), 500

@image_bp.route('/delete/<timestamp>', methods=['DELETE'])
def delete_processed_image(timestamp):
    """Delete a specific processed image"""
    try:
        deleted = False
        for filename in os.listdir(PROCESSED_FOLDER):
            if filename.startswith(f"processed_{timestamp}"):
                file_path = os.path.join(PROCESSED_FOLDER, filename)
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted processed image: {file_path}")
                    deleted = True
                    return jsonify({
                        "status": "success",
                        "message": "Image deleted successfully"
                    })
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")
                    return jsonify({"error": f"Error deleting file: {str(e)}"}), 500
        
        if not deleted:
            return jsonify({"error": "Image not found"}), 404
            
    except Exception as e:
        logger.error(f"Error in delete operation: {str(e)}")
        return jsonify({"error": "Error processing delete request"}), 500

def cleanup_old_files():
    """Clean up files older than RETENTION_DAYS"""
    logger.info('Running cleanup of old image files')
    try:
        cutoff_date = datetime.now() - timedelta(days=RETENTION_DAYS)
        
        # Clean up processed images
        for filename in os.listdir(PROCESSED_FOLDER):
            file_path = os.path.join(PROCESSED_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    file_date = datetime.fromtimestamp(os.path.getctime(file_path))
                    if file_date < cutoff_date:
                        os.remove(file_path)
                        logger.info(f'Cleaned up old image file: {file_path}')
            except Exception as e:
                logger.error(f'Error cleaning up image file {file_path}: {e}')
                
    except Exception as e:
        logger.error(f'Error during image cleanup: {e}')

# Register cleanup
import atexit
atexit.register(cleanup_old_files)

# Configuration endpoint
@image_bp.route('/config', methods=['GET'])
def get_config():
    """Get current configuration settings"""
    return jsonify({
        "upload_folder": UPLOAD_FOLDER,
        "processed_folder": PROCESSED_FOLDER,
        "retention_days": RETENTION_DAYS,
        "allowed_extensions": list(ALLOWED_EXTENSIONS),
        "max_content_length": MAX_CONTENT_LENGTH
    })