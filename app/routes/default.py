from app.services.object_detection import ObjectDetector
from flask import Blueprint, jsonify, request, send_file, render_template, current_app
import cv2
import numpy as np
import io
import json
import base64
import time
import os
from werkzeug.utils import secure_filename
import logging
from threading import Lock
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__,url_prefix='/api/v1')
detector = ObjectDetector()

# Constants
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'temp_uploads'))
PROCESSED_FOLDER = os.path.join(UPLOAD_FOLDER, 'processed')
RETENTION_DAYS = 7  # Number of days to keep processed videos
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for base64 encoding

# Create necessary directories
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Processing status storage with thread safety
processing_status = {}
status_lock = Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def update_processing_status(timestamp, status):
    with status_lock:
        processing_status[timestamp] = status
        # Save to file for persistence
        status_file = os.path.join(UPLOAD_FOLDER, f"status_{timestamp}.json")
        with open(status_file, 'w') as f:
            json.dump(status, f)

def encode_file_to_base64(file_path):
    """Encode file to base64 in chunks to handle large files"""
    try:
        file_size = os.path.getsize(file_path)
        chunks = []
        
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                chunks.append(base64.b64encode(chunk).decode('utf-8'))
        
        return ''.join(chunks)
    except Exception as e:
        logger.error(f"Error encoding file to base64: {str(e)}")
        raise

def get_file_info(filepath):
    """Get file information including creation date and size"""
    stat = os.stat(filepath)
    return {
        'filename': os.path.basename(filepath),
        'created': datetime.fromtimestamp(stat.st_ctime),
        'size': stat.st_size,
        'path': filepath
    }

@main_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.debug('Health check requested')
    return jsonify({
        "status": "healthy",
        "message": "Server is running"
    }), 200
# Update the process_complete_video function with a web-compatible codec
def process_complete_video(input_path: str, output_path: str, timestamp: str):
    """Process the entire video and save to output path using only OpenCV"""
    logger.info(f"Starting complete video processing for {input_path}")
    
    try:
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open input video at {input_path}")
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create video writer with H.264 codec if available, otherwise use MP4V
        # Try different codecs in order of preference
        for codec in ['avc1', 'H264', 'mp4v', 'XVID']:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                logger.info(f"Trying codec: {codec}")
                out = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (frame_width, frame_height)
                )
                if out.isOpened():
                    logger.info(f"Successfully opened video writer with codec: {codec}")
                    break
            except Exception as e:
                logger.warning(f"Codec {codec} failed: {str(e)}")
                continue
        
        if not out.isOpened():
            raise ValueError(f"Failed to create output video writer at {output_path}")
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            progress = (frame_count / total_frames) * 100
            
            try:
                # Process frame
                detections, collisions = detector.detect_vehicles(frame)
                processed_frame = detector.draw_detections_and_collisions(
                    frame.copy(), detections, collisions)
                
                # Write the processed frame
                out.write(processed_frame)
                
                # Update status every 30 frames
                if frame_count % 30 == 0:
                    status = {
                        "progress": progress,
                        "frame_count": frame_count,
                        "total_frames": total_frames,
                        "status": "processing"
                    }
                    update_processing_status(timestamp, status)
                    logger.debug(f"Processing progress: {progress:.1f}% ({frame_count}/{total_frames})")
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {str(e)}")
                continue
        
        # Ensure all frames are written and resources are released
        out.release()
        cap.release()
        
        # Verify the output file exists and has size
        if not os.path.exists(output_path):
            raise ValueError(f"Output file not created at {output_path}")
        if os.path.getsize(output_path) == 0:
            raise ValueError(f"Output file is empty at {output_path}")
        
        # Update final status
        update_processing_status(timestamp, {
            "progress": 100,
            "frame_count": frame_count,
            "total_frames": total_frames,
            "status": "completed",
            "output_path": output_path
        })
        
        logger.info(f"Video processing completed. Processed {frame_count} frames.")
        logger.info(f"Output saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        update_processing_status(timestamp, {
            "status": "error",
            "error": str(e)
        })
        raise
        
    finally:
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()      
@main_bp.route('/process-video', methods=['POST'])
def process_video_file():
    """Handle video file upload and processing with robust error handling"""
    logger.info('Received video processing request')
    
    # Debug: Print all incoming request details
    logger.info(f'Request files: {request}')

    # Check different ways a file might be uploaded
    video_file = None
    if 'video' in request.files:
        video_file = request.files['video']
    elif 'file' in request.files:
        video_file = request.files['file']
    elif len(request.files) > 0:
        # If there's at least one file, use the first one
        video_file = list(request.files.values())[0]

    if not video_file:
        logger.error('No video file found in request')
        return jsonify({
            "error": "No video file provided", 
            "details": {
                "files_received": list(request.files.keys()),
                "content_type": request.content_type
            }
        }), 400
        
    # Filename checks
    if not video_file.filename:
        logger.error('Uploaded file has no filename')
        return jsonify({"error": "No selected file or filename is empty"}), 400
        
    # Check file type
    if not allowed_file(video_file.filename):
        logger.error(f'Unsupported file type: {video_file.filename}')
        return jsonify({
            "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
            "filename": video_file.filename
        }), 400
        
    input_path = None
    output_path = None
    
    try:
        timestamp = str(int(time.time()))
        input_filename = secure_filename(f"input_{timestamp}_{video_file.filename}")
        output_filename = f"processed_{timestamp}_{video_file.filename}"
        
        input_path = os.path.join(UPLOAD_FOLDER, input_filename)
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        # Ensure absolute paths
        input_path = os.path.abspath(input_path)
        output_path = os.path.abspath(output_path)
        
        logger.info(f'Saving input video to: {input_path}')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the input file
        video_file.save(input_path)
        
        # Verify input file was saved
        if not os.path.exists(input_path):
            raise ValueError(f"Failed to save input file at {input_path}")
        
        # Log file size to verify
        file_size = os.path.getsize(input_path)
        logger.info(f'Saved file size: {file_size} bytes')
        
        # Initialize processing status
        update_processing_status(timestamp, {
            "progress": 0,
            "status": "starting",
            "timestamp": timestamp
        })
        
        # Process the video
        success = process_complete_video(input_path, output_path, timestamp)
        
        if success and os.path.exists(output_path):
            logger.info(f"Video processed successfully: {output_path}")
            
            # Encode the processed video to base64
            try:
                base64_video = encode_file_to_base64(output_path)
                file_info = get_file_info(output_path)
                
                return jsonify({
                    "status": "success",
                    "message": "Video processed successfully",
                    "timestamp": timestamp,
                    "file_info": {
                        "filename": file_info['filename'],
                        "size": file_info['size'],
                        "created": file_info['created'].isoformat()
                    },
                    "video_data": base64_video
                })
            except Exception as e:
                logger.error(f"Error encoding video to base64: {str(e)}")
                return jsonify({"error": "Error encoding processed video"}), 500
        else:
            raise ValueError("Video processing completed but output file not found")
            
    except Exception as e:
        logger.error(f'Error processing video: {str(e)}', exc_info=True)
        return jsonify({
            "error": str(e),
            "details": {
                "input_path": input_path,
                "output_path": output_path,
                "filename": video_file.filename if video_file else "N/A"
            }
        }), 500
        
    finally:
        # Clean up input file only
        try:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
                logger.info(f"Removed input file: {input_path}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

@main_bp.route('/download/<timestamp>', methods=['GET'])
def download_processed_video(timestamp):
    """Download a processed video by timestamp as base64"""
    try:
        # Find the processed video file
        for filename in os.listdir(PROCESSED_FOLDER):
            if filename.startswith(f"processed_{timestamp}"):
                file_path = os.path.join(PROCESSED_FOLDER, filename)
                try:
                    base64_video = encode_file_to_base64(file_path)
                    file_info = get_file_info(file_path)
                    
                    return jsonify({
                        "status": "success",
                        "file_info": {
                            "filename": file_info['filename'],
                            "size": file_info['size'],
                            "created": file_info['created'].isoformat()
                        },
                        "video_data": base64_video
                    })
                except Exception as e:
                    logger.error(f"Error encoding video to base64: {str(e)}")
                    return jsonify({"error": "Error encoding video"}), 500
                    
        return jsonify({"error": "Processed video not found"}), 404
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        return jsonify({"error": "Error downloading video"}), 500

@main_bp.route('/processing-status/<timestamp>', methods=['GET'])
def get_processing_status(timestamp):
    """Get the processing status for a specific video"""
    try:
        with status_lock:
            if timestamp in processing_status:
                return jsonify(processing_status[timestamp])
            
        # Check status file if not in memory
        status_file = os.path.join(UPLOAD_FOLDER, f"status_{timestamp}.json")
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                return jsonify(json.load(f))
                
        return jsonify({"error": "Status not found"}), 404
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({"error": "Error getting status"}), 500

@main_bp.route('/processed-videos', methods=['GET'])
def list_processed_videos():
    """List all processed videos"""
    try:
        videos = []
        for filename in os.listdir(PROCESSED_FOLDER):
            if filename.startswith("processed_"):
                file_path = os.path.join(PROCESSED_FOLDER, filename)
                videos.append(get_file_info(file_path))
        
        return jsonify({
            "videos": sorted(videos, key=lambda x: x['created'], reverse=True)
        })
    except Exception as e:
        logger.error(f"Error listing videos: {str(e)}")
        return jsonify({"error": "Error listing videos"}), 500

# Error handlers
@main_bp.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File is too large"}), 413

@main_bp.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

def cleanup_old_files():
    """Clean up files older than RETENTION_DAYS"""
    logger.info('Running cleanup of old files')
    try:
        cutoff_date = datetime.now() - timedelta(days=RETENTION_DAYS)
        
        # Clean up processed videos
        for filename in os.listdir(PROCESSED_FOLDER):
            file_path = os.path.join(PROCESSED_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    file_date = datetime.fromtimestamp(os.path.getctime(file_path))
                    if file_date < cutoff_date:
                        os.remove(file_path)
                        logger.info(f'Cleaned up old file: {file_path}')
            except Exception as e:
                logger.error(f'Error cleaning up file {file_path}: {e}')
                
        # Clean up status files
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.startswith("status_"):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                try:
                    if os.path.isfile(file_path):
                        file_date = datetime.fromtimestamp(os.path.getctime(file_path))
                        if file_date < cutoff_date:
                            os.remove(file_path)
                            logger.info(f'Cleaned up old status file: {file_path}')
                except Exception as e:
                    logger.error(f'Error cleaning up status file {file_path}: {e}')
                    
    except Exception as e:
        logger.error(f'Error during cleanup: {e}')

# Add endpoint for manual cleanup
@main_bp.route('/cleanup', methods=['POST'])
def manual_cleanup():
    """Manually trigger cleanup of old files"""
    try:
        cleanup_old_files()
        return jsonify({
            "status": "success",
            "message": "Cleanup completed successfully"
        })
    except Exception as e:
        logger.error(f"Error during manual cleanup: {str(e)}")
        return jsonify({"error": "Error during cleanup"}), 500

# Register cleanup
import atexit
atexit.register(cleanup_old_files)

# Optional: Add configuration endpoint
@main_bp.route('/config', methods=['GET'])
def get_config():
    """Get current configuration settings"""
    return jsonify({
        "upload_folder": UPLOAD_FOLDER,
        "processed_folder": PROCESSED_FOLDER,
        "retention_days": RETENTION_DAYS,
        "allowed_extensions": list(ALLOWED_EXTENSIONS),
        "max_content_length": MAX_CONTENT_LENGTH,
        "chunk_size": CHUNK_SIZE
    })

# Optional: Add disk usage endpoint
@main_bp.route('/disk-usage', methods=['GET'])
def get_disk_usage():
    """Get disk usage information for upload and processed folders"""
    try:
        upload_size = sum(os.path.getsize(os.path.join(UPLOAD_FOLDER, f)) 
                         for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f)))
        processed_size = sum(os.path.getsize(os.path.join(PROCESSED_FOLDER, f)) 
                           for f in os.listdir(PROCESSED_FOLDER) if os.path.isfile(os.path.join(PROCESSED_FOLDER, f)))
        
        return jsonify({
            "upload_folder_size": upload_size,
            "processed_folder_size": processed_size,
            "total_size": upload_size + processed_size,
            "units": "bytes"
        })
    except Exception as e:
        logger.error(f"Error getting disk usage: {str(e)}")
        return jsonify({"error": "Error getting disk usage"}), 500

# Optional: Add endpoint to delete specific processed video
@main_bp.route('/delete/<timestamp>', methods=['DELETE'])
def delete_processed_video(timestamp):
    """Delete a specific processed video"""
    try:
        deleted = False
        for filename in os.listdir(PROCESSED_FOLDER):
            if filename.startswith(f"processed_{timestamp}"):
                file_path = os.path.join(PROCESSED_FOLDER, filename)
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted processed video: {file_path}")
                    deleted = True
                    
                    # Also delete associated status file
                    status_file = os.path.join(UPLOAD_FOLDER, f"status_{timestamp}.json")
                    if os.path.exists(status_file):
                        os.remove(status_file)
                        logger.info(f"Deleted status file: {status_file}")
                    
                    return jsonify({
                        "status": "success",
                        "message": "Video deleted successfully"
                    })
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")
                    return jsonify({"error": f"Error deleting file: {str(e)}"}), 500
        
        if not deleted:
            return jsonify({"error": "Video not found"}), 404
            
    except Exception as e:
        logger.error(f"Error in delete operation: {str(e)}")
        return jsonify({"error": "Error processing delete request"}), 500