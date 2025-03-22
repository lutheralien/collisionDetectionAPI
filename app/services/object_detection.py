from ultralytics import YOLO
import cv2
import numpy as np
import os
import torch
from typing import List, Dict, Tuple, Any, Optional
from collections import deque
import math
from scipy.optimize import linear_sum_assignment
import time

class ObjectDetector:
    """
    Enhanced vehicle detection and collision prediction system using YOLOv8
    with improved tracking and collision forecasting capabilities.
    """
    def __init__(
        self, 
        model_size: str = "l",  # Upgraded from 'n' (nano) to 'm' (medium)
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.3,
        track_history: int = 60,  # Increased from 30
        collision_thresholds: Dict[str, float] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the object detector with configurable parameters.
        
        Args:
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            confidence_threshold: Minimum detection confidence
            iou_threshold: IoU threshold for tracking
            track_history: Number of frames to track for trajectory analysis
            collision_thresholds: Dict of thresholds for collision detection
            use_gpu: Whether to use GPU acceleration
        """
        # Model configuration
        self.model_path = f"models/yolov8{model_size}.pt"
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Collision detection parameters
        self.track_history = track_history
        self.collision_thresholds = collision_thresholds or {
            "distance": 50,        # Minimum distance in pixels
            "time_to_collision": 1.5,  # Time to collision threshold in seconds
            "iou": 0.15,           # IoU threshold
            "trajectory_angle": 45 # Maximum angle between converging trajectories
        }
        
        # Use GPU if available and requested
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
        
        # Ensure model exists
        self._ensure_model_files()
        
        # Load YOLO model
        self.model = YOLO(self.model_path)
        
        # Vehicle classes from COCO dataset
        # Expanded to include more vehicle types
        self.vehicle_classes = {
            2: "car", 
            3: "motorcycle", 
            5: "bus", 
            7: "truck",
            8: "boat",  # Added additional vehicle types
            9: "traffic light"  # Important context for collision analysis
        }
        
        # Tracking state
        self.reset_tracking()
        
        # Performance metrics
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)
        
        # Debug mode for visualization
        self.debug_mode = False

    def reset_tracking(self):
        """Reset all tracking and collision detection state"""
        # Vehicle tracking
        self.trajectories = {}  # vehicle_id -> list of positions
        self.velocities = {}    # vehicle_id -> (vx, vy)
        self.accelerations = {} # vehicle_id -> (ax, ay)
        self.bounding_boxes = {} # vehicle_id -> current bounding box
        self.vehicle_classes_detected = {} # vehicle_id -> class name
        self.last_seen = {}     # vehicle_id -> last frame index
        self.next_vehicle_id = 0
        
        # Collision tracking
        self.collision_history = {}  # (id1, id2) -> collision data
        self.frame_count = 0
    
    def _ensure_model_files(self):
        """Download YOLOv8 model if it doesn't exist"""
        if not os.path.exists(self.model_path):
            os.makedirs("models", exist_ok=True)
            # Download the model
            model_name = os.path.basename(self.model_path).replace(".pt", "")
            self.model = YOLO(model_name)
            self.model.save(self.model_path)
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union between two boxes.
        Each box format: [x, y, w, h]
        """
        # Convert to [x1, y1, x2, y2] format
        b1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
        b2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
        
        # Get intersecting box
        x_left = max(b1[0], b2[0])
        y_top = max(b1[1], b2[1])
        x_right = min(b1[2], b2[2])
        y_bottom = min(b1[3], b2[3])
        
        # Check for non-overlapping boxes
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate areas
        intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
        box1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
        box2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_box_center(self, box: List[float]) -> Tuple[float, float]:
        """Get the center coordinates of a bounding box"""
        x, y, w, h = box
        return (x + w/2, y + h/2)
    
    def _calculate_distance(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Euclidean distance between centers of two boxes"""
        center1 = self._get_box_center(box1)
        center2 = self._get_box_center(box2)
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _match_detections_to_tracks(self, detections: List[Dict], 
                                    current_boxes: Dict[int, List[float]]) -> Dict[int, int]:
        """
        Match new detections to existing tracks using Hungarian algorithm
        Returns a mapping of detection indices to vehicle IDs
        """
        if not detections or not current_boxes:
            return {}
        
        # Build cost matrix based on IoU
        cost_matrix = np.zeros((len(detections), len(current_boxes)))
        vehicle_ids = list(current_boxes.keys())
        
        for i, detection in enumerate(detections):
            for j, vehicle_id in enumerate(vehicle_ids):
                last_box = current_boxes[vehicle_id]
                # Higher IoU = better match, so use 1-IoU as cost
                cost_matrix[i, j] = 1.0 - self._calculate_iou(detection['box'], last_box)
        
        # Use Hungarian algorithm to find optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches with high costs (low IoU)
        matches = {}
        for row_idx, col_idx in zip(row_indices, col_indices):
            cost = cost_matrix[row_idx, col_idx]
            # Only keep matches where IoU > threshold
            if cost < (1.0 - self.iou_threshold):
                matches[row_idx] = vehicle_ids[col_idx]
                
        return matches
    
    def _update_vehicle_dynamics(self, vehicle_id: int, box: List[float]):
        """Update position, velocity and acceleration for a vehicle"""
        center = self._get_box_center(box)
        
        # Store position in trajectory history
        if vehicle_id not in self.trajectories:
            self.trajectories[vehicle_id] = deque(maxlen=self.track_history)
        
        self.trajectories[vehicle_id].append(center)
        
        # Update velocity (if we have at least 2 points)
        if len(self.trajectories[vehicle_id]) >= 2:
            prev_pos = self.trajectories[vehicle_id][-2]
            curr_pos = center
            
            # Simple velocity calculation (pixels per frame)
            vx = curr_pos[0] - prev_pos[0]
            vy = curr_pos[1] - prev_pos[1]
            self.velocities[vehicle_id] = (vx, vy)
            
            # Update acceleration (if we have at least 3 points)
            if len(self.trajectories[vehicle_id]) >= 3 and vehicle_id in self.velocities:
                if vehicle_id not in self.accelerations:
                    self.accelerations[vehicle_id] = (0, 0)
                
                prev_vx, prev_vy = self.velocities.get(vehicle_id, (0, 0))
                ax = vx - prev_vx
                ay = vy - prev_vy
                self.accelerations[vehicle_id] = (ax, ay)
    
    def _extrapolate_trajectory(self, vehicle_id: int, time_steps: int = 10) -> List[Tuple[float, float]]:
        """
        Predict future positions based on current dynamics
        Using a simple physics model with acceleration
        """
        if vehicle_id not in self.trajectories or len(self.trajectories[vehicle_id]) < 2:
            return []
        
        current_pos = self.trajectories[vehicle_id][-1]
        vx, vy = self.velocities.get(vehicle_id, (0, 0))
        ax, ay = self.accelerations.get(vehicle_id, (0, 0))
        
        future_positions = []
        x, y = current_pos
        
        for t in range(1, time_steps + 1):
            # Simple physics: x = x0 + v*t + 0.5*a*t^2
            future_x = x + vx * t + 0.5 * ax * t * t
            future_y = y + vy * t + 0.5 * ay * t * t
            future_positions.append((future_x, future_y))
        
        return future_positions
    
    def _calculate_time_to_collision(self, vehicle1_id: int, vehicle2_id: int) -> Optional[float]:
        """
        Calculate approximate time to collision between two vehicles
        Returns time in frames or None if no collision is predicted
        """
        # Extract current positions and velocities
        if (vehicle1_id not in self.trajectories or vehicle2_id not in self.trajectories or
            len(self.trajectories[vehicle1_id]) < 2 or len(self.trajectories[vehicle2_id]) < 2):
            return None
        
        pos1 = self.trajectories[vehicle1_id][-1]
        pos2 = self.trajectories[vehicle2_id][-1]
        vel1 = self.velocities.get(vehicle1_id, (0, 0))
        vel2 = self.velocities.get(vehicle2_id, (0, 0))
        
        # Calculate relative position and velocity
        rel_pos_x = pos2[0] - pos1[0]
        rel_pos_y = pos2[1] - pos1[1]
        rel_vel_x = vel2[0] - vel1[0]
        rel_vel_y = vel2[1] - vel1[1]
        
        # Check if relative velocity is very small
        rel_speed = math.sqrt(rel_vel_x**2 + rel_vel_y**2)
        if rel_speed < 0.5:  # Threshold for minimal relative movement
            return None
        
        # Compute time to closest approach (in frames)
        t_closest = -(rel_pos_x * rel_vel_x + rel_pos_y * rel_vel_y) / (rel_vel_x**2 + rel_vel_y**2)
        
        # If closest approach is in the past, no collision
        if t_closest <= 0:
            return None
        
        # Calculate distance at closest approach
        closest_distance_x = rel_pos_x + rel_vel_x * t_closest
        closest_distance_y = rel_pos_y + rel_vel_y * t_closest
        closest_distance = math.sqrt(closest_distance_x**2 + closest_distance_y**2)
        
        # Check if closest approach is within collision threshold
        vehicle1_box = self.bounding_boxes.get(vehicle1_id)
        vehicle2_box = self.bounding_boxes.get(vehicle2_id)
        
        if vehicle1_box and vehicle2_box:
            # Approximate vehicle radius as average of width/height
            v1_radius = (vehicle1_box[2] + vehicle1_box[3]) / 4
            v2_radius = (vehicle2_box[2] + vehicle2_box[3]) / 4
            collision_radius = v1_radius + v2_radius
            
            if closest_distance < collision_radius:
                return t_closest
        
        return None
    
    def _calculate_trajectory_angle(self, vehicle1_id: int, vehicle2_id: int) -> Optional[float]:
        """Calculate angle between trajectories of two vehicles"""
        if (vehicle1_id not in self.velocities or 
            vehicle2_id not in self.velocities):
            return None
        
        v1 = self.velocities[vehicle1_id]
        v2 = self.velocities[vehicle2_id]
        
        # Calculate magnitudes
        mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        # Check if either vehicle is stationary
        if mag_v1 < 0.5 or mag_v2 < 0.5:
            return None
        
        # Calculate dot product and angle
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        cos_angle = dot_product / (mag_v1 * mag_v2)
        
        # Clamp value to valid range for arccos
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        # Calculate angle in degrees
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def detect_vehicles(self, frame: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect vehicles and potential collisions in a frame
        Returns: (detections, collisions)
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Run inference with YOLOv8
        results = self.model(frame, conf=self.confidence_threshold)[0]
        
        # Process detections
        detections = []
        current_boxes = {}
        
        # First, collect all vehicle detections
        for i, result in enumerate(results.boxes.data.tolist()):
            x1, y1, x2, y2, confidence, class_id = result
            class_id = int(class_id)
            
            if class_id in self.vehicle_classes:
                # Convert to x, y, w, h format
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                box = [x, y, w, h]
                
                detections.append({
                    'box': box,
                    'class': self.vehicle_classes[class_id],
                    'class_id': class_id,
                    'confidence': confidence,
                    'det_index': i
                })
        
        # Update current boxes from existing trajectories
        for vehicle_id in self.trajectories:
            if vehicle_id in self.bounding_boxes:
                current_boxes[vehicle_id] = self.bounding_boxes[vehicle_id]
        
        # Match detections to existing tracked vehicles
        matches = self._match_detections_to_tracks(detections, current_boxes)
        
        # Update tracked vehicles
        updated_vehicle_ids = set()
        for i, detection in enumerate(detections):
            # Check if this detection matched an existing vehicle
            if i in matches:
                vehicle_id = matches[i]
                updated_vehicle_ids.add(vehicle_id)
            else:
                # New vehicle detected
                vehicle_id = self.next_vehicle_id
                self.next_vehicle_id += 1
                self.vehicle_classes_detected[vehicle_id] = detection['class']
            
            # Update vehicle state
            box = detection['box']
            self.bounding_boxes[vehicle_id] = box
            self.last_seen[vehicle_id] = self.frame_count
            self._update_vehicle_dynamics(vehicle_id, box)
            
            # Add vehicle ID to detection info
            detection['vehicle_id'] = vehicle_id
        
        # Remove vehicles that haven't been seen recently
        current_vehicle_ids = list(self.bounding_boxes.keys())
        for vehicle_id in current_vehicle_ids:
            if self.frame_count - self.last_seen.get(vehicle_id, 0) > 5:  # 5 frame threshold
                # Remove from tracking
                self.bounding_boxes.pop(vehicle_id, None)
                self.velocities.pop(vehicle_id, None)
                self.accelerations.pop(vehicle_id, None)
                # Keep trajectory for visualization but could remove if memory is a concern
        
        # Detect collisions
        collisions = []
        vehicle_ids = list(self.bounding_boxes.keys())
        
        for i in range(len(vehicle_ids)):
            for j in range(i + 1, len(vehicle_ids)):
                id1, id2 = vehicle_ids[i], vehicle_ids[j]
                
                # Skip if one of the vehicles is a traffic light
                class1 = self.vehicle_classes_detected.get(id1)
                class2 = self.vehicle_classes_detected.get(id2)
                if class1 == "traffic light" or class2 == "traffic light":
                    continue
                
                box1 = self.bounding_boxes[id1]
                box2 = self.bounding_boxes[id2]
                
                # Current proximity check
                distance = self._calculate_distance(box1, box2)
                iou = self._calculate_iou(box1, box2)
                
                # Calculate time to collision
                ttc = self._calculate_time_to_collision(id1, id2)
                
                # Calculate angle between trajectories
                angle = self._calculate_trajectory_angle(id1, id2)
                approaching = False
                if angle is not None:
                    # Vehicles are approaching if angle > 90° (moving toward each other)
                    approaching = angle > 90
                
                # Determine if collision is detected
                collision_detected = False
                collision_type = "unknown"
                
                # Immediate collision (vehicles already close/overlapping)
                if distance < self.collision_thresholds["distance"] or iou > self.collision_thresholds["iou"]:
                    collision_detected = True
                    collision_type = "immediate"
                
                # Predicted collision based on trajectories
                elif ttc is not None and ttc < self.collision_thresholds["time_to_collision"] and approaching:
                    collision_detected = True
                    collision_type = "predicted"
                
                if collision_detected:
                    collisions.append({
                        'vehicle1_id': id1,
                        'vehicle2_id': id2,
                        'vehicle1_class': class1,
                        'vehicle2_class': class2,
                        'boxes': [box1, box2],
                        'distance': distance,
                        'time_to_collision': ttc,
                        'type': collision_type,
                        'approaching_angle': angle,
                        'confidence': 1.0 - (ttc / self.collision_thresholds["time_to_collision"] 
                                           if ttc is not None else 0.5)
                    })
                    
                    # Update collision history
                    collision_pair = tuple(sorted([id1, id2]))
                    self.collision_history[collision_pair] = {
                        'first_detected': self.collision_history.get(collision_pair, {}).get('first_detected', self.frame_count),
                        'last_detected': self.frame_count,
                        'type': collision_type,
                        'confidence': collisions[-1]['confidence']
                    }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Add performance metadata
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        return detections, collisions
    
    def draw_detections_and_collisions(self, frame: np.ndarray, detections: List[Dict], 
                                     collisions: List[Dict]) -> np.ndarray:
        """Draw detections and collision warnings on the frame with enhanced visuals"""
        # Create a copy of the frame for drawing
        visualization = frame.copy()
        
        # First draw all trajectories (behind detections)
        if self.debug_mode:
            for vehicle_id, trajectory in self.trajectories.items():
                if len(trajectory) < 2:
                    continue
                
                # Draw trajectory as connected line segments
                points = np.array(list(trajectory), dtype=np.int32)
                # Color based on vehicle class
                if vehicle_id in self.vehicle_classes_detected:
                    class_name = self.vehicle_classes_detected[vehicle_id]
                    color = self._get_class_color(class_name)
                else:
                    color = (0, 255, 0)  # Default green
                
                # Draw trajectory lines with increasing thickness to show direction
                for i in range(len(points) - 1):
                    thickness = int(i / 3) + 1  # Thicker lines for more recent positions
                    cv2.line(visualization, 
                            tuple(points[i]), 
                            tuple(points[i+1]), 
                            color, 
                            thickness)
                
                # Draw future trajectory if available
                if vehicle_id in self.velocities:
                    future_positions = self._extrapolate_trajectory(vehicle_id, 10)
                    if future_positions:
                        # Draw predicted path as dashed line
                        last_pos = trajectory[-1]
                        for i, future_pos in enumerate(future_positions):
                            # Use dashed line for predictions
                            if i % 2 == 0:  # Skip every other segment for dash effect
                                start_point = last_pos if i == 0 else future_positions[i-1]
                                cv2.line(visualization, 
                                       (int(start_point[0]), int(start_point[1])), 
                                       (int(future_pos[0]), int(future_pos[1])), 
                                       (255, 255, 0), 1, cv2.LINE_AA)
                            last_pos = future_pos
        
        # Draw all vehicle detections
        for detection in detections:
            x, y, w, h = detection['box']
            
            # Get class information for color coding
            class_name = detection['class']
            color = self._get_class_color(class_name)
            
            # Format label with confidence
            label = f"{class_name} {detection['confidence']:.2f}"
            
            # Draw rectangle
            cv2.rectangle(visualization, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(visualization, (x, y - 20), (x + label_size[0], y), color, -1)
            cv2.putText(visualization, label, (x, y - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw vehicle ID if in debug mode
            if self.debug_mode and 'vehicle_id' in detection:
                cv2.putText(visualization, f"ID: {detection['vehicle_id']}", 
                          (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Then draw collision warnings
        for collision in collisions:
            # Draw boxes with red for potential collision
            for box in collision['boxes']:
                x, y, w, h = box
                
                # Collision coloring based on type
                if collision['type'] == 'immediate':
                    border_color = (0, 0, 255)  # Red for immediate collision
                else:
                    border_color = (0, 140, 255)  # Orange for predicted collision
                
                # Draw thicker rectangle for collision warning
                cv2.rectangle(visualization, (x, y), (x + w, y + h), border_color, 3)
            
            # Draw line between colliding vehicles
            box1, box2 = collision['boxes']
            center1 = (int(box1[0] + box1[2]/2), int(box1[1] + box1[3]/2))
            center2 = (int(box2[0] + box2[2]/2), int(box2[1] + box2[3]/2))
            
            # Use dashed line for predicted collisions, solid for immediate
            if collision['type'] == 'predicted':
                # Draw dashed line
                dash_length = 10
                dist = math.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)
                dashes = int(dist / dash_length)
                if dashes > 1:
                    for i in range(dashes):
                        x1 = int(center1[0] + (center2[0] - center1[0]) * i / dashes)
                        y1 = int(center1[1] + (center2[1] - center1[1]) * i / dashes)
                        x2 = int(center1[0] + (center2[0] - center1[0]) * (i + 0.5) / dashes)
                        y2 = int(center1[1] + (center2[1] - center1[1]) * (i + 0.5) / dashes)
                        cv2.line(visualization, (x1, y1), (x2, y2), (0, 140, 255), 2)
            else:
                # Solid line for immediate collision
                cv2.line(visualization, center1, center2, (0, 0, 255), 2)
            
            # Draw warning text with background
            mid_point = ((center1[0] + center2[0])//2, (center1[1] + center2[1])//2)
            
            # Different warnings based on collision type
            confidence = collision.get('confidence', 0.5) * 100
            
            if collision['type'] == 'immediate':
                warning = f"COLLISION WARNING! ({confidence:.0f}%)"
                color = (0, 0, 255)  # Red
            else:
                ttc = collision.get('time_to_collision')
                if ttc is not None:
                    warning = f"COLLISION PREDICTED: {ttc:.1f}s ({confidence:.0f}%)"
                else:
                    warning = f"POTENTIAL COLLISION ({confidence:.0f}%)"
                color = (0, 140, 255)  # Orange
            
            # Draw text with background for better visibility
            text_size, _ = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(visualization, 
                        (mid_point[0] - text_size[0]//2 - 5, mid_point[1] - text_size[1] - 5),
                        (mid_point[0] + text_size[0]//2 + 5, mid_point[1] + 5),
                        color, -1)
            cv2.putText(visualization, warning, 
                      (mid_point[0] - text_size[0]//2, mid_point[1]),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add performance metrics if in debug mode
        if self.debug_mode and self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            cv2.putText(visualization, f"FPS: {fps:.1f}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(visualization, f"Det: {len(detections)}", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(visualization, f"Col: {len(collisions)}", (10, 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return visualization
    
    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get a consistent color for a vehicle class"""
        color_map = {
            "car": (0, 255, 0),       # Green
            "motorcycle": (0, 255, 255),  # Yellow
            "bus": (255, 0, 0),       # Blue
            "truck": (255, 0, 255),   # Magenta
            "boat": (0, 165, 255),    # Orange
            "traffic light": (0, 255, 255)  # Yellow
        }
        return color_map.get(class_name, (0, 255, 0))
    
    def get_performance_stats(self) -> Dict:
        """Return performance statistics"""
        if not self.processing_times:
            return {
                "fps": 0,
                "avg_processing_time": 0,
                "num_vehicles_tracked": 0,
                "num_collisions_detected": 0
            }
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        return {
            "fps": 1.0 / avg_time if avg_time > 0 else 0,
            "avg_processing_time": avg_time,
            "num_vehicles_tracked": len(self.bounding_boxes),
            "num_collisions_detected": len(self.collision_history)
        }
    
    def toggle_debug_mode(self):
        """Toggle debug visualization mode"""
        self.debug_mode = not self.debug_mode
        return self.debug_mode
    
    def save_debug_frame(self, frame: np.ndarray, filename: str):
        """Save a debug frame with all visualizations enabled"""
        old_debug_state = self.debug_mode
        self.debug_mode = True
        
        # Detect vehicles and collisions
        detections, collisions = self.detect_vehicles(frame)
        
        # Draw all debug information
        debug_frame = self.draw_detections_and_collisions(frame, detections, collisions)
        
        # Save the frame
        cv2.imwrite(filename, debug_frame)
        
        # Restore debug state
        self.debug_mode = old_debug_state
        
        return debug_frame
    
    def analyze_video(self, video_path: str, output_path: str = None, start_frame: int = 0, 
                     max_frames: int = None, save_frames: bool = False, frames_dir: str = None):
        """
        Analyze a video file for vehicle collisions
        
        Args:
            video_path: Path to input video
            output_path: Path to save processed video (optional)
            start_frame: Frame to start analysis from
            max_frames: Maximum number of frames to process
            save_frames: Whether to save individual frames
            frames_dir: Directory to save frames to
        
        Returns:
            Summary statistics and collision events
        """
        # Reset tracking
        self.reset_tracking()
        
        # Open the input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video if requested
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Setup frames directory if saving frames
        if save_frames and frames_dir:
            os.makedirs(frames_dir, exist_ok=True)
        
        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames
        frame_count = 0
        collision_events = []
        active_collisions = set()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if we've reached max frames
                if max_frames and frame_count >= max_frames:
                    break
                
                # Process this frame
                detections, collisions = self.detect_vehicles(frame)
                
                # Track collision events
                current_collisions = set()
                for collision in collisions:
                    collision_pair = tuple(sorted([collision['vehicle1_id'], collision['vehicle2_id']]))
                    current_collisions.add(collision_pair)
                    
                    # Check if this is a new collision
                    if collision_pair not in active_collisions:
                        active_collisions.add(collision_pair)
                        # Record collision event
                        collision_events.append({
                            'frame': frame_count + start_frame,
                            'time': (frame_count + start_frame) / fps,
                            'vehicle1_id': collision['vehicle1_id'],
                            'vehicle2_id': collision['vehicle2_id'],
                            'vehicle1_class': collision['vehicle1_class'],
                            'vehicle2_class': collision['vehicle2_class'],
                            'type': collision['type'],
                            'confidence': collision.get('confidence', 0.5)
                        })
                
                # Remove inactive collisions
                inactive_collisions = active_collisions - current_collisions
                active_collisions = current_collisions
                
                # Draw visualizations
                vis_frame = self.draw_detections_and_collisions(frame, detections, collisions)
                
                # Save to output video
                if out:
                    out.write(vis_frame)
                
                # Save frame if requested
                if save_frames and frames_dir:
                    frame_path = os.path.join(frames_dir, f"frame_{frame_count + start_frame:06d}.jpg")
                    cv2.imwrite(frame_path, vis_frame)
                
                frame_count += 1
                
                # Print progress
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames ({frame_count/total_frames*100:.2f}%)")
        
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            raise
        
        finally:
            # Release resources
            cap.release()
            if out:
                out.write
                out.release()
        
        # Return analysis summary
        performance_stats = self.get_performance_stats()
        return {
            "video_info": {
                "path": video_path,
                "fps": fps,
                "resolution": (width, height),
                "total_frames": total_frames,
                "processed_frames": frame_count
            },
            "performance": performance_stats,
            "collisions": {
                "total_events": len(collision_events),
                "events": collision_events
            }
        }
    
    def analyze_image(self, image_path: str, output_path: str = None) -> Dict:
        """
        Analyze a single image for vehicle collision potential
        
        Args:
            image_path: Path to input image
            output_path: Path to save processed image (optional)
            
        Returns:
            Analysis results including detections and potential collisions
        """
        # Reset tracking since this is a single image
        self.reset_tracking()
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Process the image
        detections, collisions = self.detect_vehicles(image)
        
        # Visualize results
        vis_image = self.draw_detections_and_collisions(image, detections, collisions)
        
        # Save output if requested
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        # Format results
        results = {
            "image_info": {
                "path": image_path,
                "resolution": (image.shape[1], image.shape[0])
            },
            "detections": {
                "count": len(detections),
                "vehicles": [
                    {
                        "class": d["class"],
                        "confidence": d["confidence"],
                        "box": d["box"]
                    }
                    for d in detections
                ]
            },
            "collisions": {
                "count": len(collisions),
                "warnings": [
                    {
                        "vehicle1_class": c["vehicle1_class"],
                        "vehicle2_class": c["vehicle2_class"],
                        "type": c["type"],
                        "confidence": c.get("confidence", 0.5)
                    }
                    for c in collisions
                ]
            }
        }
        
        return results
    
    def configure(self, **kwargs):
        """Update configuration parameters"""
        # Update collision thresholds
        if "collision_thresholds" in kwargs:
            threshold_updates = kwargs.pop("collision_thresholds")
            if isinstance(threshold_updates, dict):
                for key, value in threshold_updates.items():
                    if key in self.collision_thresholds:
                        self.collision_thresholds[key] = value
        
        # Update other parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        return {
            "collision_thresholds": self.collision_thresholds,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "track_history": self.track_history,
            "debug_mode": self.debug_mode
        }
    
    def process_video_stream(self, stream_url: str, callback=None):
        """
        Process video stream in real-time
        
        Args:
            stream_url: URL or camera index for the video stream
            callback: Function to call with each processed frame and results
        """
        # Reset tracking
        self.reset_tracking()
        
        # Open video stream
        if isinstance(stream_url, int) or stream_url.isdigit():
            # Camera index
            cap = cv2.VideoCapture(int(stream_url))
        else:
            # URL
            cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream: {stream_url}")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                detections, collisions = self.detect_vehicles(frame)
                
                # Visualize
                vis_frame = self.draw_detections_and_collisions(frame, detections, collisions)
                
                # Call callback if provided
                if callback:
                    callback(vis_frame, detections, collisions)
                
                # Exit if 'q' pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()