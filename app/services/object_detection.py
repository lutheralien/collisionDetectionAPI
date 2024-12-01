from ultralytics import YOLO
import cv2
import numpy as np
import os
from typing import List, Dict, Tuple
from collections import deque
import math

class ObjectDetector:
    def __init__(self):
        # Path for the YOLOv8 model
        self.model_path = "models/yolov8n.pt"
        
        # Ensure model exists
        self._ensure_model_files()
        
        # Load YOLO model
        self.model = YOLO(self.model_path)
        
        # Vehicle classes (COCO dataset)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Tracking and collision detection parameters
        self.collision_threshold = 0.3  # IOU threshold for collision detection
        self.trajectory_points = 30  # Number of points to keep in trajectory
        self.min_distance = 50  # Minimum distance (pixels) for collision warning
        
        # Dictionary to store vehicle trajectories
        self.trajectories = {}
        self.next_vehicle_id = 0
    
    def _ensure_model_files(self):
        """Download YOLOv8 model if it doesn't exist"""
        if not os.path.exists(self.model_path):
            os.makedirs("models", exist_ok=True)
            self.model = YOLO('yolov8n.pt')
            self.model.save(self.model_path)
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection coordinates
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _calculate_distance(self, box1: List[int], box2: List[int]) -> float:
        """Calculate distance between centers of two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _predict_collision(self, traj1: deque, traj2: deque) -> bool:
        """Predict if two vehicles will collide based on their trajectories"""
        if len(traj1) < 2 or len(traj2) < 2:
            return False
            
        # Calculate velocity vectors
        v1 = (traj1[-1][0] - traj1[-2][0], traj1[-1][1] - traj1[-2][1])
        v2 = (traj2[-1][0] - traj2[-2][0], traj2[-1][1] - traj2[-2][1])
        
        # Project positions forward
        future_p1 = (traj1[-1][0] + v1[0], traj1[-1][1] + v1[1])
        future_p2 = (traj2[-1][0] + v2[0], traj2[-1][1] + v2[1])
        
        # Calculate distance between predicted positions
        distance = math.sqrt((future_p1[0] - future_p2[0])**2 + (future_p1[1] - future_p2[1])**2)
        
        return distance < self.min_distance
    
    def detect_vehicles(self, frame: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect vehicles and potential collisions in a frame
        Returns: (detections, collisions)
        """
        # Run inference
        results = self.model(frame, conf=0.5)[0]
        
        # Process detections
        detections = []
        current_boxes = {}
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            
            if int(class_id) in self.vehicle_classes:
                # Convert to x, y, w, h format
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                box = [x, y, w, h]
                
                # Update trajectories
                matched = False
                for vehicle_id, trajectory in self.trajectories.items():
                    if len(trajectory) > 0:
                        last_box = trajectory[-1]
                        if self._calculate_iou(box, last_box) > 0.3:  # IOU threshold for tracking
                            trajectory.append(box)
                            if len(trajectory) > self.trajectory_points:
                                trajectory.popleft()
                            current_boxes[vehicle_id] = box
                            matched = True
                            break
                
                if not matched:
                    # New vehicle detected
                    self.trajectories[self.next_vehicle_id] = deque([box], maxlen=self.trajectory_points)
                    current_boxes[self.next_vehicle_id] = box
                    self.next_vehicle_id += 1
                
                detections.append({
                    'box': box,
                    'class': results.names[int(class_id)],
                    'confidence': confidence
                })
        
        # Remove trajectories of vehicles that are no longer visible
        self.trajectories = {k: v for k, v in self.trajectories.items() if k in current_boxes}
        
        # Detect collisions
        collisions = []
        vehicle_ids = list(current_boxes.keys())
        for i in range(len(vehicle_ids)):
            for j in range(i + 1, len(vehicle_ids)):
                id1, id2 = vehicle_ids[i], vehicle_ids[j]
                box1, box2 = current_boxes[id1], current_boxes[id2]
                
                # Check current proximity
                distance = self._calculate_distance(box1, box2)
                iou = self._calculate_iou(box1, box2)
                
                # Check trajectory intersection
                collision_predicted = self._predict_collision(
                    self.trajectories[id1], 
                    self.trajectories[id2]
                )
                
                if distance < self.min_distance or iou > self.collision_threshold or collision_predicted:
                    collisions.append({
                        'vehicle1': id1,
                        'vehicle2': id2,
                        'boxes': [box1, box2],
                        'distance': distance,
                        'type': 'predicted' if collision_predicted else 'immediate'
                    })
        
        return detections, collisions
    
    def draw_detections_and_collisions(self, frame: np.ndarray, detections: List[Dict], 
                                     collisions: List[Dict]) -> np.ndarray:
        """Draw detections and collision warnings on the frame"""
        # First draw all detections
        for detection in detections:
            x, y, w, h = detection['box']
            label = f"{detection['class']} {detection['confidence']:.2f}"
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label with background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - 20), (x + label_size[0], y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Then draw collision warnings
        for collision in collisions:
            for box in collision['boxes']:
                x, y, w, h = box
                # Draw red rectangle for collision warning
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            
            # Draw line between colliding vehicles
            box1, box2 = collision['boxes']
            center1 = (int(box1[0] + box1[2]/2), int(box1[1] + box1[3]/2))
            center2 = (int(box2[0] + box2[2]/2), int(box2[1] + box2[3]/2))
            cv2.line(frame, center1, center2, (0, 0, 255), 2)
            
            # Draw collision warning
            warning = f"COLLISION WARNING! ({collision['type']})"
            mid_point = ((center1[0] + center2[0])//2, (center1[1] + center2[1])//2)
            cv2.putText(frame, warning, mid_point,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame