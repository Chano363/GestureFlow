import numpy as np
from typing import List, Tuple, Dict
from scipy.optimize import linear_sum_assignment

from .track import Track


class SORTTracker:
    """
    SORT (Simple Online and Realtime Tracking) tracking algorithm implementation
    """
    
    def __init__(self, max_disappeared: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize SORT tracker
        
        Args:
            max_disappeared: Maximum number of frames a target can disappear
            min_hits: Minimum number of hits required to confirm a track
            iou_threshold: IoU threshold for matching detection boxes and tracking boxes
        """
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: List[Track] = []
        self.track_id_counter = 0
        
    def update(self, detections: List[List[int]]) -> List[Dict]:
        """
        Update tracker
        
        Args:
            detections: List of detection results, each element is [x1, y1, x2, y2]
            
        Returns:
            List of tracking results, each element is {
                'id': Tracking ID,
                'bbox': [x1, y1, x2, y2],
                'confidence': Confidence,
                'class_id': Class ID
            }
        """
        if len(detections) == 0:
            for track in self.tracks:
                track.update()
            
            self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_disappeared]
            
            return self._get_active_tracks()
        
        predicted_boxes = []
        for track in self.tracks:
            predicted_box = track.predict()
            predicted_boxes.append(predicted_box)
        
        iou_matrix = self._compute_iou_matrix(detections, predicted_boxes)
        
        if len(iou_matrix) > 0:
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)  
            
            matches = []
            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(range(len(self.tracks)))
            
            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] >= self.iou_threshold:
                    matches.append((row, col))
                    if row in unmatched_detections:
                        unmatched_detections.remove(row)
                    if col in unmatched_tracks:
                        unmatched_tracks.remove(col)
        else:
            matches = []
            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(range(len(self.tracks)))
        
        for detection_idx, track_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].update()
        
        for detection_idx in unmatched_detections:
            new_track = Track(
                track_id=self.track_id_counter,
                bbox=detections[detection_idx],
                max_age=self.max_disappeared,
                min_hits=self.min_hits
            )
            self.tracks.append(new_track)
            self.track_id_counter += 1
        
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_disappeared]
        
        return self._get_active_tracks()
    
    def _compute_iou_matrix(self, detections: List[List[int]], predicted_boxes: List[List[int]]) -> np.ndarray:
        """
        Calculate IoU matrix between detection boxes and prediction boxes
        
        Args:
            detections: List of detection boxes
            predicted_boxes: List of prediction boxes
            
        Returns:
            IoU matrix
        """
        iou_matrix = np.zeros((len(detections), len(predicted_boxes)))
        
        for i, det_box in enumerate(detections):
            for j, pred_box in enumerate(predicted_boxes):
                iou_matrix[i, j] = self._compute_iou(det_box, pred_box)
        
        return iou_matrix
    
    def _compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate IoU between two bounding boxes
        
        Args:
            box1: Bounding box 1 [x1, y1, x2, y2]
            box2: Bounding box 2 [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_active_tracks(self) -> List[Dict]:
        """
        Get all active tracking boxes
        
        Returns:
            List of active tracking boxes
        """
        active_tracks = []
        
        for track in self.tracks:
            is_valid, bbox = track.get_state()
            if is_valid:
                active_tracks.append({
                    'id': track.track_id,
                    'bbox': bbox,
                    'confidence': 1.0,  
                    'class_id': 0  
                })
        
        return active_tracks