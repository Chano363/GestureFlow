import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import time

from utils.enums import GestureType, SwipeDirection


class Drawer:
    """
    Drawing tool class for drawing various elements on images
    """
    
    def __init__(self):
        """Initialize drawing tool"""
        # Color definitions (BGR format)
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'magenta': (255, 0, 255),
            'cyan': (255, 255, 0),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'gray': (128, 128, 128),
            'orange': (0, 165, 255),
            'purple': (128, 0, 128)
        }
        
        # Colors corresponding to gesture types
        self.gesture_colors = {
            GestureType.NONE: self.colors['gray'],
            GestureType.POINT: self.colors['blue'],
            GestureType.CLICK: self.colors['green'],
            GestureType.DOUBLE_CLICK: self.colors['green'],
            GestureType.LONG_PRESS: self.colors['red'],
            GestureType.SWIPE_LEFT: self.colors['yellow'],
            GestureType.SWIPE_RIGHT: self.colors['yellow'],
            GestureType.SWIPE_UP: self.colors['yellow'],
            GestureType.SWIPE_DOWN: self.colors['yellow'],
            GestureType.ZOOM_IN: self.colors['magenta'],
            GestureType.ZOOM_OUT: self.colors['magenta'],
            GestureType.ROTATE_LEFT: self.colors['cyan']
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.font_color = self.colors['white']
        
        # Line settings
        self.line_thickness = 2
        self.line_type = cv2.LINE_AA
        
        # Trajectory history
        self.trajectory_history: Dict[int, List[Tuple[int, int, float]]] = {}
        self.max_trajectory_length = 30
    
    
    def draw_bbox(self, image: np.ndarray, bbox: List[int], track_id: int, 
                  color: Tuple[int, int, int] = None, thickness: int = None) -> None:
        """
        Draw bounding box
        
        Args:
            image: Target image
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: Tracking ID
            color: Bounding box color
            thickness: Bounding box thickness
        """
        if color is None:
            color = self.colors['blue']
        if thickness is None:
            thickness = self.line_thickness
        
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw tracking ID
        id_text = f"ID: {track_id}"
        cv2.putText(image, id_text, (x1, y1 - 10), self.font, self.font_scale, 
                   self.font_color, self.font_thickness, self.line_type)
    
    def draw_label(self, image: np.ndarray, bbox: List[int], label: str, 
                  color: Tuple[int, int, int] = None) -> None:
        """
        Draw label
        
        Args:
            image: Target image
            bbox: Bounding box [x1, y1, x2, y2]
            label: Label text
            color: Label background color
        """
        if color is None:
            color = self.colors['blue']
        
        x1, y1, x2, y2 = bbox
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(label, self.font, 
                                                             self.font_scale, self.font_thickness)
        
        # Draw label background
        cv2.rectangle(image, (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - baseline - 2), self.font, 
                   self.font_scale, self.font_color, self.font_thickness, self.line_type)
    
    def draw_gesture_info(self, image: np.ndarray, bbox: List[int], info: str, 
                         color: Tuple[int, int, int] = None) -> None:
        """
        Draw gesture information
        
        Args:
            image: Target image
            bbox: Bounding box [x1, y1, x2, y2]
            info: Information text
            color: Text color
        """
        if color is None:
            color = self.colors['white']
        
        x1, y1, x2, y2 = bbox
        
        # Draw information below bounding box
        cv2.putText(image, info, (x1, y2 + 20), self.font, self.font_scale, 
                   color, self.font_thickness, self.line_type)
    
    def draw_trajectory(self, image: np.ndarray, track_id: int, point: Tuple[int, int], 
                       color: Tuple[int, int, int] = None) -> None:
        """
        Draw trajectory
        
        Args:
            image: Target image
            track_id: Tracking ID
            point: Current point (x, y)
            color: Trajectory color
        """
        if color is None:
            color = self.colors['cyan']
        
        # Add point to trajectory history
        current_time = time.time()
        if track_id not in self.trajectory_history:
            self.trajectory_history[track_id] = []
        
        self.trajectory_history[track_id].append((point[0], point[1], current_time))
        
        # Limit trajectory length
        if len(self.trajectory_history[track_id]) > self.max_trajectory_length:
            self.trajectory_history[track_id] = self.trajectory_history[track_id][-self.max_trajectory_length:]
        
        # Draw trajectory
        trajectory = self.trajectory_history[track_id]
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                # Calculate transparency (newer points are less transparent)
                alpha = i / len(trajectory)
                thickness = int(1 + alpha * 2)
                
                # Draw line segment
                cv2.line(image, (trajectory[i-1][0], trajectory[i-1][1]), 
                        (trajectory[i][0], trajectory[i][1]), color, thickness)
        
        # Draw current point
        cv2.circle(image, point, 5, color, -1)
    
    def draw_gesture_icon(self, image: np.ndarray, center: Tuple[int, int], 
                         gesture_type: GestureType, size: int = 50) -> None:
        """
        Draw gesture icon
        
        Args:
            image: Target image
            center: Center point (x, y)
            gesture_type: Gesture type
            size: Icon size
        """
        color = self.gesture_colors.get(gesture_type, self.colors['gray'])
        
        if gesture_type == GestureType.POINT:
            # Draw finger pointing icon
            cv2.circle(image, center, size // 2, color, -1)
            cv2.circle(image, center, size // 4, self.colors['white'], -1)
            
        elif gesture_type == GestureType.CLICK:
            # Draw click icon
            cv2.circle(image, center, size // 2, color, 2)
            cv2.line(image, (center[0] - size // 3, center[1]), 
                    (center[0] + size // 3, center[1]), color, 2)
            cv2.line(image, (center[0], center[1] - size // 3), 
                    (center[0], center[1] + size // 3), color, 2)
            
        elif gesture_type == GestureType.SWIPE:
            # Draw swipe icon
            start = (center[0] - size // 2, center[1])
            end = (center[0] + size // 2, center[1])
            cv2.arrowedLine(image, start, end, color, 3)
            
        elif gesture_type == GestureType.ZOOM:
            # Draw zoom icon
            cv2.circle(image, (center[0] - size // 4, center[1] - size // 4), 
                      size // 4, color, 2)
            cv2.circle(image, (center[0] + size // 4, center[1] + size // 4), 
                      size // 4, color, 2)
            
        elif gesture_type == GestureType.ROTATE:
            # Draw rotate icon
            cv2.circle(image, center, size // 2, color, 2)
            cv2.line(image, center, (center[0] + size // 2, center[1]), color, 2)
            
        elif gesture_type == GestureType.GRAB:
            # Draw grab icon
            cv2.circle(image, center, size // 2, color, -1)
            
        elif gesture_type == GestureType.RELEASE:
            # Draw release icon
            cv2.circle(image, center, size // 2, color, 2)
            
        else:
            # Default draw circle
            cv2.circle(image, center, size // 2, color, 2)
    
    def draw_swipe_direction(self, image: np.ndarray, start_point: Tuple[int, int], 
                           end_point: Tuple[int, int], direction: SwipeDirection, 
                           color: Tuple[int, int, int] = None) -> None:
        """
        Draw swipe direction
        
        Args:
            image: Target image
            start_point: Start point (x, y)
            end_point: End point (x, y)
            direction: Swipe direction
            color: Arrow color
        """
        if color is None:
            color = self.colors['yellow']
        
        # Draw arrow
        cv2.arrowedLine(image, start_point, end_point, color, 3)
        
        # Draw direction text
        direction_text = ""
        if direction == SwipeDirection.LEFT:
            direction_text = "LEFT"
        elif direction == SwipeDirection.RIGHT:
            direction_text = "RIGHT"
        elif direction == SwipeDirection.UP:
            direction_text = "UP"
        elif direction == SwipeDirection.DOWN:
            direction_text = "DOWN"
        elif direction == SwipeDirection.HORIZONTAL:
            direction_text = "HORIZONTAL"
        elif direction == SwipeDirection.VERTICAL:
            direction_text = "VERTICAL"
        
        if direction_text:
            text_pos = (end_point[0] + 10, end_point[1] - 10)
            cv2.putText(image, direction_text, text_pos, self.font, self.font_scale, 
                       color, self.font_thickness, self.line_type)
    
    def draw_fps(self, image: np.ndarray, fps: float, 
                position: Tuple[int, int] = (10, 30), 
                color: Tuple[int, int, int] = None) -> None:
        """
        Draw FPS information
        
        Args:
            image: Target image
            fps: FPS value
            position: Text position (x, y)
            color: Text color
        """
        if color is None:
            color = self.colors['green']
        
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(image, fps_text, position, self.font, self.font_scale, 
                   color, self.font_thickness, self.line_type)
    
    def draw_status(self, image: np.ndarray, status: str, 
                   position: Tuple[int, int] = (10, 60), 
                   color: Tuple[int, int, int] = None) -> None:
        """
        Draw status information
        
        Args:
            image: Target image
            status: Status text
            position: Text position (x, y)
            color: Text color
        """
        if color is None:
            color = self.colors['white']
        
        cv2.putText(image, status, position, self.font, self.font_scale, 
                   color, self.font_thickness, self.line_type)
    
    def clear_trajectory(self, track_id: int = None) -> None:
        """
        Clear trajectory
        
        Args:
            track_id: Tracking ID, if None clear all trajectories
        """
        if track_id is None:
            self.trajectory_history.clear()
        elif track_id in self.trajectory_history:
            del self.trajectory_history[track_id]
    
    def get_color(self, color_name: str) -> Tuple[int, int, int]:
        """
        Get color
        
        Args:
            color_name: Color name
            
        Returns:
            Color value (B, G, R)
        """
        return self.colors.get(color_name, self.colors['white'])
    
    def get_gesture_color(self, gesture_type: GestureType) -> Tuple[int, int, int]:
        """
        Get color corresponding to gesture type
        
        Args:
            gesture_type: Gesture type
            
        Returns:
            Color value (B, G, R)
        """
        return self.gesture_colors.get(gesture_type, self.colors['gray'])