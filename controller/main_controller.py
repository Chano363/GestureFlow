import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from detector.yolo_detector import YOLODetector
from tracker.sort_tracker import SORTTracker
from classifier.gesture_classifier import GestureClassifier
from utils.drawer import Drawer
from utils.config import Config


class MainController:
    """
    Main controller that coordinates detection, tracking and classification modules
    """

    def __init__(self, config: Config):
        """
        Initialize main controller

        Args:
            config: Configuration object
        """
        self.config = config

        device = self.config.get("detector.device", "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, falling back to CPU")
            device = "cpu"

        print(f"Using device: {device}")

        self.detector = YOLODetector(
            model_path=self.config.get(
                "detector.model_path", "models/YOLOv10x_gestures.pt"
            ),
            confidence_threshold=self.config.get("detector.confidence_threshold", 0.5),
            nms_threshold=self.config.get("detector.nms_threshold", 0.4),
            device=device,
        )

        self.tracker = SORTTracker(
            max_disappeared=self.config.get("tracker.max_disappeared", 30),
            min_hits=self.config.get("tracker.min_hits", 3),
            iou_threshold=self.config.get("tracker.iou_threshold", 0.3),
        )

        self.drawer = Drawer()

        self.gesture_history: Dict[int, List[Tuple[int, float, float]]] = {}

        self.gesture_states: Dict[int, Dict] = {}

        self.stability_frames = self.config.get(
            "gesture_params.hagrid_params.stability_frames", 5
        )
        self.stability_threshold = self.config.get(
            "gesture_params.hagrid_params.stability_threshold", 0.8
        )
        self.transition_frames = self.config.get(
            "gesture_params.hagrid_params.transition_frames", 3
        )
        self.min_gesture_duration = self.config.get(
            "gesture_params.hagrid_params.min_gesture_duration", 0.5
        )
        self.max_gesture_duration = self.config.get(
            "gesture_params.hagrid_params.max_gesture_duration", 5.0
        )

        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

    def _match_track_with_detections(
        self, track_bbox, gesture_detections, iou_threshold=0.5
    ):
        """
        Match tracked bounding box with original detections by maximum IOU

        Args:
            track_bbox: Tracked bounding box [x1, y1, x2, y2]
            gesture_detections: List of detections (x1, y1, x2, y2, class_id, confidence)
            iou_threshold: Minimum IOU threshold for matching

        Returns:
            Matched detection or None
        """
        max_iou = 0
        matched_detection = None

        for detection in gesture_detections:
            iou = self._calculate_iou(track_bbox, detection[:4])
            if iou > max_iou and iou >= iou_threshold:
                max_iou = iou
                matched_detection = detection

        return matched_detection

    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union of two bounding boxes

        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]

        Returns:
            IOU value
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 > x1 and y2 > y1:
            intersection = (x2 - x1) * (y2 - y1)
        else:
            intersection = 0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        if union > 0:
            return intersection / union
        else:
            return 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame

        Args:
            frame: Input frame

        Returns:
            (Processed frame, List of gesture information)
        """
        start_time = time.time()

        gesture_detections = self.detector.detect_gestures(frame)

        valid_detections = []
        for det in gesture_detections:
            x1, y1, x2, y2, class_id, confidence = det
            # Check if detection is valid
            if (
                not np.isnan(x1)
                and not np.isnan(y1)
                and not np.isnan(x2)
                and not np.isnan(y2)
                and not np.isinf(x1)
                and not np.isinf(y1)
                and not np.isinf(x2)
                and not np.isinf(y2)
                and x1 >= 0
                and y1 >= 0
                and x2 > x1
                and y2 > y1
                and confidence > 0.0
                and confidence <= 1.0
            ):
                valid_detections.append(det)

        tracks = self.tracker.update([det[:4] for det in gesture_detections])

        gesture_results = []
        for i, track in enumerate(tracks):
            track_id = track["id"]
            bbox = track["bbox"]

            matched_detection = self._match_track_with_detections(
                track["bbox"], gesture_detections
            )

            if matched_detection is not None:
                _, _, _, _, class_id, confidence = matched_detection

                self._update_gesture_history(track_id, class_id, confidence, start_time)

                smoothed_gesture = self._get_smoothed_gesture(track_id)

                self._update_gesture_state(track_id, smoothed_gesture, bbox)

                gesture_results.append(
                    {
                        "id": track_id,
                        "bbox": bbox,
                        "class_id": smoothed_gesture[0],
                        "class_name": self.detector.get_gesture_name(
                            smoothed_gesture[0]
                        ),
                        "confidence": smoothed_gesture[1],
                        "state": self.gesture_states.get(track_id, {}),
                    }
                )

        result_frame = self._draw_results(frame.copy(), gesture_results)

        self._update_fps()

        return result_frame, gesture_results

    def _update_gesture_history(
        self, track_id: int, class_id: int, confidence: float, timestamp: float
    ) -> None:
        """
        Update gesture history

        Args:
            track_id: Tracking ID
            class_id: Class ID
            confidence: Confidence
            timestamp: Timestamp
        """
        if track_id not in self.gesture_history:
            self.gesture_history[track_id] = []

        self.gesture_history[track_id].append((class_id, confidence, timestamp))

        current_time = time.time()
        self.gesture_history[track_id] = [
            record
            for record in self.gesture_history[track_id]
            if current_time - record[2] < 1.0
        ]

    def _get_smoothed_gesture(self, track_id: int) -> Tuple[int, float]:
        """
        Get smoothed gesture

        Args:
            track_id: Tracking ID

        Returns:
            (Smoothed class ID, Smoothed confidence)
        """
        if (
            track_id not in self.gesture_history
            or len(self.gesture_history[track_id]) == 0
        ):
            return 0, 0.0

        state = self.gesture_states.get(track_id, {})
        is_stable = state.get("is_stable", False)
        gesture_duration = state.get("gesture_duration", 0)

        class_counts = {}
        class_confidences = {}

        for class_id, confidence, _ in self.gesture_history[track_id]:
            if class_id not in class_counts:
                class_counts[class_id] = 0
                class_confidences[class_id] = []

            class_counts[class_id] += 1
            class_confidences[class_id].append(confidence)

        max_count = 0
        best_class_id = 0

        for class_id, count in class_counts.items():
            if count > max_count:
                max_count = count
                best_class_id = class_id

        avg_confidence = sum(class_confidences[best_class_id]) / len(
            class_confidences[best_class_id]
        )

        if not is_stable:
            avg_confidence *= 0.7

        if gesture_duration < self.min_gesture_duration:
            avg_confidence *= 0.8
        elif gesture_duration > self.max_gesture_duration:
            avg_confidence *= 0.9

        return best_class_id, avg_confidence

    def _update_gesture_state(
        self, track_id: int, gesture: Tuple[int, float], bbox: List[int]
    ) -> None:
        """
        Update gesture state

        Args:
            track_id: Tracking ID
            gesture: Gesture (class ID, confidence)
            bbox: Bounding box
        """
        class_id, confidence = gesture
        current_time = time.time()

        if track_id not in self.gesture_states:
            self.gesture_states[track_id] = {
                "prev_gesture": 0,
                "current_gesture": class_id,
                "gesture_start_time": current_time,
                "gesture_duration": 0,
                "stable_frames": 0,
                "is_stable": False,
                "bbox": bbox,
            }
        else:
            state = self.gesture_states[track_id]
            prev_gesture = state["current_gesture"]

            if prev_gesture != class_id:
                state["gesture_start_time"] = current_time
                state["stable_frames"] = 0
                state["is_stable"] = False

            state["prev_gesture"] = prev_gesture
            state["current_gesture"] = class_id
            state["gesture_duration"] = current_time - state["gesture_start_time"]
            state["bbox"] = bbox

            if confidence >= self.stability_threshold:
                state["stable_frames"] += 1
                if state["stable_frames"] >= self.stability_frames:
                    state["is_stable"] = True
            else:
                state["stable_frames"] = 0
                state["is_stable"] = False

    def _draw_results(
        self, frame: np.ndarray, gesture_results: List[Dict]
    ) -> np.ndarray:
        """
        Draw results on frame

        Args:
            frame: Input frame
            gesture_results: List of gesture results

        Returns:
            Drawn frame
        """
        for result in gesture_results:
            track_id = result["id"]
            bbox = result["bbox"]
            class_name = result["class_name"]
            confidence = result["confidence"]
            state = result["state"]

            self.drawer.draw_bbox(frame, bbox, track_id)

            label = f"{class_name}: {confidence:.2f}"
            self.drawer.draw_label(frame, bbox, label)

            if "gesture_duration" in state:
                duration = state["gesture_duration"]
                self.drawer.draw_gesture_info(frame, bbox, f"Duration: {duration:.2f}s")

        self.drawer.draw_fps(frame, self.current_fps)

        return frame

    def _update_fps(self) -> None:
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time

    def get_gesture_states(self) -> Dict[int, Dict]:
        """
        Get all gesture states

        Returns:
            Dictionary of gesture states
        """
        return self.gesture_states.copy()

    def reset(self) -> None:
        """Reset controller state"""
        self.gesture_history.clear()
        self.gesture_states.clear()
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
