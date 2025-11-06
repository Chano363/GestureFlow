import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict

from .base_detector import BaseDetector


class YOLODetector(BaseDetector):
    """
    YOLOv10 detector implementation that can complete both detection and classification tasks
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device: str = "cpu",
    ):
        """
        Initialize YOLO detector

        Args:
            model_path: Model file path
            confidence_threshold: Confidence threshold
            nms_threshold: NMS threshold
            device: Device to run model on (cpu or cuda)
        """
        super().__init__()
        try:
            from ultralytics import YOLO

            self.model = YOLO(model_path)
            self.device = device
            if torch.cuda.is_available() and device != "cpu":
                self.model.to(device)
        except ImportError:
            self.device = "cpu"
            self.model = torch.load(model_path, map_location="cpu")

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.gesture_classes = [
            "grabbing",
            "grip",
            "holy",
            "point",
            "call",
            "three3",
            "timeout",
            "xsign",
            "hand_heart",
            "hand_heart2",
            "little_finger",
            "middle_finger",
            "take_picture",
            "dislike",
            "fist",
            "four",
            "like",
            "mute",
            "ok",
            "one",
            "palm",
            "peace",
            "peace_inverted",
            "rock",
            "stop",
            "stop_inverted",
            "three",
            "three2",
            "two_up",
            "two_up_inverted",
            "three_gun",
            "thumb_index",
            "thumb_index2",
            "no_gesture",
        ]

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect objects in image

        Args:
            image: Input image

        Returns:
            List of detection results, each element is (x1, y1, x2, y2, confidence)
        """
        results = self.model(
            image, conf=self.confidence_threshold, iou=self.nms_threshold
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append((x1, y1, x2, y2, conf))

        return detections

    def detect_gestures(
        self, image: np.ndarray
    ) -> List[Tuple[int, int, int, int, int, float]]:
        """
        Detect and classify gestures in image - YOLOv10 can complete both detection and classification

        Args:
            image: Input image

        Returns:
            List of gesture detection results, each element is (x1, y1, x2, y2, gesture_class_id, confidence)
        """
        results = self.model(
            image, conf=self.confidence_threshold, iou=self.nms_threshold
        )

        gesture_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    class_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    gesture_detections.append((x1, y1, x2, y2, class_id, conf))

        return gesture_detections

    def get_gesture_name(self, class_id: int) -> str:
        """
        Get gesture name by class ID

        Args:
            class_id: Class ID

        Returns:
            Gesture name
        """
        if 0 <= class_id < len(self.gesture_classes):
            return self.gesture_classes[class_id]
        return "unknown"
