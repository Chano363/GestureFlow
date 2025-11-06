#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gesture Controller - Handles gesture recognition results and triggers corresponding actions
"""

import time
from typing import Dict, List, Callable, Optional, Tuple
from collections import defaultdict, deque

from utils.enums import GestureType, SwipeDirection, ZoomDirection, RotationDirection
from utils.config import Config


class GestureState:
    """Gesture state tracking class"""

    def __init__(self):
        self.gesture_type = None
        self.confidence = 0.0
        self.start_time = None
        self.last_update_time = None
        self.is_active = False
        self.position_history = deque(maxlen=10)
        self.stable_frames = 0

    def update(
        self, gesture_type: GestureType, confidence: float, position: Tuple[int, int]
    ):
        self.gesture_type = gesture_type
        self.confidence = confidence
        self.last_update_time = time.time()
        self.position_history.append(position)

        if not self.is_active:
            self.is_active = True
            self.start_time = self.last_update_time
            self.stable_frames = 1
        elif self.gesture_type == gesture_type:
            self.stable_frames += 1
        else:
            self.stable_frames = 1

    def reset(self):
        self.gesture_type = None
        self.confidence = 0.0
        self.start_time = None
        self.last_update_time = None
        self.is_active = False
        self.position_history.clear()
        self.stable_frames = 0

    def get_duration(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def get_position(self) -> Optional[Tuple[int, int]]:
        if not self.position_history:
            return None
        return self.position_history[-1]

    def get_movement_vector(self) -> Tuple[float, float]:
        if len(self.position_history) < 2:
            return (0.0, 0.0)

        start_pos = self.position_history[0]
        current_pos = self.position_history[-1]

        dx = current_pos[0] - start_pos[0]
        dy = current_pos[1] - start_pos[1]

        return (dx, dy)


class GestureController:

    def __init__(self, config: Config):
        self.config = config
        self.gesture_states: Dict[int, GestureState] = defaultdict(GestureState)
        self.gesture_callbacks: List[Callable] = []

        self.min_confidence = config.get("gesture_controller.min_confidence", 0.7)
        self.min_duration = config.get("gesture_controller.min_duration", 0.5)
        self.stability_frames = config.get("gesture_controller.stability_frames", 5)
        self.swipe_threshold = config.get("gesture_controller.swipe_threshold", 50)
        self.zoom_threshold = config.get("gesture_controller.zoom_threshold", 0.1)
        self.rotation_threshold = config.get(
            "gesture_controller.rotation_threshold", 15
        )

    def register_gesture_callback(self, callback: Callable):
        self.gesture_callbacks.append(callback)

    def _class_id_to_gesture_type(self, class_id: int) -> Optional[GestureType]:
        try:
            return GestureType(class_id)
        except ValueError:
            return None

    def _update_gesture_state(
        self, track_id: int, class_id: int, confidence: float, position: Tuple[int, int]
    ):
        gesture_type = self._class_id_to_gesture_type(class_id)

        if gesture_type is None:
            self.gesture_states[track_id].reset()
            return

        self.gesture_states[track_id].update(gesture_type, confidence, position)

    def _check_gesture_stability(self, track_id: int) -> bool:
        state = self.gesture_states[track_id]

        if not state.is_active:
            return False

        if state.get_duration() < self.min_duration:
            return False

        if state.stable_frames < self.stability_frames:
            return False

        if state.confidence < self.min_confidence:
            return False

        return True

    def _detect_swipe_direction(
        self, movement_vector: Tuple[float, float]
    ) -> Optional[SwipeDirection]:
        dx, dy = movement_vector

        magnitude = (dx**2 + dy**2) ** 0.5

        if magnitude < self.swipe_threshold:
            return None

        if abs(dx) > abs(dy):
            return SwipeDirection.RIGHT if dx > 0 else SwipeDirection.LEFT
        else:
            return SwipeDirection.DOWN if dy > 0 else SwipeDirection.UP

    def _detect_zoom_direction(
        self, initial_distance: float, current_distance: float
    ) -> Optional[ZoomDirection]:
        if initial_distance == 0:
            return None

        ratio = current_distance / initial_distance

        if abs(1.0 - ratio) < self.zoom_threshold:
            return None

        return ZoomDirection.IN if ratio > 1.0 else ZoomDirection.OUT

    def _detect_rotation_direction(
        self, initial_angle: float, current_angle: float
    ) -> Optional[RotationDirection]:
        angle_diff = (current_angle - initial_angle) % 360

        if angle_diff > 180:
            angle_diff -= 360

        if abs(angle_diff) < self.rotation_threshold:
            return None

        return (
            RotationDirection.CLOCKWISE
            if angle_diff > 0
            else RotationDirection.COUNTER_CLOCKWISE
        )

    def _trigger_gesture_callbacks(
        self, track_id: int, gesture_type: GestureType, gesture_data: dict = None
    ):
        gesture_name = gesture_type.name.lower()

        for callback in self.gesture_callbacks:
            try:
                callback(gesture_type.value, gesture_name, track_id, gesture_data)
            except Exception as e:
                print(f"Error in gesture callback: {e}")

    def update_state(self, track_id: int, gesture_data: dict):
        class_id = gesture_data.get("class_id", -1)
        confidence = gesture_data.get("confidence", 0.0)
        position = gesture_data.get("position", (0, 0))

        self._update_gesture_state(track_id, class_id, confidence, position)

        if not self._check_gesture_stability(track_id):
            return

        state = self.gesture_states[track_id]
        gesture_type = state.gesture_type
        movement_vector = state.get_movement_vector()

        if gesture_type == GestureType.SWIPE:
            swipe_direction = self._detect_swipe_direction(movement_vector)
            if swipe_direction:
                gesture_name = f"swipe_{swipe_direction.name.lower()}"
                self._trigger_gesture_callbacks(
                    track_id, gesture_type, {"direction": swipe_direction.name.lower()}
                )
                state.reset()

        elif gesture_type == GestureType.ZOOM:
            zoom_direction = (
                ZoomDirection.IN if movement_vector[1] < 0 else ZoomDirection.OUT
            )
            gesture_name = f"zoom_{zoom_direction.name.lower()}"
            self._trigger_gesture_callbacks(
                track_id, gesture_type, {"direction": zoom_direction.name.lower()}
            )
            state.reset()

        elif gesture_type == GestureType.ROTATION:
            rotation_direction = (
                RotationDirection.CLOCKWISE
                if movement_vector[0] > 0
                else RotationDirection.COUNTER_CLOCKWISE
            )
            gesture_name = f"rotation_{rotation_direction.name.lower()}"
            self._trigger_gesture_callbacks(
                track_id, gesture_type, {"direction": rotation_direction.name.lower()}
            )
            state.reset()

        else:
            self._trigger_gesture_callbacks(track_id, gesture_type)
            state.reset()

    def reset_state(self, track_id: int = None):
        if track_id is not None:
            self.gesture_states[track_id].reset()
        else:
            for state in self.gesture_states.values():
                state.reset()

    def get_active_gestures(self) -> Dict[int, GestureType]:
        active_gestures = {}
        for track_id, state in self.gesture_states.items():
            if state.is_active and self._check_gesture_stability(track_id):
                active_gestures[track_id] = state.gesture_type
        return active_gestures
