#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enumerations - Defines all enumeration types used in the system
"""

from enum import Enum


class GestureType(Enum):
    """Gesture type enumeration"""
    
    # Basic gestures
    NONE = 0
    POINT = 1
    CLICK = 2
    DOUBLE_CLICK = 3
    LONG_PRESS = 4
    
    # Swipe gestures
    SWIPE_LEFT = 5
    SWIPE_RIGHT = 6
    SWIPE_UP = 7
    SWIPE_DOWN = 8
    
    # Zoom gestures
    ZOOM_IN = 9
    ZOOM_OUT = 10
    
    # Rotation gestures
    ROTATE_LEFT = 11
    ROTATE_RIGHT = 12
    
    # Hand gestures
    FIST = 13
    PALM = 14
    THUMB_UP = 15
    THUMB_DOWN = 16
    OK = 17
    PEACE = 18
    ROCK = 19
    PAPER = 20
    SCISSORS = 21
    
    # HaGRID gestures
    CALL = 22
    DISLIKE = 23
    LIKE = 24
    MUTED = 25
    FOUR_FINGERS = 26
    STOP = 27
    STOP_INVERTED = 28
    TWO_FINGERS_UP = 29
    
    # Complex gestures
    SWIPE = 30
    ROTATION = 31
    GRAB = 32
    RELEASE = 33
    
    # Special gestures
    UNKNOWN = 99


class SwipeDirection(Enum):
    """Swipe direction enumeration"""
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class ZoomDirection(Enum):
    """Zoom direction enumeration"""
    IN = 0
    OUT = 1


class RotationDirection(Enum):
    """Rotation direction enumeration"""
    CLOCKWISE = 0
    COUNTER_CLOCKWISE = 1


class AnnotationType(Enum):
    """Annotation type enumeration"""
    NONE = 0
    PEN = 1
    HIGHLIGHTER = 2
    ERASER = 3
    TEXT = 4
    SHAPE = 5


class InteractionType(Enum):
    """Interaction type enumeration"""
    NONE = 0
    QUESTION = 1
    ANSWER = 2
    POLL = 3
    QUIZ = 4
    FEEDBACK = 5


class SystemAction(Enum):
    """System action enumeration"""
    NONE = 0
    START = 1
    STOP = 2
    PAUSE = 3
    RESUME = 4
    RESET = 5
    SAVE = 6
    LOAD = 7
    EXPORT = 8
    IMPORT = 9
    SETTINGS = 10


class PresentationAction(Enum):
    """Presentation action enumeration"""
    NONE = 0
    NEXT_SLIDE = 1
    PREV_SLIDE = 2
    FIRST_SLIDE = 3
    LAST_SLIDE = 4
    GOTO_SLIDE = 5
    ZOOM_IN = 6
    ZOOM_OUT = 7
    ZOOM_RESET = 8
    START_PRESENTATION = 9
    END_PRESENTATION = 10
    PAUSE_PRESENTATION = 11
    RESUME_PRESENTATION = 12


class AnnotationAction(Enum):
    """Annotation action enumeration"""
    NONE = 0
    START_DRAWING = 1
    STOP_DRAWING = 2
    CHANGE_COLOR = 3
    CHANGE_WIDTH = 4
    CHANGE_TYPE = 5
    CLEAR_ALL = 6
    UNDO = 7
    REDO = 8
    SAVE = 9
    LOAD = 10


class TrackingState(Enum):
    """Tracking state enumeration"""
    LOST = 0
    INITIALIZING = 1
    TRACKING = 2
    OCCLUDED = 3


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class DeviceType(Enum):
    """Device type enumeration"""
    CPU = 0
    GPU = 1
    TPU = 2
    AUTO = 3


class ModelType(Enum):
    """Model type enumeration"""
    DETECTOR = 0
    CLASSIFIER = 1
    TRACKER = 2
    RECOGNIZER = 3


class InputSource(Enum):
    """Input source enumeration"""
    CAMERA = 0
    VIDEO_FILE = 1
    IMAGE_SEQUENCE = 2
    STREAM = 3


class OutputFormat(Enum):
    """Output format enumeration"""
    WINDOW = 0
    VIDEO_FILE = 1
    IMAGE_SEQUENCE = 2
    STREAM = 3


class UIElement(Enum):
    """UI element enumeration"""
    NONE = 0
    MAIN_WINDOW = 1
    CONTROL_PANEL = 2
    STATUS_BAR = 3
    MENU_BAR = 4
    TOOLBAR = 5
    SIDEBAR = 6
    DIALOG = 7
    POPUP = 8


class EventType(Enum):
    """Event type enumeration"""
    NONE = 0
    GESTURE_DETECTED = 1
    GESTURE_STARTED = 2
    GESTURE_ENDED = 3
    TRACKING_STARTED = 4
    TRACKING_LOST = 5
    SYSTEM_ERROR = 6
    USER_ACTION = 7
    TIMER = 8
    CUSTOM = 9