import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import pyautogui
import tkinter as tk
from tkinter import ttk
import threading

from utils.config import Config


class OverlayWindow:
    """Overlay window for drawing annotations on top of other applications"""

    def __init__(self):
        self.root = None
        self.canvas = None
        self.annotations = []
        self.current_annotation = None
        self.is_drawing = False
        self.drawing_positions = []

        self._create_overlay()

    def _create_overlay(self):
        """Create transparent overlay window"""
        try:
            self.root = tk.Tk()
            self.root.title("Gesture Annotation Overlay")
            self.root.geometry(
                f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0"
            )
            self.root.overrideredirect(True)
            self.root.attributes("-transparentcolor", "white")
            self.root.attributes("-topmost", True)
            self.root.attributes("-alpha", 0.8)

            self.canvas = tk.Canvas(self.root, bg="white", highlightthickness=0)
            self.canvas.pack(fill=tk.BOTH, expand=True)

            self.root.withdraw()

        except Exception as e:
            print(f"Error creating overlay window: {e}")
            self.root = None
            self.canvas = None

    def show(self):
        if self.root:
            self.root.deiconify()
            self.root.lift()

    def hide(self):
        if self.root:
            self.root.withdraw()

    def start_drawing(self, position: Optional[Tuple[int, int]] = None):
        if not self.root:
            return

        self.is_drawing = True
        self.current_annotation = {"points": [position] if position else []}

        if position:
            self.drawing_positions = [position]
        else:
            self.drawing_positions = []

    def update_drawing_position(self, position: Tuple[int, int]):
        if not self.is_drawing or not self.current_annotation or not self.canvas:
            return

        self.current_annotation["points"].append(position)
        self.drawing_positions.append(position)

        if len(self.current_annotation["points"]) > 1:
            prev_point = self.current_annotation["points"][-2]
            curr_point = self.current_annotation["points"][-1]
            self.canvas.create_line(
                prev_point[0],
                prev_point[1],
                curr_point[0],
                curr_point[1],
                fill="red",
                width=3,
                capstyle=tk.ROUND,
            )

    def end_drawing(self, position: Optional[Tuple[int, int]] = None):
        if not self.is_drawing:
            return

        if position and self.current_annotation:
            self.current_annotation["points"].append(position)

        if self.current_annotation and self.current_annotation["points"]:
            self.annotations.append(self.current_annotation)

        self.current_annotation = None
        self.is_drawing = False
        self.drawing_positions = []

    def clear_annotations(self):
        if self.canvas:
            self.canvas.delete("all")
        self.annotations.clear()
        self.current_annotation = None
        self.is_drawing = False
        self.drawing_positions = []

    def update(self):
        if self.root:
            try:
                self.root.update_idletasks()
                self.root.update()
            except:
                pass


class AnnotationController:
    """
    Screen annotation controller for drawing annotations on screen
    """

    def __init__(self, config: Config = None):
        self.config = config

        self.annotations: List[Dict] = []
        self.current_annotation = None
        self.is_drawing = False
        self.annotation_color = (255, 0, 0)  # Red
        self.annotation_thickness = 3
        self.annotation_type = "pen"
        self.drawing_positions = []

        self.overlay = OverlayWindow()
        self.overlay_active = False

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.1

    def start_drawing(self, position: Optional[Tuple[int, int]] = None):
        """
        Start drawing annotation

        Args:
            position: Starting position (x, y) for drawing
        """
        self.overlay_active = True
        self.overlay.show()
        self.overlay.start_drawing(position)

    def end_drawing(self, position: Optional[Tuple[int, int]] = None):
        """
        End drawing annotation

        Args:
            position: Ending position (x, y) for drawing
        """
        self.overlay.end_drawing(position)

    def update_drawing_position(self, position: Tuple[int, int]):
        """
        Update current drawing position

        Args:
            position: Current position (x, y) for drawing
        """
        if self.overlay_active:
            self.overlay.update_drawing_position(position)

    def update_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Update frame with annotations (for camera view)

        Args:
            frame: Input frame

        Returns:
            Annotated frame
        """
        return frame

    def clear_annotations(self):
        self.overlay.clear_annotations()

    def hide_overlay(self):
        self.overlay.hide()
        self.overlay_active = False

    def show_overlay(self):
        self.overlay.show()
        self.overlay_active = True
