import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pyautogui
import cv2
from PIL import Image, ImageDraw, ImageFont
import logging

from utils.config import Config


class PresentationController:
    """
    Controller for presentation slides
    """

    def __init__(self, config: Config = None):
        self.config = config

        self.current_slide_index = 0
        self.total_slides = 0

        self.slide_history: List[int] = []

        self.zoom_level = 1.0
        self.zoom_center = None

        self.annotations: List[Dict] = []
        self.current_annotation = None

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.1

        self.logger = logging.getLogger(__name__)

    def next_slide(self) -> bool:
        try:
            self.logger.info("Sending 'right' key for next slide")
            pyautogui.press("right")

            if self.current_slide_index < self.total_slides - 1:
                self.slide_history.append(self.current_slide_index)
                self.current_slide_index += 1
                return True

            return False
        except Exception as e:
            self.logger.error(f"Error in next_slide: {e}")
            return False

    def previous_slide(self) -> bool:
        try:
            self.logger.info("Sending 'left' key for previous slide")
            pyautogui.press("left")

            if self.current_slide_index > 0:
                self.slide_history.append(self.current_slide_index)
                self.current_slide_index -= 1
                return True

            return False
        except Exception as e:
            self.logger.error(f"Error in previous_slide: {e}")
            return False

    def go_to_slide(self, slide_index: int) -> bool:
        try:
            if 0 <= slide_index < self.total_slides:
                self.slide_history.append(self.current_slide_index)
                self.current_slide_index = slide_index
                return True

            return False
        except Exception as e:
            self.logger.error(f"Error in go_to_slide: {e}")
            return False

    def first_slide(self) -> bool:
        try:
            self.logger.info("Sending 'ctrl+home' keys for first slide")
            pyautogui.hotkey("ctrl", "home")

            if self.current_slide_index != 0:
                self.slide_history.append(self.current_slide_index)
                self.current_slide_index = 0
                return True

            return False
        except Exception as e:
            self.logger.error(f"Error in first_slide: {e}")
            return False

    def last_slide(self) -> bool:
        try:
            self.logger.info("Sending 'ctrl+end' keys for last slide")
            pyautogui.hotkey("ctrl", "end")

            if self.current_slide_index != self.total_slides - 1:
                self.slide_history.append(self.current_slide_index)
                self.current_slide_index = self.total_slides - 1
                return True

            return False
        except Exception as e:
            self.logger.error(f"Error in last_slide: {e}")
            return False

    def undo_navigation(self) -> bool:
        try:
            if len(self.slide_history) > 0:
                previous_slide = self.slide_history.pop()
                self.current_slide_index = previous_slide
                return True

            return False
        except Exception as e:
            self.logger.error(f"Error in undo_navigation: {e}")
            return False

    def start_slideshow(self) -> None:
        try:
            self.logger.info("Sending 'F5' key to start slideshow")
            pyautogui.press("f5")
            self.logger.info("Sent 'F5' key successfully")
        except Exception as e:
            self.logger.error(f"Error in start_slideshow: {e}")

    def end_slideshow(self) -> None:
        try:
            self.logger.info("Sending 'ESC' key to end slideshow")
            pyautogui.press("esc")
            self.logger.info("Sent 'ESC' key successfully")
        except Exception as e:
            self.logger.error(f"Error in end_slideshow: {e}")

    def toggle_slideshow(self) -> None:
        try:
            self.logger.info("Sending 'F5' key to toggle slideshow")
            pyautogui.press("f5")
            self.logger.info("Sent 'F5' key successfully")
        except Exception as e:
            self.logger.error(f"Error in toggle_slideshow: {e}")

    def pause_slideshow(self) -> None:
        try:
            self.logger.info("Sending '.' key to pause slideshow")
            pyautogui.press(".")
            self.logger.info("Sent '.' key successfully")
        except Exception as e:
            self.logger.error(f"Error in pause_slideshow: {e}")

    def resume_slideshow(self) -> None:
        try:
            self.logger.info("Sending '.' key to resume slideshow")
            pyautogui.press(".")
            self.logger.info("Sent '.' key successfully")
        except Exception as e:
            self.logger.error(f"Error in resume_slideshow: {e}")

    def blackout_screen(self) -> None:
        try:
            self.logger.info("Sending 'B' key for blackout")
            pyautogui.press("b")
            self.logger.info("Sent 'B' key successfully")
        except Exception as e:
            self.logger.error(f"Error in blackout_screen: {e}")

    def whiteout_screen(self) -> None:
        try:
            self.logger.info("Sending 'W' key for whiteout")
            pyautogui.press("w")
            self.logger.info("Sent 'W' key successfully")
        except Exception as e:
            self.logger.error(f"Error in whiteout_screen: {e}")

    def set_zoom(
        self, zoom_level: float, center: Optional[Tuple[int, int]] = None
    ) -> None:
        try:
            self.zoom_level = max(0.5, min(3.0, zoom_level))
            if center:
                self.zoom_center = center
        except Exception as e:
            self.logger.error(f"Error in set_zoom: {e}")

    def zoom_in(self, center: Optional[Tuple[int, int]] = None) -> None:
        try:
            self.logger.info("Zooming in")
            self.set_zoom(self.zoom_level * 1.2, center)
        except Exception as e:
            self.logger.error(f"Error in zoom_in: {e}")

    def zoom_out(self, center: Optional[Tuple[int, int]] = None) -> None:
        try:
            self.logger.info("Zooming out")
            self.set_zoom(self.zoom_level / 1.2, center)
        except Exception as e:
            self.logger.error(f"Error in zoom_out: {e}")

    def reset_zoom(self) -> None:
        try:
            self.zoom_level = 1.0
            self.zoom_center = None
        except Exception as e:
            self.logger.error(f"Error in reset_zoom: {e}")

    def get_current_slide_index(self) -> int:
        return self.current_slide_index

    def get_total_slides(self) -> int:
        return self.total_slides

    def set_total_slides(self, total_slides: int) -> None:
        self.total_slides = max(0, total_slides)

    def get_slide_progress(self) -> Tuple[int, int]:
        return self.current_slide_index, self.total_slides

    def get_zoom_level(self) -> float:
        return self.zoom_level
