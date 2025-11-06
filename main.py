#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

# 解决环境冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import logging
import time
from pathlib import Path
import pyautogui

import cv2
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import Config
from controller.main_controller import MainController
from controller.gesture_controller import GestureController
from classroom.presentation import PresentationController
from classroom.annotation import AnnotationController
from classroom.interaction import InteractionController


def setup_logging(log_level, log_file=None):
    """Set up logging"""
    log_dir = os.path.dirname(log_file) if log_file else "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()), format=log_format, handlers=handlers
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Dynamic Gesture Recognition System")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Configuration file path"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--video", type=str, help="Video file path")
    parser.add_argument("--output", type=str, help="Output video file path")
    parser.add_argument(
        "--display", action="store_true", default=True, help="Display results"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    parser.add_argument("--log-file", type=str, help="Log file path")

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

    config = Config(args.config)
    logger.info(f"Configuration file loaded: {args.config}")

    main_controller = MainController(config)
    gesture_controller = GestureController(config)
    presentation_controller = PresentationController(config)
    annotation_controller = AnnotationController(config)
    interaction_controller = InteractionController(config)

    interaction_state = {
        "drawing": False,
        "dragging": False,
        "track_id": None,
        "last_gesture_time": 0,
        "last_position": None,
    }

    GESTURE_COOLDOWN = 0.3

    def on_gesture(gesture_id, gesture_name, track_id, gesture_data=None):
        """Gesture callback function"""
        logger.info(
            f"Gesture detected: {gesture_name} (ID: {gesture_id}, Track ID: {track_id})"
        )

        confidence = gesture_data.get("confidence", 0) if gesture_data else 0
        logger.info(f"Gesture confidence: {confidence:.2f}")

        current_time = time.time()

        position = None
        if gesture_data and "position" in gesture_data:
            position = gesture_data["position"]

        if gesture_name in ["two_up", "peace"]:  # V手势或和平手势，都执行下一张幻灯片
            logger.info(f"Executing next_slide command for {gesture_name}")
            presentation_controller.next_slide()
        elif gesture_name == "two_up_inverted":  # 倒V手势，上一张幻灯片
            logger.info("Executing previous_slide command")
            presentation_controller.previous_slide()
        elif gesture_name == "ok":  # OK手势，开始播放
            logger.info("Executing start_slideshow command")
            presentation_controller.start_slideshow()
        elif gesture_name == "stop":  # 停止手势，结束播放
            logger.info("Executing end_slideshow command")
            presentation_controller.end_slideshow()
        elif gesture_name == "grabbing":  # 抓取手势，放大
            logger.info("Executing zoom_in command")
            presentation_controller.zoom_in()
        elif gesture_name == "fist":  # 拳头手势，缩小
            logger.info("Executing zoom_out command")
            presentation_controller.zoom_out()

        elif gesture_name == "one" or gesture_name == "point":  # 使用'one'手势开始绘制
            if not interaction_state['drawing'] and not interaction_state['dragging']:
                annotation_controller.start_drawing(position)
                interaction_state['drawing'] = True
                interaction_state['track_id'] = track_id
                interaction_state['last_gesture_time'] = current_time
                interaction_state['last_position'] = position
                logger.info(f"Started drawing with track ID: {track_id} at position: {position}")
                
        elif gesture_name == "palm" or gesture_name == "stop":
            if interaction_state['drawing']:
                annotation_controller.end_drawing(position)
                interaction_state['drawing'] = False
                interaction_state['track_id'] = None
                interaction_state['last_gesture_time'] = current_time
                interaction_state['last_position'] = None
                logger.info("Ended drawing")

    gesture_controller.register_gesture_callback(on_gesture)

    if args.video:
        cap = cv2.VideoCapture(args.video)
        logger.info(f"Video file opened: {args.video}")
    else:
        camera_id = (
            args.camera
            if args.camera is not None
            else config.get("system.camera_id", 0)
        )
        cap = cv2.VideoCapture(camera_id)
        logger.info(f"Camera opened: {camera_id}")

    if not args.video:
        width = config.get("system.camera_width", 1280)
        height = config.get("system.camera_height", 720)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    writer = None
    if args.output:
        fps = config.get("main_controller.fps", 30)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
        logger.info(f"Video writer initialized: {args.output}")

    interaction_controller.start_session()

    try:
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    logger.info("Video processing completed")
                else:
                    logger.warning("Unable to read camera frame")
                break

            processed_frame, gesture_results = main_controller.process_frame(frame)

            for gesture_result in gesture_results:
                track_id = gesture_result["id"]
                bbox = gesture_result["bbox"]
                position = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

                gesture_data = gesture_result.copy()
                gesture_data["position"] = position

                gesture_name = gesture_result.get("class_name", "unknown")
                gesture_id = gesture_result.get("class_id", -1)

                on_gesture(gesture_id, gesture_name, track_id, gesture_data)

                gesture_controller.update_state(track_id, gesture_result)

                if (
                    interaction_state["drawing"]
                    and interaction_state["track_id"] == track_id
                ):
                    if (
                        interaction_state["last_position"] is None
                        or abs(position[0] - interaction_state["last_position"][0]) > 1
                        or abs(position[1] - interaction_state["last_position"][1]) > 1
                    ):
                        screen_position = _convert_camera_to_screen_coords(
                            position, frame.shape[1], frame.shape[0]
                        )
                        annotation_controller.update_drawing_position(screen_position)
                        interaction_state["last_position"] = position
                        logger.debug(
                            f"Updated drawing position: {position} -> {screen_position}"
                        )

            annotated_frame = annotation_controller.update_frame(processed_frame)
            if frame_count % 2 == 0:
                annotation_controller.overlay.update()
            if writer:
                writer.write(annotated_frame)

            if args.display:
                cv2.imshow("Dynamic Gesture Recognition", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):  # Space key to pause
                    cv2.waitKey(0)
                elif key == ord("s"):  # 's' key to take screenshot
                    screenshot_dir = "screenshots"
                    os.makedirs(screenshot_dir, exist_ok=True)
                    timestamp = int(time.time())
                    screenshot_path = f"{screenshot_dir}/screenshot_{timestamp}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    logger.info(f"Screenshot saved: {screenshot_path}")
                elif key == ord("c"):  # 'c' key to clear annotations
                    annotation_controller.clear_annotations()
                    interaction_state["drawing"] = False
                    interaction_state["dragging"] = False
                    interaction_state["track_id"] = None
                    interaction_state["last_position"] = None
                    logger.info("Annotations cleared")
                elif key == ord("d"):  # 'd' key to toggle drawing mode
                    if not interaction_state['drawing'] and not interaction_state['dragging']:
                        screen_width, screen_height = pyautogui.size()
                        center_position = (screen_width // 2, screen_height // 2)
                        annotation_controller.start_drawing(center_position)
                        interaction_state['drawing'] = True
                        interaction_state['last_position'] = center_position
                        logger.info("Manual drawing mode enabled")
                    else:
                        annotation_controller.end_drawing()
                        interaction_state['drawing'] = False
                        interaction_state['dragging'] = False
                        interaction_state['track_id'] = None
                        interaction_state['last_position'] = None
                        logger.info("Manual drawing mode disabled")
                elif key == ord("p"):  # 'p' key to toggle presentation mode
                    presentation_controller.toggle_slideshow()
                    logger.info("Toggled slideshow mode")

            frame_count += 1

            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                logger.info(f"Processed frames: {frame_count}, FPS: {fps:.2f}")

    except KeyboardInterrupt:
        logger.info("Program interrupted by user")

    except Exception as e:
        logger.error(f"Program error: {str(e)}", exc_info=True)

    finally:
        cap.release()
        if writer:
            writer.release()

        try:
            annotation_controller.hide_overlay()
        except:
            pass

        try:
            cv2.destroyAllWindows()
        except:
            pass

        try:
            interaction_controller.end_session()
        except:
            pass


def _convert_camera_to_screen_coords(
    position: tuple[int, int], frame_width: int, frame_height: int
) -> tuple[int, int]:
    """
    Convert camera coordinates to screen coordinates

    Args:
        position: Camera coordinates (x, y)
        frame_width: Camera frame width
        frame_height: Camera frame height

    Returns:
        Screen coordinates (x, y)
    """
    try:
        screen_width, screen_height = pyautogui.size()
    except Exception as e:
        screen_width, screen_height = 1920, 1080

    mirrored_x = frame_width - position[0]
    screen_x = int(mirrored_x * screen_width / frame_width)
    screen_y = int(position[1] * screen_height / frame_height)

    return (screen_x, screen_y)

if __name__ == "__main__":
    main()
