#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Management - Loads and manages system configuration
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union


class Config:
    """Configuration management class for loading and managing system configuration"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager"""
        self.config_data: Dict[str, Any] = {}
        self.config_file = config_file
        self._load_default_config()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def _load_default_config(self):
        """Load default configuration"""
        self.config_data = {
            # Detector configuration
            "detector": {
                "model_path": "models/detector.onnx",
                "input_size": [640, 640],
                "confidence_threshold": 0.5,
                "nms_threshold": 0.45,
                "device": "cpu"
            },
            
            # Tracker configuration
            "tracker": {
                "max_disappeared": 30,
                "max_distance": 50,
                "min_confidence": 0.3
            },
            
            # Classifier configuration
            "classifier": {
                "model_path": "models/classifier.onnx",
                "input_size": [224, 224],
                "confidence_threshold": 0.7,
                "device": "cpu"
            },
            
            # Gesture controller configuration
            "gesture_controller": {
                "min_confidence": 0.7,
                "min_duration": 0.5,
                "stability_frames": 5,
                "swipe_threshold": 50,
                "zoom_threshold": 0.1,
                "rotation_threshold": 15
            },
            
            # Classroom interaction configuration
            "classroom": {
                "session_timeout": 3600,  # 1 hour
                "max_students": 100,
                "auto_save_interval": 300,  # 5 minutes
                "gesture_history_size": 1000
            },
            
            # Presentation controller configuration
            "presentation": {
                "slide_transition_duration": 0.5,
                "zoom_levels": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                "default_zoom": 1.0,
                "annotation_persistence": True
            },
            
            # Screen annotation configuration
            "annotation": {
                "default_color": "#FF0000",
                "default_width": 3,
                "available_colors": ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"],
                "available_widths": [1, 3, 5, 7, 10],
                "eraser_size": 20
            },
            
            # System configuration
            "system": {
                "log_level": "INFO",
                "log_file": "logs/system.log",
                "max_log_size": 10485760,  # 10MB
                "backup_count": 5,
                "debug_mode": False
            },
            
            # UI configuration
            "ui": {
                "window_width": 1280,
                "window_height": 720,
                "fps": 30,
                "show_fps": False,
                "show_confidence": True,
                "show_tracking_id": True
            }
        }
    
    def load_config(self, config_file: str):
        """Load configuration from file"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        file_ext = os.path.splitext(config_file)[1].lower()
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if file_ext in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif file_ext == '.json':
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {file_ext}")
            
            # Merge with default configuration
            self._merge_config(self.config_data, file_config)
            self.config_file = config_file
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    def save_config(self, config_file: Optional[str] = None):
        """Save configuration to file"""
        if config_file is None:
            config_file = self.config_file
        
        if config_file is None:
            raise ValueError("No configuration file specified")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        file_ext = os.path.splitext(config_file)[1].lower()
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                if file_ext in ['.yaml', '.yml']:
                    yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
                elif file_ext == '.json':
                    json.dump(self.config_data, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported configuration file format: {file_ext}")
            
            if config_file != self.config_file:
                self.config_file = config_file
                
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value by key"""
        keys = key.split('.')
        config = self.config_data
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def has(self, key: str) -> bool:
        """Check if configuration key exists"""
        return self.get(key) is not None
    
    def _merge_config(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Recursively merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get_detector_config(self) -> Dict[str, Any]:
        """Get detector configuration"""
        return self.get("detector", {})
    
    def get_tracker_config(self) -> Dict[str, Any]:
        """Get tracker configuration"""
        return self.get("tracker", {})
    
    def get_classifier_config(self) -> Dict[str, Any]:
        """Get classifier configuration"""
        return self.get("classifier", {})
    
    def get_gesture_controller_config(self) -> Dict[str, Any]:
        """Get gesture controller configuration"""
        return self.get("gesture_controller", {})
    
    def get_classroom_config(self) -> Dict[str, Any]:
        """Get classroom interaction configuration"""
        return self.get("classroom", {})
    
    def get_presentation_config(self) -> Dict[str, Any]:
        """Get presentation controller configuration"""
        return self.get("presentation", {})
    
    def get_annotation_config(self) -> Dict[str, Any]:
        """Get screen annotation configuration"""
        return self.get("annotation", {})
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration"""
        return self.get("system", {})
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration"""
        return self.get("ui", {})
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return json.dumps(self.config_data, indent=2, ensure_ascii=False)
    
    def __repr__(self) -> str:
        """Official string representation of configuration"""
        return f"Config(config_file={self.config_file})"