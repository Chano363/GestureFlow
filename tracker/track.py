import numpy as np
from typing import List, Tuple, Optional

from .kalman_filter import KalmanFilter


class Track:
    """
    跟踪对象定义
    """
    
    def __init__(self, track_id: int, bbox: List[int], max_age: int = 30, min_hits: int = 3):
        """
        初始化跟踪对象
        
        Args:
            track_id: 跟踪ID
            bbox: 边界框 [x1, y1, x2, y2]
            max_age: 最大未更新帧数
            min_hits: 最小命中次数
        """
        self.track_id = track_id
        self.bbox = bbox
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.age = 0  
        self.hit_streak = 0
        self.time_since_update = 0
        
        self.kf = KalmanFilter(bbox)
        
    def update(self, bbox: Optional[List[int]] = None) -> None:
        """
        更新跟踪对象
        
        Args:
            bbox: 新的边界框，如果为None则仅进行预测
        """
        self.age += 1
        
        if bbox is not None:
            self.kf.update(bbox)
            self.hit_streak += 1
            self.time_since_update = 0
            self.bbox = bbox
        else:
            self.time_since_update += 1
            self.hit_streak = 0        
    def predict(self) -> List[int]:
        """
        预测下一帧的位置
        
        Returns:
            预测的边界框 [x1, y1, x2, y2]
        """
        predicted_bbox = self.kf.predict()
        
        if len(predicted_bbox) >= 4:
            x1, y1, x2, y2 = predicted_bbox[:4]
            if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2) or
                    np.isinf(x1) or np.isinf(y1) or np.isinf(x2) or np.isinf(y2)):
                self.bbox = [int(x1), int(y1), int(x2), int(y2)]
        
        return self.bbox
    
    def get_state(self) -> Tuple[bool, List[int]]:
        """
        获取跟踪对象状态
        
        Returns:
            (是否有效, 边界框)
        """
        is_valid = self.hit_streak >= self.min_hits and self.time_since_update <= self.max_age
        return is_valid, self.bbox