import numpy as np
from typing import List, Tuple


class KalmanFilter:
    
    def __init__(self, bbox: List[int]):
        """
        初始化卡尔曼滤波器
        
        Args:
            bbox: 初始边界框 [x1, y1, x2, y2]
        """
        self.x = np.zeros((8, 1))
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        self.x[0, 0] = x1 + w / 2.0  # x
        self.x[1, 0] = y1 + h / 2.0  # y
        self.x[2, 0] = w * h         # s
        self.x[3, 0] = w / float(h)  # r
        
        self.R = np.eye(4) * 1e-1
        
        self.Q = np.eye(8) * 1e-2
        
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = 1.0
            
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y
        self.H[2, 2] = 1.0  # s
        self.H[3, 3] = 1.0  # r
        
        self.last_bbox = list(bbox)

    def predict(self) -> List[int]:
        """
        预测下一帧的状态
        
        Returns:
            预测的边界框 [x1, y1, x2, y2]
        """
        self.x = np.dot(self.F, self.x)
        
        x, y, s, r = self.x[:4, 0]
        w = np.sqrt(s * r)
        h = s / w
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2) or \
           np.isinf(x1) or np.isinf(y1) or np.isinf(x2) or np.isinf(y2):
            return self.last_bbox.copy()
        
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = max(x1 + 1, int(x2))
        y2 = max(y1 + 1, int(y2))
        
        result = [x1, y1, x2, y2]
        self.last_bbox = result.copy()
        return result
    
    def update(self, bbox: List[int]) -> None:
        """
        更新卡尔曼滤波器
        
        Args:
            bbox: 观测到的边界框 [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        s = w * h
        r = w / float(h)
        
        self.x[0, 0] = x
        self.x[1, 0] = y
        self.x[2, 0] = s
        self.x[3, 0] = r
        
        self.last_bbox = list(bbox)