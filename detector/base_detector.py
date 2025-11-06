from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseDetector(ABC):
    """
    检测器基类，定义检测器的通用接口
    """

    def __init__(self):
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        检测图像中的对象

        Args:
            image: 输入图像

        Returns:
            检测结果列表，每个元素为(x1, y1, x2, y2, confidence)
        """
        pass

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理

        Args:
            image: 输入图像

        Returns:
            预处理后的图像
        """
        return image

    def postprocess(self, detections: List) -> List:
        """
        检测结果后处理

        Args:
            detections: 原始检测结果

        Returns:
            处理后的检测结果
        """
        return detections
