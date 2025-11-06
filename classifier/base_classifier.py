from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import numpy as np


class BaseClassifier(ABC):
    """
    分类器基类，定义分类器的通用接口
    """
    
    def __init__(self):
        """初始化分类器"""
        pass
    
    @abstractmethod
    def classify(self, data: Any) -> Tuple[int, float]:
        """
        分类输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            (类别ID, 置信度)
        """
        pass
    
    @abstractmethod
    def get_class_names(self) -> List[str]:
        """
        获取类别名称列表
        
        Returns:
            类别名称列表
        """
        pass
    
    def preprocess(self, data: Any) -> Any:
        """
        数据预处理
        
        Args:
            data: 输入数据
            
        Returns:
            预处理后的数据
        """
        return data
    
    def postprocess(self, predictions: Any) -> Tuple[int, float]:
        """
        预测结果后处理
        
        Args:
            predictions: 原始预测结果
            
        Returns:
            (类别ID, 置信度)
        """
        return predictions