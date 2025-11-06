import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple

from .base_classifier import BaseClassifier


class GestureClassifier(BaseClassifier):
    """
    手势分类器实现
    """
    
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (224, 224)):
        """
        初始化手势分类器
        
        Args:
            model_path: ONNX模型路径
            input_size: 输入图像尺寸 (width, height)
        """
        super().__init__()
        self.model_path = model_path
        self.input_size = input_size
        
        # 加载ONNX模型
        self.session = ort.InferenceSession(model_path)
        
        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 类别名称 - 根据实际情况修改
        self.class_names = [
            "none",      # 无手势
            "point",     # 指向
            "click",     # 点击
            "swipe",     # 滑动
            "zoom",      # 缩放
            "rotate",    # 旋转
            "grab",      # 抓取
            "release"    # 释放
        ]
        
    def classify(self, image: np.ndarray) -> Tuple[int, float]:
        """
        分类图像中的手势
        
        Args:
            image: 输入图像
            
        Returns:
            (类别ID, 置信度)
        """
        # 预处理图像
        processed_image = self.preprocess(image)
        
        # 模型推理
        inputs = {self.input_name: processed_image}
        outputs = self.session.run([self.output_name], inputs)
        predictions = outputs[0][0]
        
        # 获取最高置信度的类别
        class_id = np.argmax(predictions)
        confidence = float(predictions[class_id])
        
        return class_id, confidence
    
    def classify_batch(self, images: List[np.ndarray]) -> List[Tuple[int, float]]:
        """
        批量分类图像中的手势
        
        Args:
            images: 输入图像列表
            
        Returns:
            分类结果列表，每个元素为(类别ID, 置信度)
        """
        results = []
        
        # 预处理所有图像
        processed_images = np.stack([self.preprocess(img) for img in images])
        
        # 模型推理
        inputs = {self.input_name: processed_images}
        outputs = self.session.run([self.output_name], inputs)
        predictions = outputs[0]
        
        # 处理每个预测结果
        for pred in predictions:
            class_id = np.argmax(pred)
            confidence = float(pred[class_id])
            results.append((class_id, confidence))
        
        return results
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 调整图像大小
        resized_image = cv2.resize(image, self.input_size)
        
        # 转换为RGB
        if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # 归一化到[0, 1]
        normalized_image = resized_image.astype(np.float32) / 255.0
        
        # 标准化 (使用ImageNet的均值和标准差)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized_image = (normalized_image - mean) / std
        
        # 转换为CHW格式并添加batch维度
        processed_image = np.transpose(normalized_image, (2, 0, 1))
        processed_image = np.expand_dims(processed_image, axis=0)
        
        return processed_image
    
    def get_class_names(self) -> List[str]:
        """
        获取类别名称列表
        
        Returns:
            类别名称列表
        """
        return self.class_names
    
    def get_class_name(self, class_id: int) -> str:
        """
        根据类别ID获取类别名称
        
        Args:
            class_id: 类别ID
            
        Returns:
            类别名称
        """
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return "unknown"