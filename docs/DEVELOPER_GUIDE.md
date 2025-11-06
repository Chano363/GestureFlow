# 开发者指南

本文档为开发者提供动态手势识别系统的详细开发指南，包括系统架构、开发环境设置、代码规范、扩展指南和测试方法。

## 目录

- [系统架构](#系统架构)
- [开发环境设置](#开发环境设置)
- [代码规范](#代码规范)
- [扩展指南](#扩展指南)
- [测试指南](#测试指南)
- [常见问题](#常见问题)

## 系统架构

### 整体架构

系统采用分层架构设计，主要包含以下层次：

1. **数据层**：负责数据的采集、预处理和存储
2. **检测层**：负责手部检测，基于YOLOv10模型
3. **跟踪层**：负责多目标跟踪，基于SORT算法
4. **分类层**：负责手势分类，基于深度学习模型
5. **控制层**：负责手势状态管理和事件处理
6. **应用层**：负责具体应用场景的实现，如PPT控制、标注等

### 核心组件

- **检测器 (Detector)**：负责从图像中检测手部区域
- **跟踪器 (Tracker)**：负责跟踪多个手部目标
- **分类器 (Classifier)**：负责识别手部手势
- **控制器 (Controller)**：负责协调各组件并处理手势事件
- **应用模块 (Application)**：负责实现具体的应用功能

### 数据流

1. 图像采集 → 手部检测 → 目标跟踪 → 手势分类 → 手势事件处理 → 应用响应

## 开发环境设置

### 环境要求

- Python 3.9+
- PyTorch
- OpenCV
- NumPy
- ONNX Runtime
- PyYAML
- PyAutoGUI
- filterpy
- ultralytics
- CUDA (可选，用于GPU加速)

### 安装步骤

1. 克隆项目仓库
```bash
git clone https://github.com/your-repo/gesture-recognition.git
cd gesture-recognition
```

2. 使用environment.yml创建并激活conda环境
```bash
# 使用environment.yml创建环境
conda env create -f environment.yml

# 激活环境
conda activate gesture_recognition

# 如果需要GPU版本，可以修改environment.yml文件，添加GPU相关依赖
# 例如：将cpuonly替换为pytorch-cuda=11.8（根据您的CUDA版本调整）
```

4. 下载预训练模型
```bash
# 下载检测模型
wget https://example.com/models/YOLOv10x_gestures.pt -O models/YOLOv10x_gestures.pt

# 下载分类模型
wget https://example.com/models/gesture_classifier.onnx -O models/gesture_classifier.onnx
```

### IDE配置

推荐使用PyCharm或VSCode进行开发，配置如下：

1. 安装Python插件
2. 设置项目解释器为创建的虚拟环境
3. 配置代码格式化工具（如black、isort）
4. 配置代码检查工具（如pylint、flake8）

## 代码规范

### 命名规范

- **类名**：使用PascalCase，如`GestureClassifier`
- **函数名**：使用snake_case，如`process_frame`
- **变量名**：使用snake_case，如`gesture_result`
- **常量名**：使用UPPER_SNAKE_CASE，如`MAX_CONFIDENCE_THRESHOLD`
- **私有成员**：使用单下划线前缀，如`_internal_method`

### 文档规范

- 所有公共类、方法和函数都应有docstring
- 使用Google风格的docstring格式
- 类docstring应简要描述类的功能和用途
- 方法docstring应包含参数、返回值和异常说明

示例：
```python
def process_frame(self, frame: np.ndarray) -> ProcessResult:
    """处理单帧图像
    
    Args:
        frame: 输入帧，形状为 (H, W, C) 的numpy数组
        
    Returns:
        处理结果，包含检测、跟踪和分类结果
        
    Raises:
        ProcessingError: 处理过程中出现错误
    """
    # 实现代码
```

### 类型注解

- 所有公共接口都应使用类型注解
- 使用typing模块中的类型，如List、Dict、Optional等
- 复杂类型可以使用TypeVar或Protocol

示例：
```python
from typing import List, Dict, Optional, Tuple, Union

def detect_objects(
    image: np.ndarray, 
    confidence_threshold: float = 0.5
) -> List[Detection]:
    """检测图像中的对象"""
    # 实现代码
```

### 错误处理

- 使用自定义异常类，不要直接抛出Exception
- 异常消息应清晰描述问题原因
- 捕获异常时应记录日志并采取适当的恢复措施

示例：
```python
try:
    result = self.detector.detect(image)
except ModelLoadError as e:
    logger.error(f"Failed to load model: {e}")
    raise
except DetectionError as e:
    logger.warning(f"Detection failed: {e}")
    return []
```

### 日志规范

- 使用logging模块，不要直接使用print
- 根据重要性使用适当的日志级别（DEBUG、INFO、WARNING、ERROR）
- 日志消息应简洁明了，包含足够的上下文信息

示例：
```python
import logging

logger = logging.getLogger(__name__)

def process_frame(self, frame: np.ndarray) -> ProcessResult:
    logger.debug(f"Processing frame with shape: {frame.shape}")
    
    try:
        # 处理代码
        logger.info("Frame processed successfully")
        return result
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        raise ProcessingError(f"Failed to process frame: {e}")
```

## 扩展指南

### 添加新的手势类型

1. 在`utils/enums.py`中的`GestureType`枚举中添加新手势
2. 在`data/gestures.yaml`中添加新手势的定义和配置
3. 如果需要新的手势分类，训练并添加新的分类模型
4. 更新相关的控制器和应用逻辑

示例：
```python
# 在 utils/enums.py 中添加
class GestureType(Enum):
    # 现有手势...
    NEW_GESTURE = "new_gesture"  # 添加新手势
```

```yaml
# 在 data/gestures.yaml 中添加
gestures:
  # 现有手势...
  new_gesture:
    id: 20
    name: "new_gesture"
    description: "新手势描述"
    # 其他配置...
```

### 添加新的检测器

1. 继承`BaseDetector`类
2. 实现`detect`方法
3. 在工厂类中注册新的检测器类型
4. 更新配置文件以支持新的检测器

示例：
```python
# 在 detector/custom_detector.py 中
class CustomDetector(BaseDetector):
    """自定义检测器"""
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__()
        # 初始化代码
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        # 实现检测逻辑
        return detections
```

### 添加新的跟踪器

1. 继承`BaseTracker`类
2. 实现`update`方法
3. 在工厂类中注册新的跟踪器类型
4. 更新配置文件以支持新的跟踪器

示例：
```python
# 在 tracker/custom_tracker.py 中
class CustomTracker(BaseTracker):
    """自定义跟踪器"""
    
    def __init__(self, **kwargs):
        super().__init__()
        # 初始化代码
    
    def update(self, detections: List[Detection]) -> List[Track]:
        # 实现跟踪逻辑
        return tracks
```

### 添加新的应用场景

1. 创建新的控制器类，继承`BaseController`
2. 实现必要的方法和回调函数
3. 在主控制器中注册新的应用控制器
4. 更新配置文件以支持新的应用场景

示例：
```python
# 在 application/custom_app.py 中
class CustomAppController(BaseController):
    """自定义应用控制器"""
    
    def __init__(self, config: Config):
        super().__init__(config)
        # 初始化代码
    
    def handle_gesture(self, gesture_type: GestureType, track_id: int) -> None:
        # 处理手势事件
        pass
```

### 添加新的配置选项

1. 在配置文件中添加新的配置项
2. 在`Config`类中添加获取和设置方法
3. 在相关组件中使用新的配置项
4. 更新配置文档

示例：
```yaml
# 在 config.yaml 中添加
custom_feature:
  enabled: true
  parameter1: 100
  parameter2: "value"
```

```python
# 在 utils/config.py 中
class Config:
    def get_custom_parameter1(self) -> int:
        return self.get("custom_feature.parameter1", 100)
```

## 测试指南

### 单元测试

- 使用pytest框架进行单元测试
- 测试文件命名规范：`test_*.py`
- 测试类命名规范：`Test*`
- 测试方法命名规范：`test_*`

示例：
```python
# 在 tests/test_detector.py 中
import pytest
import numpy as np
from detector.yolo_detector import YOLODetector

class TestYOLODetector:
    def setup_method(self):
        """测试前的设置"""
        self.model_path = "models/YOLOv10x_gestures.pt"
        self.detector = YOLODetector(self.model_path)
    
    def test_detect_with_valid_image(self):
        """测试有效图像的检测"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = self.detector.detect(image)
        assert isinstance(detections, list)
    
    def test_detect_with_invalid_image(self):
        """测试无效图像的检测"""
        with pytest.raises(InvalidInputError):
            self.detector.detect(None)
```

### 集成测试

- 测试多个组件之间的交互
- 使用模拟对象（mock）隔离外部依赖
- 测试完整的工作流程

示例：
```python
# 在 tests/test_integration.py 中
import pytest
from unittest.mock import Mock
from controller.main_controller import MainController
from utils.config import Config

class TestMainControllerIntegration:
    def setup_method(self):
        """测试前的设置"""
        self.config = Config("config/test_config.yaml")
        self.controller = MainController(self.config)
    
    def test_full_processing_pipeline(self):
        """测试完整的处理流程"""
        # 创建测试图像
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 处理图像
        result = self.controller.process_frame(image)
        
        # 验证结果
        assert result.processed_image is not None
        assert isinstance(result.gesture_events, list)
```

### 性能测试

- 使用基准测试评估系统性能
- 测试不同输入大小和复杂度下的性能
- 监控内存使用和CPU/GPU利用率

示例：
```python
# 在 tests/test_performance.py 中
import time
import pytest
import numpy as np
from controller.main_controller import MainController
from utils.config import Config

class TestPerformance:
    def setup_method(self):
        """测试前的设置"""
        self.config = Config("config/test_config.yaml")
        self.controller = MainController(self.config)
    
    def test_processing_speed(self):
        """测试处理速度"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        start_time = time.time()
        for _ in range(100):
            self.controller.process_frame(image)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.1  # 平均每帧处理时间应小于100ms
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_detector.py

# 运行特定测试方法
pytest tests/test_detector.py::TestYOLODetector::test_detect_with_valid_image

# 生成覆盖率报告
pytest --cov=src tests/

# 生成HTML覆盖率报告
pytest --cov=src --cov-report=html tests/
```

## 常见问题

### 模型加载失败

**问题**：启动时出现模型加载错误
**解决方案**：
1. 检查模型文件是否存在
2. 验证模型文件路径是否正确
3. 确认模型文件格式是否受支持
4. 检查设备配置是否正确（CPU/GPU）

### 检测精度低

**问题**：手部检测精度不理想
**解决方案**：
1. 调整置信度阈值
2. 检查输入图像质量
3. 考虑使用更高分辨率的输入
4. 尝试不同的NMS阈值

### 跟踪不稳定

**问题**：目标跟踪不稳定，频繁切换ID
**解决方案**：
1. 调整跟踪器参数（如max_disappeared）
2. 提高检测精度
3. 增加最小命中次数（min_hits）
4. 考虑使用更稳定的跟踪算法

### 手势识别错误

**问题**：手势识别准确率低
**解决方案**：
1. 检查分类模型是否正确加载
2. 调整分类置信度阈值
3. 确保输入图像区域正确
4. 考虑重新训练或微调模型

### 性能问题

**问题**：系统运行速度慢
**解决方案**：
1. 启用GPU加速
2. 减小输入图像尺寸
3. 使用模型量化或剪枝
4. 优化代码逻辑

### 内存泄漏

**问题**：长时间运行后内存占用持续增长
**解决方案**：
1. 检查是否有未释放的资源
2. 使用内存分析工具定位问题
3. 确保循环中的对象及时释放
4. 检查是否有循环引用

## 开发工具

### 代码格式化

使用black进行代码格式化：
```bash
black src/ tests/
```

### 代码检查

使用pylint进行代码检查：
```bash
pylint src/
```

### 类型检查

使用mypy进行类型检查：
```bash
mypy src/
```

### 导入排序

使用isort整理导入：
```bash
isort src/ tests/
```

### 依赖管理

使用pip-tools管理依赖：
```bash
pip-compile requirements.in
pip-sync requirements.txt
```

## 贡献指南

1. Fork项目仓库
2. 创建功能分支：`git checkout -b feature/new-feature`
3. 提交更改：`git commit -am 'Add new feature'`
4. 推送分支：`git push origin feature/new-feature`
5. 创建Pull Request

### 提交信息规范

使用约定式提交格式：
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

类型包括：
- feat: 新功能
- fix: 修复bug
- docs: 文档更新
- style: 代码格式调整
- refactor: 代码重构
- test: 测试相关
- chore: 构建过程或辅助工具的变动

示例：
```
feat(detector): add support for custom model paths

Allow users to specify custom model paths in the configuration file.
This makes it easier to switch between different models without
changing the code.

Closes #123
```

## 发布流程

1. 更新版本号
2. 更新CHANGELOG.md
3. 创建Git标签
4. 构建发布包
5. 上传到PyPI（可选）

```bash
# 更新版本号
bump2version patch  # 或 minor, major

# 创建标签
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# 构建发布包
python setup.py sdist bdist_wheel

# 上传到PyPI
twine upload dist/*
```