# API 文档

本文档详细描述了动态手势识别系统的API接口，包括类、方法和函数的详细说明。

## 环境设置

### 使用environment.yml创建conda环境（推荐）

```bash
# 使用environment.yml创建环境
conda env create -f environment.yml

# 激活环境
conda activate gesture_recognition

# 如果需要GPU版本，可以修改environment.yml文件，添加GPU相关依赖
# 例如：将cpuonly替换为pytorch-cuda=11.8（根据您的CUDA版本调整）

# 验证安装
python -c "import torch, cv2, numpy, pyyaml"  # 检查核心依赖是否安装成功
```

**注意**：项目使用environment.yml管理依赖，确保了环境的一致性和可复现性。environment.yml文件中已包含所有必要的依赖，包括Python 3.9、OpenCV、PyTorch、ONNX Runtime、PyAutoGUI等。

## 目录

- [核心模块](#核心模块)
  - [检测模块 (detector)](#检测模块-detector)
  - [跟踪模块 (tracker)](#跟踪模块-tracker)
  - [分类模块 (classifier)](#分类模块-classifier)
  - [控制模块 (controller)](#控制模块-controller)
  - [应用模块 (classroom)](#应用模块-classroom)
  - [工具模块 (utils)](#工具模块-utils)
- [数据结构](#数据结构)
- [配置系统](#配置系统)
- [异常处理](#异常处理)

## 核心模块

### 检测模块 (detector)

#### BaseDetector

检测器基类，定义了检测器的通用接口。

```python
class BaseDetector(ABC):
    """检测器基类"""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """检测图像中的目标
        
        Args:
            image: 输入图像，形状为 (H, W, C) 的numpy数组
            
        Returns:
            检测结果列表，每个Detection包含边界框和置信度
            
        Raises:
            DetectionError: 检测过程中出现错误
        """
        pass
```

#### YOLODetector

基于YOLOv10的检测器实现。

```python
class YOLODetector(BaseDetector):
    """YOLOv10检测器"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, 
                 nms_threshold: float = 0.4, device: str = "cpu", 
                 input_size: Tuple[int, int] = (640, 640)):
        """初始化YOLO检测器
        
        Args:
            model_path: 模型文件路径
            confidence_threshold: 置信度阈值，范围[0.0, 1.0]
            nms_threshold: 非极大值抑制阈值，范围[0.0, 1.0]
            device: 计算设备，"cpu"或"cuda"
            input_size: 模型输入尺寸 (width, height)
            
        Raises:
            ModelLoadError: 模型加载失败
            DeviceError: 指定设备不可用
        """
        # 初始化代码
        
    def detect(self, image: np.ndarray) -> List[Detection]:
        """检测图像中的目标
        
        Args:
            image: 输入图像，形状为 (H, W, C) 的numpy数组
            
        Returns:
            检测结果列表，每个Detection包含边界框和置信度
            
        Raises:
            DetectionError: 检测过程中出现错误
            InvalidInputError: 输入图像格式不正确
        """
        # 检测实现
```

### 跟踪模块 (tracker)

#### Track

跟踪对象类，表示被跟踪的单个目标。

```python
class Track:
    """跟踪对象"""
    
    def __init__(self, track_id: int, bbox: BoundingBox, 
                 class_id: Optional[int] = None, 
                 confidence: float = 1.0, 
                 time_since_update: int = 0, 
                 hit_streak: int = 1):
        """初始化跟踪对象
        
        Args:
            track_id: 跟踪ID
            bbox: 边界框
            class_id: 类别ID（可选）
            confidence: 置信度
            time_since_update: 自上次更新以来的帧数
            hit_streak: 连续匹配的帧数
        """
        # 初始化代码
        
    def predict(self) -> None:
        """预测下一帧的位置
        
        使用卡尔曼滤波器预测目标在下一帧中的位置
        """
        # 预测实现
        
    def update(self, detection: Detection) -> None:
        """使用检测结果更新跟踪对象
        
        Args:
            detection: 检测结果，用于更新跟踪对象
        """
        # 更新实现
```

#### SORTTracker

基于SORT算法的多目标跟踪器。

```python
class SORTTracker:
    """SORT多目标跟踪器"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: int = 50):
        """初始化SORT跟踪器
        
        Args:
            max_disappeared: 目标消失的最大帧数，超过此值将被删除
            max_distance: 最大关联距离，用于匈牙利算法
        """
        # 初始化代码
        
    def update(self, detections: List[Detection]) -> List[Track]:
        """更新跟踪器
        
        Args:
            detections: 当前帧的检测结果列表
            
        Returns:
            更新后的跟踪对象列表
        """
        # 更新实现
```

### 分类模块 (classifier)

#### BaseClassifier

分类器基类，定义了分类器的通用接口。

```python
class BaseClassifier(ABC):
    """分类器基类"""
    
    @abstractmethod
    def classify(self, image: np.ndarray, bbox: BoundingBox) -> ClassificationResult:
        """分类图像中的手势
        
        Args:
            image: 输入图像，形状为 (H, W, C) 的numpy数组
            bbox: 手部边界框
            
        Returns:
            分类结果，包含类别ID和置信度
            
        Raises:
            ClassificationError: 分类过程中出现错误
        """
        pass
```

#### GestureClassifier

基于深度学习的手势分类器。

```python
class GestureClassifier(BaseClassifier):
    """手势分类器"""
    
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (224, 224), 
                 device: str = "cpu"):
        """初始化手势分类器
        
        Args:
            model_path: 模型文件路径
            input_size: 模型输入尺寸 (width, height)
            device: 计算设备，"cpu"或"cuda"
            
        Raises:
            ModelLoadError: 模型加载失败
            DeviceError: 指定设备不可用
        """
        # 初始化代码
        
    def classify(self, image: np.ndarray, bbox: BoundingBox) -> ClassificationResult:
        """分类图像中的手势
        
        Args:
            image: 输入图像，形状为 (H, W, C) 的numpy数组
            bbox: 手部边界框
            
        Returns:
            分类结果，包含类别ID和置信度
            
        Raises:
            ClassificationError: 分类过程中出现错误
            InvalidInputError: 输入图像或边界框格式不正确
        """
        # 分类实现
```

### 控制模块 (controller)

#### MainController

系统主控制器，协调各个模块并处理手势事件。

```python
class MainController:
    """主控制器"""
    
    def __init__(self, config: Config):
        """初始化主控制器
        
        Args:
            config: 系统配置对象
            
        Raises:
            ConfigError: 配置错误
            ComponentInitError: 组件初始化失败
        """
        # 初始化代码
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """处理单帧图像
        
        Args:
            frame: 输入帧，形状为 (H, W, C) 的numpy数组
            
        Returns:
            处理结果，包含处理后的图像和手势结果列表
            
        Raises:
            ProcessingError: 处理过程中出现错误
        """
        # 处理实现
```

#### GestureController

手势控制器，负责手势识别和稳定性检查。

```python
class GestureController:
    """手势控制器"""
    
    def __init__(self, config: Config):
        """初始化手势控制器
        
        Args:
            config: 系统配置对象
        """
        # 初始化代码
        
    def update_state(self, track_id: int, gesture_result: Dict) -> None:
        """更新手势状态
        
        Args:
            track_id: 跟踪ID
            gesture_result: 手势识别结果
        """
        # 更新实现
        
    def register_gesture_callback(self, callback: Callable) -> None:
        """注册手势回调函数
        
        Args:
            callback: 回调函数，接收(gesture_id, gesture_name, track_id)参数
        """
        # 注册实现
```

### 应用模块 (classroom)

#### PresentationController

课件控制器，负责通过手势控制幻灯片。

```python
class PresentationController:
    """课件控制器"""
    
    def __init__(self, config: Config):
        """初始化课件控制器
        
        Args:
            config: 系统配置对象
        """
        # 初始化代码
        
    def next_slide(self) -> None:
        """切换到下一张幻灯片"""
        # 实现幻灯片切换
        
    def previous_slide(self) -> None:
        """切换到上一张幻灯片"""
        # 实现幻灯片切换
        
    def zoom_in(self) -> None:
        """放大幻灯片"""
        # 实现放大功能
        
    def zoom_out(self) -> None:
        """缩小幻灯片"""
        # 实现缩小功能
        
    def start_slideshow(self) -> None:
        """开始幻灯片放映"""
        # 实现开始放映功能
        
    def end_slideshow(self) -> None:
        """结束幻灯片放映"""
        # 实现结束放映功能
```

#### AnnotationController

标注控制器，负责屏幕标注功能。

```python
class AnnotationController:
    """标注控制器"""
    
    def __init__(self, config: Config):
        """初始化标注控制器
        
        Args:
            config: 系统配置对象
        """
        # 初始化代码
        
    def start_drawing(self) -> None:
        """开始绘制"""
        # 实现绘制开始
        
    def end_drawing(self) -> None:
        """停止绘制"""
        # 实现绘制停止
        
    def update_frame(self, frame: np.ndarray) -> None:
        """更新当前帧
        
        Args:
            frame: 当前帧图像
        """
        # 更新帧实现
```

#### InteractionController

互动控制器，负责课堂互动功能。

```python
class InteractionController:
    """互动控制器"""
    
    def __init__(self, config: Config):
        """初始化互动控制器
        
        Args:
            config: 系统配置对象
        """
        # 初始化代码
        
    def start_session(self) -> None:
        """开始互动会话"""
        # 实现会话开始
        
    def end_session(self) -> None:
        """结束互动会话"""
        # 实现会话结束
        
    def record_gesture(self, gesture_id: int, gesture_name: str, track_id: int) -> None:
        """记录手势
        
        Args:
            gesture_id: 手势ID
            gesture_name: 手势名称
            track_id: 跟踪ID
        """
        # 实现手势记录
        
    def export_data(self, file_path: str) -> None:
        """导出互动数据
        
        Args:
            file_path: 导出文件路径
        """
        # 实现数据导出
```

### 工具模块 (utils)

#### Config

配置管理类，负责加载和管理系统配置。

```python
class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径
            
        Raises:
            ConfigLoadError: 配置文件加载失败
            ConfigParseError: 配置文件解析失败
        """
        # 初始化代码
        
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键，如 "detector.confidence_threshold"
            default: 默认值，当键不存在时返回
            
        Returns:
            配置值
        """
        # 获取实现
```

#### Drawer

绘图工具类，负责在图像上绘制检测结果和手势信息。

```python
class Drawer:
    """绘图工具类"""
    
    def __init__(self, config: Config):
        """初始化绘图工具
        
        Args:
            config: 系统配置对象
        """
        # 初始化代码
        
    def draw_detection(self, image: np.ndarray, detection: Detection) -> np.ndarray:
        """绘制检测结果
        
        Args:
            image: 输入图像
            detection: 检测结果
            
        Returns:
            绘制后的图像
        """
        # 绘制实现
```

## 数据结构

### Detection

检测结果数据结构。

```python
@dataclass
class Detection:
    """检测结果"""
    bbox: BoundingBox  # 边界框
    confidence: float  # 置信度
    class_id: int  # 类别ID
    class_name: str  # 类别名称
```

### BoundingBox

边界框数据结构。

```python
@dataclass
class BoundingBox:
    """边界框"""
    x: float  # 左上角x坐标
    y: float  # 左上角y坐标
    width: float  # 宽度
    height: float  # 高度
    
    @property
    def x2(self) -> float:
        """右下角x坐标"""
        return self.x + self.width
        
    @property
    def y2(self) -> float:
        """右下角y坐标"""
        return self.y + self.height
        
    @property
    def center(self) -> Tuple[float, float]:
        """中心点坐标"""
        return (self.x + self.width / 2, self.y + self.height / 2)
```

### ClassificationResult

分类结果数据结构。

```python
@dataclass
class ClassificationResult:
    """分类结果"""
    class_id: int  # 类别ID
    class_name: str  # 类别名称
    confidence: float  # 置信度
    probabilities: Dict[int, float]  # 各类别的概率分布
```

### GestureEvent

手势事件数据结构。

```python
@dataclass
class GestureEvent:
    """手势事件"""
    gesture_type: GestureType  # 手势类型
    track_id: int  # 跟踪ID
    confidence: float  # 置信度
    timestamp: float  # 时间戳
    position: Tuple[float, float]  # 手势位置
    duration: float  # 手势持续时间
```

## 配置系统

### 配置文件结构

配置文件采用YAML格式，主要包含以下部分：

```yaml
# 检测器配置
detector:
  type: "yolo"
  model_path: "models/YOLOv10x_gestures.pt"
  confidence_threshold: 0.3
  iou_threshold: 0.45
  device: "cuda"
  input_size: [640, 640]

# 跟踪器配置
tracker:
  type: "sort"
  max_disappeared: 30
  min_hits: 3
  max_age: 60
  iou_threshold: 0.3

# 分类器配置
classifier:
  type: "gesture"
  model_path: "models/gesture_classifier.onnx"
  input_size: [224, 224]
  confidence_threshold: 0.7
  device: "cpu"

# 主控制器配置
main_controller:
  fps: 30
  buffer_size: 1
  gesture_history_size: 5
  smoothing_window_size: 3
  gesture_duration_threshold: 0.5

# 手势控制器配置
gesture_controller:
  gesture_timeout: 0.5
  sequence_time_window: 1.0
  max_consecutive_gestures: 10
  min_confidence: 0.6
  min_duration: 0.3
  stability_frames: 3
  swipe_threshold: 30
  zoom_threshold: 0.05
  rotation_threshold: 10

# 课件控制器配置
presentation:
  auto_save: true
  save_interval: 60
  transition_duration: 0.5
  zoom_step: 0.1
  max_zoom: 3.0
  min_zoom: 0.5

# 标注控制器配置
annotation:
  line_width: 3
  point_radius: 5
  font_size: 16
  font_family: "Arial"
  default_color: [0, 255, 0]
  colors:
    - [255, 0, 0]
    - [0, 255, 0]
    - [0, 0, 255]
    - [255, 255, 0]
    - [255, 0, 255]
    - [0, 255, 255]
    - [128, 0, 128]
    - [255, 165, 0]

# 互动控制器配置
interaction:
  max_events: 1000
  save_interval: 30
  export_format: "csv"
  enable_statistics: true

# 绘图工具配置
drawer:
  show_fps: true
  show_tracking_id: true
  show_confidence: true
  show_gesture_name: true
  show_trajectory: true
  trajectory_length: 20
  trajectory_color: [0, 255, 255]
  bbox_thickness: 2
  text_thickness: 1

# 系统配置
system:
  log_level: "INFO"
  log_file: "logs/system.log"
  save_results: true
  results_dir: "results"
  camera_id: 0
  camera_width: 1280
  camera_height: 720
  auto_restart: true
  restart_interval: 3600

# 性能配置
performance:
  enable_gpu: true
  gpu_memory_fraction: 0.7
  num_threads: 4
  enable_tensorrt: false
  precision: "fp32"

# 调试配置
debug:
  save_frames: false
  frames_dir: "debug/frames"
  save_detections: false
  detections_dir: "debug/detections"
  save_tracks: false
  tracks_dir: "debug/tracks"
  save_gestures: false
  gestures_dir: "debug/gestures"
  visual_debug: true
```

## 异常处理

### 自定义异常

```python
class GestureRecognitionError(Exception):
    """手势识别系统基础异常"""
    pass

class ModelLoadError(GestureRecognitionError):
    """模型加载错误"""
    pass

class DeviceError(GestureRecognitionError):
    """设备错误"""
    pass

class DetectionError(GestureRecognitionError):
    """检测错误"""
    pass

class TrackingError(GestureRecognitionError):
    """跟踪错误"""
    pass

class ClassificationError(GestureRecognitionError):
    """分类错误"""
    pass

class ConfigError(GestureRecognitionError):
    """配置错误"""
    pass

class ConfigLoadError(ConfigError):
    """配置加载错误"""
    pass

class ConfigParseError(ConfigError):
    """配置解析错误"""
    pass

class ComponentInitError(GestureRecognitionError):
    """组件初始化错误"""
    pass

class ProcessingError(GestureRecognitionError):
    """处理错误"""
    pass

class InvalidInputError(GestureRecognitionError):
    """无效输入错误"""
    pass
```

### 异常处理示例

```python
try:
    # 初始化检测器
    detector = YOLODetector(model_path, device="cuda")
except ModelLoadError as e:
    logger.error(f"Failed to load model: {e}")
    # 使用备用模型或退出
except DeviceError as e:
    logger.warning(f"CUDA device not available, falling back to CPU: {e}")
    detector = YOLODetector(model_path, device="cpu")

try:
    # 处理帧
    result = controller.process_frame(frame)
except ProcessingError as e:
    logger.error(f"Error processing frame: {e}")
    # 跳过当前帧或使用备用处理方法
except InvalidInputError as e:
    logger.error(f"Invalid input frame: {e}")
    # 跳过当前帧
```

## 手势类型

系统基于HaGRIDv2数据集，支持以下16种基本手势：

1. **call** - 打电话手势
2. **dislike** - 不喜欢手势
3. **fist** - 拳头手势
4. **four** - 四指手势
5. **like** - 点赞手势
6. **mute** - 静音手势
7. **ok** - OK手势
8. **one** - 一指手势
9. **palm** - 手掌手势
10. **peace** - 和平手势
11. **rock** - 摇滚手势
12. **stop** - 停止手势
13. **stop_inverted** - 反向停止手势
根据HaGRIDv2数据集和实际模型训练，系统支持以下完整的33种手势：

1. **grabbing** - 抓取手势
2. **grip** - 抓握手势
3. **holy** - 神圣手势
4. **point** - 指向手势
5. **call** - 打电话手势
6. **three3** - 三指手势变体
7. **timeout** - 暂停手势
8. **xsign** - X手势
9. **hand_heart** - 手型心形1
10. **hand_heart2** - 手型心形2
11. **little_finger** - 小指手势
12. **middle_finger** - 中指手势
13. **take_picture** - 拍照手势
14. **dislike** - 不喜欢手势
15. **fist** - 拳头手势
16. **four** - 四指手势
17. **like** - 点赞手势
18. **mute** - 静音手势
19. **ok** - OK手势
20. **one** - 一指手势
21. **palm** - 手掌手势
22. **peace** - 和平手势
23. **peace_inverted** - 反向和平手势
24. **rock** - 摇滚手势
25. **stop** - 停止手势
26. **stop_inverted** - 反向停止手势
27. **three** - 三指手势
28. **three2** - 三指手势变体2
29. **two_up** - 两指向上手势
30. **two_up_inverted** - 两指向下手势
31. **three_gun** - 三指枪手势
32. **thumb_index** - 拇指食指手势1
33. **thumb_index2** - 拇指食指手势2

另外，系统还包括以下特殊类别：

- **no_gesture** - 无手势

在当前实现中，以下是实际用于控制操作的手势：

## 手势控制映射

系统支持以下手势控制操作，具体映射关系以当前代码实现为准：

- **next_slide**: 切换到下一张幻灯片（实际使用手势：two_up, peace）
- **previous_slide**: 切换到上一张幻灯片（实际使用手势：two_up_inverted）
- **zoom_in**: 放大幻灯片（实际使用手势：grabbing）
- **zoom_out**: 缩小幻灯片（实际使用手势：fist）
- **start_slideshow**: 开始幻灯片放映（实际使用手势：ok）
- **end_slideshow**: 结束幻灯片放映（实际使用手势：stop）
- **start_drawing**: 开始绘制（实际使用手势：one, point）
- **end_drawing**: 停止绘制（实际使用手势：palm, stop）