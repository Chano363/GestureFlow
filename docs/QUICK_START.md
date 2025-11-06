# 快速开始指南

本指南将帮助您快速上手动态手势识别系统，从安装到运行基本功能。

## 目录

- [系统要求](#系统要求)
- [安装步骤](#安装步骤)
- [快速运行](#快速运行)
- [基本使用](#基本使用)
- [常见问题](#常见问题)

## 系统要求

### 硬件要求

- **CPU**: Intel Core i5 或同等性能以上
- **内存**: 8GB RAM (推荐16GB)
- **摄像头**: USB摄像头或内置摄像头，支持720p或更高分辨率
- **GPU**: NVIDIA GPU (可选，用于加速，建议GTX 1060或更高)

### 软件要求

- **操作系统**: Windows 10/11, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8 或更高版本
- **CUDA**: 11.0+ (仅在使用GPU加速时需要)

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/gesture-recognition.git
cd gesture-recognition
```

### 2. 创建虚拟环境

使用environment.yml创建并激活conda环境（推荐）

```bash
# 使用environment.yml创建环境
conda env create -f environment.yml

# 激活环境
conda activate gesture_recognition

# 如果需要GPU版本，可以修改environment.yml文件，添加GPU相关依赖
# 例如：将cpuonly替换为pytorch-cuda=11.8（根据您的CUDA版本调整）
```

### 3. 下载预训练模型

系统需要两个预训练模型：手部检测模型和手势分类模型。

```bash
# 创建模型目录
mkdir -p models

# 下载手部检测模型 (YOLOv10)
wget https://example.com/models/YOLOv10x_gestures.pt -O models/YOLOv10x_gestures.pt

# 下载手势分类模型
wget https://example.com/models/gesture_classifier.onnx -O models/gesture_classifier.onnx
```

### 4. 验证安装

```bash
# 检查Python版本
python --version

# 检查PyTorch安装
python -c "import torch; print(torch.__version__)"

# 检查OpenCV安装
python -c "import cv2; print(cv2.__version__)"

# 检查模型文件
ls -la models/
```

## 快速运行

### 1. 基本运行

```bash
# 使用默认配置运行
python main.py
```

### 2. 使用摄像头运行

```bash
# 使用默认摄像头
python main.py --source 0

# 使用指定摄像头ID
python main.py --source 1

# 使用视频文件
python main.py --source path/to/video.mp4
```

### 3. 使用GPU加速

```bash
# 启用GPU加速
python main.py --device cuda

# 指定GPU设备ID
python main.py --device cuda:0
```

### 4. 保存结果

```bash
# 保存处理后的视频
python main.py --save-video --output-path results/output.mp4

# 保存检测结果
python main.py --save-detections --output-dir results/detections
```

## 基本使用

### 1. 系统界面

运行系统后，您将看到一个实时视频窗口，显示：

- 检测到的手部区域（绿色边界框）
- 手部跟踪ID（每个手部的唯一标识符）
- 识别的手势类型（如"point", "fist"等）
- 置信度分数（0-1之间的数值）
- 系统FPS（每秒帧数）

### 2. 支持的手势

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

### 3. 手势控制操作

在当前实现中，以下手势用于系统控制操作：

- **下一张幻灯片**：two_up、peace手势
- **上一张幻灯片**：two_up_inverted手势
- **放大**：grabbing手势
- **缩小**：fist手势
- **开始幻灯片放映**：ok手势
- **结束幻灯片放映**：stop手势
- **开始绘制**：one、point手势
- **结束绘制**：palm、stop手势

### 3. 基本控制

#### 退出系统

按 `q` 键或 `ESC` 键退出系统。

#### 暂停/继续

按 `空格` 键暂停/继续处理。

#### 保存当前帧

按 `s` 键保存当前帧。

#### 切换显示模式

按 `d` 键切换不同的显示模式：
- 模式1：显示检测结果和手势信息
- 模式2：仅显示检测结果
- 模式3：仅显示手势信息
- 模式4：不显示任何信息

### 4. 配置选项

您可以通过修改 `config.yaml` 文件来自定义系统行为：

```yaml
# 检测器配置
detector:
  confidence_threshold: 0.3  # 调整此值以提高/降低检测灵敏度
  device: "cuda"  # 使用GPU加速

# 跟踪器配置
tracker:
  max_disappeared: 30  # 调整此值以改变跟踪稳定性

# 分类器配置
classifier:
  confidence_threshold: 0.7  # 调整此值以提高/降低手势识别灵敏度

# 显示配置
drawer:
  show_fps: true  # 显示FPS
  show_confidence: true  # 显示置信度
  show_tracking_id: true  # 显示跟踪ID
```

### 5. 命令行参数

系统支持以下命令行参数：

```bash
python main.py [OPTIONS]

选项:
  --source, -s            输入源 (摄像头ID、视频文件路径)
                          [默认: 0]
  --config, -c            配置文件路径
                          [默认: config.yaml]
  --device, -d            计算设备 (cpu/cuda)
                          [默认: cpu]
  --save-video            保存处理后的视频
  --output-path, -o       输出文件路径
  --save-detections       保存检测结果
  --output-dir            输出目录
  --display, --no-display 显示/隐藏结果窗口
                          [默认: True]
  --help, -h              显示帮助信息
```

## 常见问题

### 1. 模型加载失败

**问题**: 启动时出现模型加载错误

**解决方案**:
1. 检查模型文件是否存在：
   ```bash
   ls -la models/
   ```
2. 确认模型文件路径是否正确，检查 `config.yaml` 中的路径设置
3. 验证模型文件是否完整，重新下载可能损坏的模型文件
4. 检查设备配置是否正确（CPU/GPU）

### 2. 摄像头无法打开

**问题**: 系统无法访问摄像头

**解决方案**:
1. 检查摄像头是否被其他应用程序占用
2. 尝试使用不同的摄像头ID：
   ```bash
   python main.py --source 1
   ```
3. 确认摄像头驱动程序已正确安装
4. 检查摄像头权限（Linux/Mac可能需要授权）

### 3. 检测精度低

**问题**: 手部检测效果不理想

**解决方案**:
1. 调整置信度阈值，降低 `detector.confidence_threshold` 值：
   ```yaml
   detector:
     confidence_threshold: 0.2  # 从0.3降低到0.2
   ```
2. 确保光照条件良好，避免背光
3. 尝试使用更高分辨率的摄像头
4. 确保手部在摄像头视野内，不要太远或太近

### 4. 手势识别错误

**问题**: 手势识别准确率低

**解决方案**:
1. 调整分类置信度阈值：
   ```yaml
   classifier:
     confidence_threshold: 0.6  # 从0.7降低到0.6
   ```
2. 确保手势清晰可见，避免遮挡
3. 保持手势稳定至少0.5秒
4. 检查手势是否在支持的手势列表中

### 5. 性能问题

**问题**: 系统运行缓慢，FPS低

**解决方案**:
1. 启用GPU加速：
   ```bash
   python main.py --device cuda
   ```
2. 降低输入分辨率：
   ```yaml
   system:
     camera_width: 640   # 从1280降低到640
     camera_height: 480   # 从720降低到480
   ```
3. 调整处理参数：
   ```yaml
   main_controller:
     fps: 15  # 从30降低到15
   ```

### 6. 内存不足

**问题**: 运行时出现内存不足错误

**解决方案**:
1. 减小批处理大小：
   ```yaml
   performance:
     buffer_size: 1  # 从更大的值减小到1
   ```
2. 降低输入分辨率
3. 关闭不必要的应用程序释放内存
4. 使用更轻量级的模型

### 7. 安装依赖失败

**问题**: pip install 失败

**解决方案**:
1. 更新pip：
   ```bash
   pip install --upgrade pip
   ```
2. 使用国内镜像源：
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
3. 逐个安装依赖，找出问题包：
   ```bash
   pip install opencv-python
   pip install torch
   # ...
   ```

## 下一步

恭喜！您已经成功安装并运行了动态手势识别系统。接下来，您可以：

1. 阅读[API文档](API_DOCUMENTATION.md)了解系统API
2. 查看[开发者指南](DEVELOPER_GUIDE.md)学习如何扩展系统
3. 探索[配置选项](config.yaml)自定义系统行为
4. 尝试不同的应用场景，如PPT控制、屏幕标注等

如果您遇到任何问题，请查看项目的[常见问题](README.md#故障排除)部分或在GitHub上提交issue。