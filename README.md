# 动态手势识别系统 (Dynamic Gesture Recognition System)

一个基于深度学习的实时动态手势识别系统，使用YOLOv10同时完成手部检测和手势分类，专为基于HaGRIDv2数据集训练的模型设计。该系统支持多目标跟踪和应用控制，适用于课堂教学、演示和互动场景。

## 项目结构

```
dynamic_recognition/
├── models/                     # 模型文件夹
│   ├── YOLOv10x_gestures.pt   # YOLOv10手势检测与分类模型
├── detector/                   # 检测模块
│   ├── __init__.py
│   ├── yolo_detector.py       # YOLO检测器
│   └── base_detector.py       # 检测器基类
├── tracker/                    # 跟踪模块
│   ├── __init__.py
│   ├── sort_tracker.py        # SORT跟踪算法
│   ├── kalman_filter.py       # 卡尔曼滤波器
│   └── track.py               # 跟踪对象定义
├── classifier/                 # 分类模块
│   ├── __init__.py
│   ├── gesture_classifier.py  # 手势分类器
│   └── base_classifier.py     # 分类器基类
├── controller/                 # 控制模块
│   ├── __init__.py
│   ├── main_controller.py     # 主控制器
│   └── gesture_controller.py  # 手势控制器
├── classroom/                  # 教学功能模块
│   ├── __init__.py
│   ├── presentation.py        # 课件控制
│   ├── annotation.py          # 屏幕标注
│   └── interaction.py         # 课堂互动
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── drawer.py              # 绘图工具
│   ├── config.py              # 配置管理
│   └── enums.py               # 枚举定义
├── data/                       # 数据文件夹
│   ├── gestures.yaml          # 手势定义文件
│   └── interaction_data.json  # 交互数据存储
├── logs/                       # 日志文件夹
├── exports/                    # 导出数据文件夹
├── requirements.txt            # 依赖包列表
├── config.yaml                # 系统配置文件
├── main.py                    # 系统入口
└── README.md                  # 项目说明
```

## 技术架构

### 系统架构设计

本系统采用分层架构设计，各模块职责清晰，便于维护和扩展：

1. **数据层 (Data Layer)**
   - 负责数据的存储和管理
   - 包括手势定义、交互数据、配置文件等
   - 支持多种数据格式：YAML、JSON、CSV等

2. **检测层 (Detection Layer)**
   - 基于YOLOv10实现手部检测和手势分类
   - 支持多种输入源：摄像头、视频文件、图像序列
   - 提供检测结果的后处理和过滤机制

3. **跟踪层 (Tracking Layer)**
   - 使用SORT算法实现多目标跟踪
   - 基于卡尔曼滤波器预测目标位置
   - 处理目标遮挡、消失和重现情况

4. **分类层 (Classification Layer)**
   - 对检测到的手部区域进行精细分类
   - 支持多种分类算法和模型
   - 提供分类结果的置信度和稳定性评估

5. **控制层 (Control Layer)**
   - 将手势识别结果映射为具体操作
   - 支持自定义手势映射规则
   - 处理多手势组合和序列识别

6. **应用层 (Application Layer)**
   - 实现具体的应用功能
   - 包括课件控制、屏幕标注、互动记录等
   - 提供用户界面和交互反馈

### 核心算法

1. **YOLOv10检测算法**
   - 实时目标检测框架
   - 同时完成检测和分类任务
   - 支持多种输入分辨率和批处理

2. **SORT跟踪算法**
   - 简单高效的在线多目标跟踪算法
   - 基于卡尔曼滤波器和匈牙利算法
   - 处理目标关联和轨迹管理

3. **手势稳定性检查**
   - 基于时间窗口的手势稳定性评估
   - 考虑手势持续时间和置信度变化
   - 减少误识别和抖动

4. **手势序列识别**
   - 基于隐马尔可夫模型(HMM)的序列识别
   - 支持自定义手势序列模板
   - 处理时序依赖和上下文信息

### 性能优化策略

1. **模型优化**
   - 模型量化和剪枝
    - 知识蒸馏和模型压缩
    - 多尺度推理和特征金字塔

2. **计算优化**
    - GPU加速和并行计算
    - 多线程处理和流水线设计
    - 内存池和对象复用

3. **算法优化**
    - 早期退出和自适应推理
    - 区域提议和注意力机制
    - 动态阈值和自适应参数

### 环境配置注意事项

项目使用environment.yml管理依赖，确保了环境的一致性和可复现性。主要优势：
- 自动处理依赖版本兼容性
- 同时支持conda和pip包管理
- 支持CPU和GPU环境配置
- 简化环境配置流程，避免依赖冲突

### 扩展性设计

1. **模块化设计**
   - 松耦合的模块结构
   - 标准化的接口定义
   - 插件式的功能扩展

2. **配置驱动**
   - 外部配置文件控制
   - 运行时参数调整
   - 多环境配置管理

3. **多平台支持**
   - 跨平台兼容性设计
   - 抽象硬件接口
   - 条件编译和平台适配

## 功能特点

### 核心功能
- **实时手势检测与分类**: 使用YOLOv10同时检测手部区域并分类手势类型，基于HaGRIDv2数据集的33种手势，另包含无手势(no_gesture)类别
- **多目标跟踪**: 使用SORT算法跟踪多个手部目标
- **手势稳定性检查**: 基于HaGRIDv2数据集的特定参数，确保手势识别的稳定性
- **动作映射**: 将识别的手势映射到具体应用操作
- **可视化界面**: 实时显示检测结果和手势信息

### 应用功能
- **课件控制**: 通过手势控制幻灯片切换、缩放等操作
- **屏幕标注**: 支持多种标注工具，如画笔、线条、形状等
- **互动记录**: 记录和分析课堂互动数据，支持导出

### 高级功能
- **手势序列识别**: 支持识别连续的手势序列，实现复杂操作控制
- **实时性能监控**: 提供系统性能监控界面，实时显示FPS、检测数量等指标
- **多语言支持**: 支持中文、英文等多种语言界面切换


## 安装与使用

### 环境要求
- Python 3.9+
- OpenCV
- PyTorch
- ONNX Runtime
- NumPy
- PyYAML
- PyAutoGUI
- filterpy
- ultralytics

### 安装步骤

1. 克隆项目
```bash
git clone https://github.com/yourusername/dynamic_gestures.git
cd dynamic_gestures/project
```

2. 使用environment.yml创建并激活conda环境（推荐）
```bash
# 使用environment.yml创建环境
conda env create -f environment.yml

# 激活环境
conda activate gesture_recognition

# 如果需要GPU版本，可以修改environment.yml文件，添加GPU相关依赖
# 例如：将cpuonly替换为pytorch-cuda=11.8（根据您的CUDA版本调整）
```

3. 下载模型文件
下载基于HaGRIDv2数据集训练的YOLOv10模型：
- 从以下链接下载模型文件：[HaGRIDv2数据集](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/hagrid)
- 将模型文件放置在 `models/` 目录下，命名为 `YOLOv10x_gestures.pt`

或者使用自己训练的模型：
- 确保模型基于HaGRIDv2数据集训练
- 将模型文件放置在 `models/` 目录下
- 在 `config.yaml` 中更新模型路径

### 使用方法

1. 基本使用（使用默认摄像头）
```bash
python main.py --display
```

2. 使用指定摄像头
```bash
python main.py --camera 1 --display
```

3. 处理视频文件
```bash
python main.py --video path/to/video.mp4 --output output.mp4 --display
```

4. 自定义配置
```bash
python main.py --config custom_config.yaml --display
```

5. 设置日志级别
```bash
python main.py --log-level DEBUG --log-file app.log --display
```

## 手势定义

基于HaGRIDv2数据集，系统支持以下33种手势：

1. **call** - 打电话：模拟打电话的手势，拇指和小指伸出
2. **dislike** - 不喜欢：大拇指朝下的手势
3. **fist** - 拳头：握拳手势
4. **four** - 四指：除大拇指外的四指伸出
5. **like** - 点赞：大拇指朝上的手势
6. **mute** - 静音：食指放在嘴唇上的静音手势
7. **ok** - OK：拇指和食指形成圆圈的OK手势
8. **one** - 一指：食指伸出的指向手势
9. **palm** - 手掌：五指张开的手掌手势
10. **peace** - 和平：食指和中指伸出的剪刀手/和平手势
11. **rock** - 摇滚：食指和小指伸出的摇滚手势
12. **stop** - 停止：手掌朝前的停止手势
13. **stop_inverted** - 反向停止：手掌朝后的反向停止手势
14. **three** - 三指：拇指、食指和中指伸出的三指手势
15. **two_up** - 两指向上：拇指和食指伸出的V字手势
16. **two_up_inverted** - 两指向下：拇指和食指向下的倒V字手势
17. **three_gun** - 三指枪：食指、中指和拇指组成的枪手势
18. **grip** - 握持：握拳但拇指伸出的手势
19. **grabbing** - 抓取：五指弯曲抓取的手势
20. **hand_heart** - 手心比心：双手组成心形手势
21. **hand_heart2** - 手背比心：手背组成心形手势
22. **holy** - 圣洁：食指和中指并拢，其他手指弯曲的手势
23. **little_finger** - 小指：小指伸出，其他手指弯曲的手势
24. **middle_finger** - 中指：竖中指手势
25. **point** - 指向：食指伸出指向手势
26. **take_picture** - 拍照：模拟拍照的V字手势
27. **thumb_index** - 拇指食指：拇指和食指伸出的手势
28. **thumb_index2** - 拇指食指2：另一种拇指和食指伸出的手势
29. **three3** - 三指3：第三种三指手势
30. **timeout** - 超时：T字形手势
31. **xsign** - X手势：X形状手势
32. **no_gesture** - 无手势：未识别到有效手势

每种手势都有特定的稳定性检查和持续时间要求，确保识别的准确性。

详细的手势定义和参数配置请参考`data/gestures.yaml`文件。

## PPT控制手势映射

由于当前模型对动态手势识别能力有限，系统使用静态手势控制PPT操作：

| 手势 | 功能 |
|------|------|
| point (指向) | 下一页 |
| peace (和平) | 上一页 |
| xsign (X手势) | 开始放映 |
| stop (停止) | 结束放映 |
| grabbing (抓取) | 放大 |
| fist (拳头) | 缩小 |

## 屏幕标注手势映射

系统支持使用手势进行屏幕标注，可在演示过程中添加标记和注释：

| 手势 | 功能 |
|------|------|
| grab (抓取) | 开始绘制 |
| release (释放) | 结束绘制 |

使用说明：
1. 做出"grab"手势（五指弯曲抓取的手势）开始绘制模式
2. 在绘制模式下，移动手部即可在屏幕上绘制轨迹
3. 做出"release"手势（释放手势）结束绘制模式
4. 绘制内容会叠加在当前显示内容上，不影响原始演示内容
5. 屏幕标注功能与PPT控制功能可同时使用，实现演示与标注的无缝切换

## 配置说明

系统配置文件为`config.yaml`，包含以下主要配置项：

- **detector**: 检测器配置
- **tracker**: 跟踪器配置
- **classifier**: 分类器配置
- **main_controller**: 主控制器配置
- **gesture_controller**: 手势控制器配置
- **presentation**: 课件控制器配置
- **annotation**: 标注控制器配置
- **interaction**: 互动控制器配置
- **drawer**: 绘图工具配置
- **system**: 系统配置
- **performance**: 性能配置
- **debug**: 调试配置

## 性能优化

1. **GPU加速**: 设置`detector.device`和`classifier.device`为`cuda`
2. **模型量化**: 使用TensorRT进行模型优化
3. **多线程处理**: 调整`performance.num_threads`参数
4. **分辨率调整**: 降低输入图像分辨率以提高帧率

## 故障排除

1. **摄像头无法打开**: 检查摄像头ID和权限
2. **模型加载失败**: 确认模型文件路径和格式
3. **识别率低**: 调整置信度阈值和光照条件
4. **性能问题**: 降低分辨率或启用GPU加速

## 开发与扩展

### 添加新手势
1. 在`data/gestures.yaml`中定义新手势
2. 收集和标注训练数据
3. 训练新的分类模型
4. 更新`classifier/gesture_classifier.py`中的类别列表

### 添加新功能
1. 在相应模块中创建新类
2. 继承适当的基类
3. 实现必要的方法
4. 在控制器中集成新功能

## 未来计划

### 计划实现的功能

以下功能目前尚未实现，计划在未来的版本中开发：

#### 短期计划 (v1.4.0)
- **多用户交互**: 支持多人同时进行手势交互，适用于协作场景
- **自定义手势映射**: 允许用户自定义手势与操作的映射关系
- **云端数据同步**: 支持将交互数据同步到云端，实现跨设备数据共享
- **AI辅助分析**: 基于机器学习算法分析用户交互模式，提供优化建议
- **增强手势稳定性**: 优化手势识别算法，提高在复杂环境下的识别稳定性
- **3D手势识别**: 支持深度摄像头，实现3D空间中的手势识别
- **手势编辑器**: 开发可视化手势编辑器，简化自定义手势创建过程
- **插件系统**: 实现插件架构，支持第三方功能扩展

#### 中期计划 (v2.0.0)
- **全身姿态识别**: 扩展到手部以外的身体部位识别，支持全身姿态交互
- **情感识别**: 结合面部表情分析，实现情感感知交互
- **跨平台支持**: 开发移动端和Web端版本，实现多平台协同
- **云端模型服务**: 提供云端模型推理服务，降低本地硬件要求
- **开发者API**: 提供完整的API接口，方便第三方集成

#### 长期愿景
- **多模态融合**: 融合语音、视觉、触觉等多种交互方式
- **自适应学习**: 系统根据用户习惯自动优化识别模型
- **边缘计算部署**: 优化模型大小和计算需求，支持边缘设备部署
- **开源社区**: 建立开源社区，促进技术共享和协作开发
- **商业化应用**: 探索商业化应用场景，形成完整的解决方案

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献

欢迎提交问题报告和功能请求。如需贡献代码，请遵循以下步骤：

1. Fork本项目
2. 创建功能分支
3. 提交更改
4. 推送到分支

## 常见问题

### Q: 如何提高手势识别的准确率？
A: 可以通过以下方式提高准确率：
1. 调整置信度阈值（在config.yaml中）
2. 确保良好的光照条件
3. 手势保持稳定且清晰
4. 使用高分辨率摄像头

### Q: 系统支持哪些平台？
A: 目前支持Windows、Linux和macOS平台，需要Python 3.7+环境。

### Q: 如何添加自定义手势？
A: 请参考"开发与扩展"部分的"添加新手势"小节。

### Q: 系统性能如何优化？
A: 请参考"性能优化"部分，主要包括GPU加速、模型量化、多线程处理等方法。

## 技术支持

如有问题或建议，请通过以下方式联系：
- 提交Issue: [GitHub Issues](https://github.com/chano363/gestureflow/issues)

## 致谢

感谢以下开源项目和数据集：
- [HaGRIDv2数据集](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/hagrid)
- [YOLOv10](https://github.com/THU-MIG/yolov10)
- [SORT算法](https://github.com/abewley/sort)