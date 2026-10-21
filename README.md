cf穿越火线数据集5w yolo目标检测

cfyolo数据集
5万cf两类潜伏者保卫者yolo精标数据集 纯手动标注

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fef99fc3a65e47d7967878747a6cb436.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/82e86010edb747d9af81ec859f40aa3a.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9db0dcdc559f45d2a08679becf51f2b1.png)


---

### 📦 项目名称：CF-YOLO50K (穿越火线阵营识别数据集)

#### 1. 数据集概况 (Dataset Overview)

| 属性 | 描述 |
| :--- | :--- |
| **数据总量** | **50,000+** 张高分辨率游戏截图 |
| **标注方式** | **100% 纯人工手动标注** (拒绝半自动/伪标签，确保边界框紧贴角色模型) |
| **目标类别** | **2 类**：1. `潜伏者` (Black List / Terrorists)2. `保卫者` (Global Risk / CT) |
| **场景覆盖** | 运输船、黑色城镇、供电所、潜艇等全主流地图；涵盖白天/黑夜模式、烟雾弹干扰、闪光白屏、远距离/近距离、不同皮肤/角色模型。 |
| **标注格式** | YOLO 格式 (`class_id x_center y_center width height`) |
| **适用模型** | YOLOv5, v8, v9, v10, v11, RT-DETR 等 |
| **应用场景** | 电竞战术分析、自动击杀回放、AI 辅助瞄准研究、游戏直播实时阵营识别。 |

#### 2. 数据集目录结构
```text
CF-YOLO50K/
├── images/
│   ├── train/  (40,000 张)
│   └── val/    (10,000 张)
├── labels/
│   ├── train/  (对应的 .txt 文件)
│   └── val/
├── classes.txt (类别名称)
└── cf_data.yaml (YOLO 配置文件)
```

**`classes.txt` 内容:**
```text
0: 潜伏者
1: 保卫者
```

**`cf_data.yaml` 配置示例:**
```yaml
path: /datasets/CF-YOLO50K  # 数据集根目录
train: images/train
val: images/test

# 类别数量
nc: 2

# 类别名称
names:
  0: 潜伏者
  1: 保卫者
```

---

### 💻 核心训练代码 (基于 YOLOv8/v10)

这里提供一套完整的、针对游戏画面优化的训练脚本。游戏画面通常色彩鲜艳、对比度高，且角色移动速度快，因此需要特定的增强策略。

#### 1. 环境准备
```bash
pip install ultralytics opencv-python torch torchvision
```

#### 2. Python 训练脚本 (`train_cf.py`)

```python
from ultralytics import YOLO
import os

def train_crossfire_model():
    # 1. 加载预训练模型 (推荐使用 n 或 s 版本以获得高FPS，m 版本平衡精度)
    # 如果是从头训练，可以加载 'yolov8n.pt'
    model = YOLO('yolov8n.pt') 

    # 2. 定义训练参数
    # 游戏画面特点：目标小、运动快、背景复杂 -> 需要更强的增强
    results = model.train(
        data='cf_data.yaml',       # 数据集配置文件路径
        epochs=100,                # 训练轮次 (5w数据量通常50-100轮收敛)
        imgsz=640,                 # 输入分辨率 (可根据显卡显存调整为 1280 以提高小目标检测)
        batch=32,                  # 批次大小
        workers=8,                 # 数据加载线程数
        device='0',                # 使用 GPU 0
        optimizer='SGD',           # SGD 通常在目标检测中泛化性更好
        lr0=0.01,                  # 初始学习率
        lrf=0.1,                   # 最终学习率 (lr0 * lrf)
        
        # --- 针对游戏画面的特殊增强 ---
        augment=True,              # 开启数据增强
        hsv_h=0.015,               # 色调增强 (游戏皮肤颜色多变，不宜过大)
        hsv_s=0.7,                 # 饱和度增强 (游戏画面色彩鲜艳)
        hsv_v=0.4,                 # 亮度增强 (模拟烟雾/黑暗场景)
        degrees=0.0,               # 旋转角度 (FPS游戏地平线通常水平，不建议大角度旋转)
        translate=0.1,             # 平移
        scale=0.5,                 # 缩放 (模拟远近不同距离的敌人)
        shear=0.0,                 # 剪切
        perspective=0.0,           # 透视变换 (游戏引擎已有透视，无需额外变换)
        flipud=0.0,                # 上下翻转 (游戏角色不会倒立，必须为0)
        fliplr=0.5,                # 左右翻转 (合理)
        mosaic=1.0,                # Mosaic 增强 (对密集战斗场景非常有效)
        mixup=0.1,                 # Mixup 增强
        
        # 其他优化
        patience=20,               # 早停机制 (20轮不提升则停止)
        save=True,                 # 保存 checkpoints
        save_period=-1,            # 每个epoch都保存
        verbose=True,              # 详细日志
        project='runs/detect',     # 保存路径
        name='cf_yolov8n_v1',      # 实验名称
        exist_ok=False,            # 覆盖现有实验
        seed=42                    # 随机种子
    )
    
    print("✅ 训练完成！模型保存在:", results.save_dir)
    
    # 3. 验证模型精度
    metrics = model.val()
    print(f"🎯 mAP50: {metrics.box.map50:.4f}")
    print(f"🎯 mAP50-95: {metrics.box.map:.4f}")

    # 4. 导出模型 (可选，用于部署)
    # model.export(format='onnx', dynamic=True) # 导出为 ONNX 用于 C++/C# 部署
    # model.export(format='engine') # 导出为 TensorRT 用于高性能推理

if __name__ == '__main__':
    train_crossfire_model()
```

---

### 🚀 针对 CF 数据集的特殊优化建议

由于你提到是**纯手动标注**且数据量达到**5万**，这是非常高质量的数据。为了发挥最大价值，请注意以下几点：

#### 1. 解决“烟雾弹”和“闪光弹”干扰
*   **问题**：CF中烟雾弹会严重遮挡角色，导致检测丢失。
*   **对策**：在标注时，**即使角色被烟雾遮挡了50%以上，只要人眼能推断出轮廓，依然要标注**。这能教会模型学习“部分可见”的特征。
*   **代码调整**：在训练参数中保持 `mosaic=1.0`，Mosaic 增强能模拟遮挡效果，显著提升抗干扰能力。

#### 2. 小目标检测优化 (远距离敌人)
*   **问题**：在黑色城镇等大图远处，敌人像素占比极小。
*   **对策**：
    *   将 `imgsz` 从默认的 640 提升到 **1280** (如果显存允许，如 RTX 3090/4090)。
    *   或者修改模型头 (Head)，增加一个针对更小目标的检测层 (P2 层)。
    *   在 `cf_data.yaml` 中确保验证集包含大量远距离截图。

#### 3. 阵营混淆问题
*   **问题**：某些皮肤或光线条件下，潜伏者和保卫者颜色相近。
*   **对策**：检查数据集中是否有“镜像”错误。确保标注严格区分。YOLO 对纹理和颜色非常敏感，5万数据量足以让模型学会区分细微的制服差异（如袖标、帽子颜色）。

#### 4. 部署到游戏辅助/分析工具
如果你打算将此模型用于实时分析（例如 OBS 插件或本地分析器）：
*   **推理引擎**：不要直接用 PyTorch。导出为 **ONNX** -> **TensorRT** (NVIDIA显卡) 或 **OpenVINO** (Intel CPU)。
*   **帧率**：CF 通常是 144Hz 或 240Hz。你不需要每帧都检测。每隔 3-5 帧检测一次，结合 **BoTSORT** 或 **ByteTrack** 进行跟踪，即可实现流畅的实时阵营标记。

### 📈 预期效果
基于 5万 纯手动精标数据：
*   **mAP50**: 预计可达 **0.95 - 0.98** (两类分类相对简单，数据量大且标注质量高)。
*   **召回率 (Recall)**: 在烟雾场景下应保持在 **0.90** 以上。
*   **推理速度**: YOLOv8n 在 RTX 3060 上可达 **200+ FPS** (1080p)，完全满足实时需求。


