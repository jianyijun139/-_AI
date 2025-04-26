# YOLOv11电动自行车进电梯检测模型课程

-   数据集清洗
    -   清空测试集和验证集的错误标签
    -   校验图片格式是否可以被读取
-   读取数据集
    -   读取数据集配置文件
    -   读取标签文件计算标签
    -   展示部分数据集
-   数据集清洗
    -   清空测试集和验证集的错误标签
    -   校验图片格式是否可以被读取
-   开始训练
    -   训练参数
    -   训练
    -   训练效果
-   模型推理
    -   推理视频并绘制检测框
-   转换为 `onnx` 并推理
    -   转换`pt`模型为 `onnx`模型
    -   `onnx` 模型推理
-   总结

<!-- TOC -->

-   [电动自行车进电梯检测模型课程](#evmoto)
    -   [数据集清洗](#数据集清洗)
        -   [清空测试集和验证集的错误标签](#清空测试集和验证集的错误标签)
    -   [读取数据](#读取数据)
        -   [展示部分数据集合](#展示部分数据集合)
    -   [开始训练](#开始训练)
        -   [训练参数](#训练参数)
        -   [训练](#训练)
        -   [训练效果](#训练效果)
    -   [模型推理](#模型推理)
        -   [推理视频并绘制检测框](#推理视频并绘制检测框)
    -   [转换为 `onnx` 并推理](#转换为-onnx-并推理)
        _ [转换`pt`模型为 `onnx`模型](#转换pt模型为-onnx模型)
        _ [`onnx` 模型推理](#onnx-模型推理)
        <!-- TOC -->

完整项目代码实战操作，请进入“人工智能教学实训平台”[https://www.lswai.com](https://www.lswai.com) 体验完整操作流程。

[![rtx4090.jpeg](https://i-blog.csdnimg.cn/img_convert/88991aa879d98e824d1fbd2c7aca917d.jpeg)](https://www.lswai.com)


## 数据集清洗

### 清空测试集和验证集的错误标签

```python


def clean_corrupt_images(directory):
    """
    清理包含 JPEG 数据问题的图片。

    Args:
        directory (str): 要检查和清理的目录路径。
    """
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]
    total_images = 0
    removed_images = 0

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            total_images += 1

            # 检查文件扩展名
            if not any(file.lower().endswith(ext) for ext in valid_extensions):
                continue

            try:
                # 尝试打开并重新保存图片
                with Image.open(file_path) as img:
                    img = img.convert("RGB")  # 转换为 RGB 格式以重新保存
                    img.save(file_path, "JPEG")
            except Exception as e:
                # 如果发生异常，删除该文件
                print(f"删除损坏图片: {file_path} ({e})")
                os.remove(file_path)
                removed_images += 1

    print(f"扫描完成，总图片: {total_images}, 删除损坏图片: {removed_images}")



# 调用函数

# 设置要清理的目录
directory_to_clean = "/home/hhd/桌面/电动车/data/train/images"

# 调用函数
clean_corrupt_images(directory_to_clean)

# 示例用法
valid_folder = "/home/hhd/桌面/电动车/data/test/label"
image_folder = "/home/hhd/桌面/电动车/data/test/images"
delete_empty_txt_and_images(valid_folder, image_folder)
```

1. **清理包含问题的图片文件**

    - 第一段代码 `clean_corrupt_images(directory)` 的作用是扫描给定目录中的图片，尝试识别和修复可能存在问题的图片文件。如果图片无法正常打开或存在其他问题（如文件损坏），则删除这些文件。

    确保数据集中图片的完整性和可用性，避免模型训练或推理过程中因损坏图片导致问题。

    - **功能细节：**
        - 通过 `os.walk()` 遍历目录及其子目录中的所有文件。
        - 检查文件的扩展名是否为常见图片格式（`.jpg`, `.jpeg`, `.png` 等）。
        - 利用 `Pillow` 库尝试打开图片，并将其重新保存为 JPEG 格式。
        - 如果打开或保存图片时出现异常，认为该图片是损坏的并将其删除。

2) **清理空的标签文件和对应图片**
    - 第二段代码 `delete_empty_txt_and_images(valid_folder, image_folder)` 的作用是删除 `valid_folder` 中空的 `.txt` 文件，并删除 `image_folder` 中与这些空 `.txt` 文件同名的图片。
      测试集和验证集中的每张图片通常需要对应的标注文件。若标注文件为空，说明图片没有有效的标注，应删除该图片及对应标注，训练集可以使用空标注文件实现反例。
    - **功能细节：**
        - 遍历 `valid_folder` 中的所有文件，检查文件扩展名是否为 `.txt`。
        - 对于空的 `.txt` 文件，删除它们，并在 `image_folder` 中查找同名图片进行删除。
        - 通过 `os.path.getsize()` 检查文件大小是否为 0，来判定 `.txt` 文件是否为空。

通过清理无效标注文件和对应图片，维持数据集的准确性和一致性，避免模型在训练中受到无效数据的干扰。

## 读取数据

读取 `data.yaml` 配置文件

```python

    dataset_path = '/root/data_yaml'

    # 设置 YAML 文件的路径
    yaml_file_path = os.path.join(dataset_path, 'data.yaml')

    # 加载并打印 YAML 文件的内容
    with open(yaml_file_path, 'r') as file:
        yaml_content = yaml.load(file, Loader=yaml.FullLoader)
        print(yaml.dump(yaml_content, default_flow_style=False))

out:

names:
- Bicycle
- EVMotorcycle
nc: 2
test: ../test/images
train: ../train/images
val: ../valid/images

```

读取数据集，分析数据构成。
在训练模型之前，了解数据集中每个类别的分布很重要。通过统计标签数量，可以发现数据是否存在类别不平衡（某些类别样本过多或过少），这会直接影响模型的性能。
这份数据的自行车类别只为防止将自行车识别为电动车而添加的的反向实例 因此数量不多

```python
from collections import Counter
from pathlib import Path
import yaml

# 加载数据集配置文件
with open(yaml_file_path, 'r') as file:
    data_config = yaml.safe_load(file)

# # 获取每个类别的名称
class_names = data_config['names']
#
# # 合并所有统计结果
total_counts = train_counts + val_counts + test_counts
print('total_counts',total_counts)
# 输出每个类别的标签数量
for class_id, count in total_counts.items():
    print(f"类别 {class_names[class_id]} 有 {count} 个标签")

out:

data_config labels {'train': '../train/images', 'val': '../valid/images', 'test': '../test/images', 'nc': 2, 'names': ['Bicycle', 'EVMotorcycle']}
counting labels
622
counting labels
1276
counting labels
13207
Counter({1: 688, 0: 54}) Counter({1: 1446, 0: 122}) Counter({1: 14709, 0: 1401})
total_counts Counter({1: 16843, 0: 1577})
类别 EVMotorcycle 有 16843 个标签
类别 Bicycle 有 1577 个标签


```

#### 展示部分数据集合

展示数据集 是否正确
使用 `plt` 构建 一个画布 绘制一些数据集 用于确认 图片可以被正常读取

```python


# 列出目录下所有jpg图片
image_files = [file for file in os.listdir(train_images_path) if file.endswith('.jpg')]

# 创建 3x4 子图
fig, axes = plt.subplots(3, 4, figsize=(20, 11))

plt.suptitle('Sample Images from Training Dataset', fontsize=20)
plt.tight_layout()
plt.show()

```

![img.png](https://i-blog.csdnimg.cn/img_convert/d2b4cd933645f73f3504f244a1cc4827.png)

## 开始训练

### 训练参数

```python
# Train the model on our custom dataset

results = model.train(
    data=yaml_file_path,  # Path to the dataset configuration file
    model=model_cfg_path,
    epochs=180,  # Number of epochs to train for
    imgsz=640,  # Size of input images as integer
    device=0,                # Device to run on, i.e. cuda device=0
    patience=20,  # Epochs to wait for no observable improvement for early stopping of training
    batch=80,  # Number of images per batch
    optimizer='auto',  # Optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
    lr0=0.0005,  # Initial learning rate
    lrf=0.1,  # Final learning rate (lr0 * lrf)
)

```

1. **`data`**

    - **解释**: 数据集配置文件的路径，通常是一个 YAML 文件，包含训练、验证、测试数据集的路径以及类别名称等信息。
    - **作用**: 告诉模型从哪里加载数据以及类别相关的信息。
    - **示例值**: `"data.yaml"`。

2. **`model`**

    - **解释**: 模型的配置文件路径，用于加载预定义的模型结构或自定义模型。
    - **作用**: 指定要训练的模型架构，支持加载预训练权重。
    - **示例值**: `"yolov8s.pt"`（我们这里使用 yolo11s.pt）。

3. **`epochs`**

    - **解释**: 训练的总轮数。
    - **作用**: 定义训练的迭代次数，影响模型学习的程度。太少可能导致欠拟合，太多可能导致过拟合。
    - **示例值**: `180`。

4. **`imgsz`**

    - **解释**: 输入图像的尺寸，通常为正方形的边长（如 `640` 或 `736`）。
    - **作用**: 指定模型输入的图像大小，影响模型的性能和推理速度。
    - **示例值**: `640`（训练和推理中图像会被缩放到这个大小）。

5. **`patience`**

    - **解释**: 提前停止（early stopping）的耐心值。指在验证集上若在指定轮数内无性能提升，则提前结束训练。
    - **作用**: 避免浪费计算资源并减少过拟合风险。
    - **示例值**: `25`（25 轮内无改进将停止训练）。

6. **`batch`**

    - **解释**: 每次训练迭代中处理的图像数量（批大小）。
    - **作用**: 控制内存占用和训练效率。较大的批次通常需要更多显存，但可能更稳定。
    - **示例值**: `80`。

7. **`optimizer`**

    - **解释**: 优化器类型，用于调整模型参数以最小化损失函数。
    - **作用**: 决定模型的学习方式。支持常见优化器如 SGD、Adam、AdamW 等。
    - **示例值**: `'auto'`（自动选择最优优化器）。

8. **`lr0`**

    - **解释**: 初始学习率。
    - **作用**: 控制模型训练时的初始步长。较小的学习率可以提高稳定性，但可能收敛较慢。
    - **示例值**: `0.0005`。

9. **`lrf`**

    - **解释**: 最终学习率相对于初始学习率的比例。
    - **作用**: 定义学习率从初始值到最终值的变化范围（`lr_final = lr0 * lrf`）。通常用于学习率衰减。
    - **示例值**: `0.1`（最终学习率是初始学习率的 10%）。

10. **`dropout`**
    - **解释**: Dropout 正则化的比例。
    - **作用**: 防止过拟合，通过随机丢弃神经元的方式增强模型的泛化能力。
    - **示例值**: `0.1`（每次训练会随机丢弃 10% 的神经元）。

### 训练

```python

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      30/180      14.8G     0.5278       0.37      1.031          8        640: 100%|██████████| 133/133 [00:39<00:00,  3.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 7/7 [00:02<00:00,  2.96it/s]
                   all       1276       1568      0.909      0.783      0.882      0.732


YOLO11n summary (fused): 238 layers, 2,582,542 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 7/7 [00:25<00:00,  3.69s/it]
                   all       1276       1568      0.909      0.783      0.883      0.732
                     0        116        122      0.865      0.721      0.825      0.682
                     1       1183       1446      0.953      0.845       0.94      0.782
Speed: 12.4ms preprocess, 0.5ms inference, 0.0ms loss, 0.2ms postprocess per image
```

-   **每组学习率（`lr/pg0、lr/pg1、lr/pg2`）：** 这些值表示神经网络中不同层组的学习率。较低的学习率意味着模型在训练期间更新其权重的速度更慢。各组之间一致的学习率表明学习过程中的统一调整。
-   **`50% IoU` 时的平均精度（`metrics/mAP50(B)`）：** 该指标衡量模型检测与地面实况至少有 50% 交并 (IoU) 的对象的准确性。 0.88 的分数表明该模型在此 IoU 阈值下很准确。
-   **`IoU` 从 `50%` 到 `95%` 的平均精度 (`metrics/mAP50-95(B)`)：** 这是在不同 IoU 阈值（从 50% 到 95%）下计算的 mAP 平均值。 0.73 分表示在这些不同阈值上总体准确度良好。
-   **精度（`metrics/precision(B)`）：** 精度衡量正确预测的正观测值与总预测正值的比率。 0.90 分意味着该模型的预测非常精确。
-   **召回率（`metrics/recall(B)`）：** 召回率计算正确预测的正观测值与实际类中所有观测值的比率。 0.78 的召回率表明该模型非常擅长查找数据集中的所有相关案例。
-   **模型计算复杂度（`model/GFLOPs`）：** 表示模型的计算需求，GFLOPs 值表明复杂度适中。
-   **推理速度 (`model/speed_PyTorch(ms)`)：** 模型进行单次预测（推理）所需的时间。 0.5 ms 相当快，这对于实时应用程序来说是很好的。
-   **训练损失（`train/box_loss、train/cls_loss、train/dfl_loss`）：** 这些是训练期间不同类型的损失。 “box_loss”是指边界框预测中的误差，“cls_loss”是指分类误差，“dfl_loss”是指分布焦点损失。值越低表示性能越好。
-   **验证损失（`val/box_loss、val/cls_loss、val/dfl_loss`）：** 与训练损失类似，这些是在验证数据集上计算的损失。它们让我们了解模型对新的、未见过的数据的推广效果如何。训练和验证的损失值几乎相似，表明模型没有过度拟合。

### 训练效果

![results.png](https://i-blog.csdnimg.cn/img_convert/031a225ca5b89dd66d70f9941f03e763.png)

![val_batch0_labels.jpg](https://i-blog.csdnimg.cn/img_convert/27d59803892c71c635c18f61bdbda387.jpeg)

-   **损失函数收敛情况：**

    -   `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `val/box_loss`, `val/cls_loss` 曲线显示出较好的下降趋势，表明模型在训练中逐渐学习到特征，损失在收敛。
    -   `val` 的损失（验证集）在前期下降明显，后期趋于平稳，表明模型在验证集上的表现也逐渐稳定。

-   **精度指标：**
    -   `metrics/mAP50` 和 `metrics/mAP50-95` 均呈上升趋势，尤其 `mAP50` 曲线已经趋近 0.9，说明模型在检测任务上取得了较高的准确率。
    -   `metrics/precision` 和 `metrics/recall` 表现良好，稳定上升并保持较高的值（约 0.9），表明模型具备较好的查准率和召回率。

*   **边界框预测：**

    -   物体边界框看起来较为精确，覆盖目标物体的关键区域，无明显过多或过少的检测现象。
    -   多数检测框匹配正确类别，说明分类性能较好。

*   **多目标检测：**

    -   在拥挤场景（如楼梯间或电动车停放区域），模型对多个目标的检测仍表现出较好的区分能力，边界框无明显重叠问题。

*   **异常场景：**
    -   在较复杂的场景（如背景干扰较多或物体遮挡较多的情况下），模型仍能正确识别大多数目标，但可能有少量漏检（如较小目标）或错误标注现象。

**优点**

1. **训练收敛性良好**：损失函数下降平稳，模型在验证集上的性能提升稳定。
2. **高精度检测**：从 `mAP`、`precision` 和 `recall` 的表现来看，模型对目标物体的检测精度较高。
3. **推理表现优秀**：图片中的检测框清晰、准确，适用于实际场景。

**不足及改进建议**

1. **复杂场景下的小目标检测：**

    - 如果场景中有小目标（如部分被遮挡的电动车细节），模型可能存在漏检。
    - **建议：** 增加小目标样本的权重，或使用更高分辨率的输入图片。

2. **类间混淆问题：**
    - 复杂背景或遮挡情况下，模型可能误将非目标区域标注为物体。
    - **建议：** 加强数据增强（如随机裁剪、背景变化），提升模型的鲁棒性。

## 模型推理

### 推理视频并绘制检测框

```python
from ultralytics import YOLO
import cv2


# 打开视频文件
cap = cv2.VideoCapture(source)

# 获取视频帧率和尺寸
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用 mp4 格式编码
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 推理并保存结果
for result in model(source, stream=True,conf=0.40,line_width=1):
    frame = result.orig_img  # 获取原始帧
    annotated_frame = result.plot()  # 绘制检测结果

    # 将带注释的帧写入输出视频
    out.write(annotated_frame)

# 释放资源
cap.release()
out.release()
print(f"推理结果已保存至 {output_path}")

```

从视频中来看 对应电动车的识别准确率非常高.

## 转换为 `onnx` 并推理

### 转换`pt`模型为 `onnx`模型

在 YOLO11 中，使用 `model.export(format="onnx")` 方法可以将模型导出为 ONNX 格式。
该方法提供了多个可选参数，允许用户根据需求自定义导出过程。
以下是这些参数的详细说明：

**`format`** (`str`, 默认值：`'torchscript'`):

-   指定导出模型的目标格式。
-   可选值包括 `'onnx'`、`'torchscript'`、`'tensorflow'` 等。
-   设置为 `'onnx'` 时，模型将导出为 ONNX 格式。

**`imgsz`** (`int` 或 `tuple`, 默认值：`640`):

-   定义模型输入的图像尺寸。
-   可以是整数（表示方形图像的边长）或元组 `(height, width)`（指定特定的高和宽）。
-   确保导出的模型与预期的输入尺寸匹配。

**`keras`** (`bool`, 默认值：`False`):

-   是否导出为 Keras 格式（适用于 TensorFlow SavedModel）。
-   设置为 `True` 时，模型将以 Keras 格式导出，便于在 TensorFlow 环境中使用。

**`optimize`** (`bool`, 默认值：`False`):

-   在导出为 TorchScript 时，是否应用针对移动设备的优化。
-   设置为 `True` 可减少模型大小并提升在移动设备上的性能。

**`half`** (`bool`, 默认值：`False`):

-   是否启用半精度（FP16）量化。
-   设置为 `True` 可减少模型大小，并在支持 FP16 的硬件上加速推理。

**`int8`** (`bool`, 默认值：`False`):

-   是否启用 INT8 量化。
-   设置为 `True` 可进一步压缩模型，并在边缘设备上加速推理，但可能会有轻微的精度损失。

**`dynamic`** (`bool`, 默认值：`False`):

-   是否允许动态输入尺寸（适用于 ONNX、TensorRT 和 OpenVINO 导出）。
-   设置为 `True` 时，导出的模型可以处理不同尺寸的输入图像，增加灵活性。

**`simplify`** (`bool`, 默认值：`True`):

-   在导出为 ONNX 时，是否使用 `onnx-simplifier` 简化模型图。
-   设置为 `True` 可提升模型的性能和兼容性。

**`opset`** (`int`, 默认值：`None`):

-   指定 ONNX 的 opset 版本，以确保与不同的 ONNX 解析器和运行时兼容。
-   如果未设置，将使用支持的最新版本。

**`workspace`** (`float` 或 `None`, 默认值：`None`): - 设置 TensorRT 优化时的最大工作空间大小（以 GiB 为单位）。 - 用于平衡内存使用和性能；设置为 `None` 时，TensorRT 将自动分配最大可用内存。

使用 YOLO 自带的导出

```python
model.export(format='onnx',int8=True,imgsz=736,dynamic=True)
```

### `onnx` 模型推理

```python
# from ultralytics import YOLO
# model = YOLO('runs/detect/train3/weights/best.pt')
# success = model.export(format="onnx", simplify=True)
# assert success
# print("转换成功")

import onnxruntime
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms


def std_output(pred):
    pred = np.squeeze(pred)
    pred = np.transpose(pred, (1, 0))
    pred_class = pred[..., 4:]
    pred_conf = np.max(pred_class, axis=-1)
    pred = np.insert(pred, 4, pred_conf, axis=-1)
    return pred


def nms(detections, iou_threshold):
    if len(detections) == 0:
        return []

    # 按置信度排序
    detections = detections[detections[:, 4].argsort()[::-1]]

    keep = []
    while len(detections) > 0:
        # 取置信度最高的框
        highest = detections[0]
        keep.append(highest)

        if len(detections) == 1:
            break

        # 计算剩余框与最高置信度框的IOU
        rest_boxes = detections[1:]
        iou = compute_iou(highest, rest_boxes)

        # 保留IOU小于阈值的框
        detections = rest_boxes[iou < iou_threshold]
    return np.array(keep)

# 进行推理
outputs = ort_session.run(None, {'images': input_data})  # 更新输入名称为模型期望的名称

# 获取第一个输出数组
output_array = outputs[0]

# 解析检测结果
num_detections = output_array.shape[2]

# 定义置信度阈值
confidence_threshold = 0.1
new_array = std_output(output_array)

# 过滤低置信度的检测框
filtered_detections = new_array[new_array[:, 4] > confidence_threshold]

# 进行非极大值抑制
nms_detections = nms(filtered_detections, iou_threshold=0.2)

# 打开原始图像
image = Image.open(image_path)

# 创建绘图对象
draw = ImageDraw.Draw(image)

# 绘制经过NMS处理后的检测框
for detection in nms_detections:
    bbox = detection[:4]
    x0 = (bbox[0] - bbox[2] / 2) * scale_x
    y0 = (bbox[1] - bbox[3] / 2) * scale_y
    x1 = (bbox[0] + bbox[2] / 2) * scale_x
    y1 = (bbox[1] + bbox[3] / 2) * scale_y

    # 绘制边界框
    draw.rectangle([(x0, y0), (x1, y1)], outline="red")

# 显示图像
image.show()
```

![img_3.png](https://i-blog.csdnimg.cn/img_convert/f83fccc81b4313048f7b2648eb42acbf.png)

1.  **加载 ONNX 模型**

```python
onnx_model_path = '/media/hhd/in_for_HHD/data_m/桌面/电动车/Elevator_monitoring/Elevator_monitoring/runs/detect/train50/weights/best.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model_path)
```

**功能**:

-   设定 ONNX 模型文件路径。
-   使用 `onnxruntime.InferenceSession` 加载 ONNX 模型，创建一个推理会话 `ort_session`，用于执行推理。

2.  **加载并预处理图像**

```python
image_path = '/media/hhd/in_for_HHD/data_m/桌面/xiazi/131.jpg'
image = Image.open(image_path)
preprocess = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])
input_data = preprocess(image).unsqueeze(0).numpy()  # 添加批处理维度并转换为NumPy数组
```

**将图像转换为适合模型输入的格式**:

1. **加载图像**:
    - 使用 `PIL.Image.open` 加载图像文件到内存。
2. **图像预处理**:
    - `transforms.Resize((640, 640))`: 将图像调整到模型期望的输入尺寸 (640x640)。
    - `transforms.ToTensor()`: 将图像转换为张量，归一化到 `[0, 1]`。
3. **扩展维度**:
    - `.unsqueeze(0)`: 添加批处理维度 (Batch Size = 1)。
    - `.numpy()`: 转换为 NumPy 数组以匹配 ONNX 模型输入要求。

3) **执行模型推理**

```python
outputs = ort_session.run(None, {'images': input_data})  # 更新输入名称为模型期望的名称
output_array = outputs[0]
```

获得推理结果，`outputs` 是一个列表，`outputs[0]` 是目标检测结果

**功能**:

-   `ort_session.run` 执行推理，传入预处理后的图像数据。
-   `None` 表示返回所有输出张量。
-   `{'images': input_data}` 是输入字典，键为模型期望的输入名称。

4. **解析检测结果**

```python
num_detections = output_array.shape[2]
confidence_threshold = 0.1
new_array = std_output(output_array)
```

**步骤**:

1. **检测结果的维度信息**:
    - `output_array.shape[2]` 获取检测框数量。
2. **置信度阈值**:
    - `confidence_threshold` 定义最低置信度，过滤掉低置信度的框。
3. **数据标准化**:

    - `std_output(output_array)` 假定是标准化处理函数，将模型输出数据转换为标准格式（例如：[x, y, w, h, confidence, class_id]）。

4. **过滤低置信度的检测框**

```python
filtered_detections = new_array[new_array[:, 4] > confidence_threshold]
```

减少不可靠的检测框，提高检测结果的质量
**功能**:

-   从 `new_array` 中筛选置信度大于阈值的检测框。
-   `[:, 4]` 提取置信度列。

6. **非极大值抑制 (NMS)**

```python
nms_detections = nms(filtered_detections, iou_threshold=0.2)
```

保留最具代表性的检测框，减少冗余

**功能**:

-   对筛选后的检测框应用非极大值抑制，移除高度重叠的框。
-   `iou_threshold` 控制框之间的重叠阈值。

7. **绘制检测框**

```python
image = Image.open(image_path)
draw = ImageDraw.Draw(image)
original_image_size = image.size
feature_map_size = (640, 640)
scale_x = original_image_size[0] / feature_map_size[0]
scale_y = original_image_size[1] / feature_map_size[1]

# 类别标签列表（手动定义）
labels = ['Electric vehicle','bicycle' ]  # 示例类别，按模型的类别顺序定义

    # 计算边界框坐标
    x0 = (bbox[0] - bbox[2] / 2) * scale_x
    y0 = (bbox[1] - bbox[3] / 2) * scale_y
    x1 = (bbox[0] + bbox[2] / 2) * scale_x
    y1 = (bbox[1] + bbox[3] / 2) * scale_y

    # 绘制边界框
    draw.rectangle([(x0, y0), (x1, y1)], outline="red")

    # 绘制类别和置信度
    text = f"{class_name} ({confidence:.2f})"
    text_position = (x0, y0 - 10)  # 文本位置（框上方）
    draw.text(text_position, text, fill="blue")  # 用蓝色绘制文本

```

**功能**:

1. **比例计算**:
    - 根据原始图像尺寸和模型输入尺寸计算比例 (`scale_x`, `scale_y`)。
2. **边界框绘制**:
    - 从 `nms_detections` 获取框坐标 `[x_center, y_center, width, height]`。
    - 将中心点坐标转为左上角和右下角坐标，按比例映射到原始图像。
    - 用红色矩形框绘制检测结果。

完整项目代码实战操作，请进入“人工智能教学实训平台”[https://www.lswai.com](https://www.lswai.com) 体验完整操作流程。

[![rtx4090.jpeg](https://i-blog.csdnimg.cn/img_convert/1384361f4e24c5600651e7731bcd2860.jpeg)](https://www.lswai.com)
