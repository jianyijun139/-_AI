# YOLOv11抽烟检测模型教学文档

-   数据集制作
    -   读取数据集
        -   读取数据集配置文件
        -   读取标签文件计算标签
    -   数据集清洗
        -   清空数据集内的错误标签
        -   删除空标签文件和对应图片
        -   校验图片格式是否可以被读取
-   开始训练
    -   训练参数
    -   训练
    -   训练效果
-   模型部署
    -   使用 `pose` 模型预推理人物,剪裁后使用 `smoke` 模型推理
    -   推理效果
-   总结

<!-- TOC -->

-   [抽烟检测模型课程](#抽烟检测模型课程)
    -   [数据集制作](#数据集制作)
        _ [读取数据集](#读取数据集)
        _ [读取数据集配置文件](#读取数据集配置文件)
        _ [获取图片数量](#获取图片数量-)
        _ [读取标签文件计算标签](#读取标签文件计算标签)
        _ [数据清理](#数据清理)
        _ [剔除错误标签](#剔除错误标签)
        _ [开始训练](#开始训练)
        _ [#训练参数](#训练参数)
        _ [训练](#训练)
        _ [模型部署](#模型部署) \* [使用 `pose` 模型预推理人物,剪裁后使用 `smoke` 模型推理](#使用-pose-模型预推理人物剪裁后使用-smoke-模型推理)
        <!-- TOC -->

完整项目代码实战操作，请进入“人工智能教学实训平台”[https://www.lswai.com](https://www.lswai.com) 体验完整操作流程。

[![rtx4090.jpeg](https://i-blog.csdnimg.cn/img_convert/b86d2e60f80c8caa0e2a954f285936c9.jpeg)](https://www.lswai.com)


## 数据集制作

### 读取数据集

#### 读取数据集配置文件

```python
dataset_path = '/root/data_yaml/smoke1s2k'

# 设置 YAML 文件的路径
yaml_file_path = os.path.join(dataset_path, 'data.yaml')

# 加载并打印 YAML 文件的内容
with open(yaml_file_path, 'r') as file:
    yaml_content = yaml.load(file, Loader=yaml.FullLoader)
    print(yaml.dump(yaml_content, default_flow_style=False))


out:
  names:
  - cigarette
  - Smoke
  nc: 2
  test: ../test/images
  train: ../train/images
  val: ../valid/images
```

验证配置文件是否正确，如果正确则可以进行下一步操作。

#### 获取图片数量

```python
# 设置训练和验证图像集的路径
train_images_path = os.path.join(dataset_path, 'train', 'images')
valid_images_path = os.path.join(dataset_path, 'valid', 'images')

# 初始化图像数量计数器
num_train_images = 0
num_valid_images = 0

# 初始化集合以保存图像的唯一尺寸
train_image_sizes = set()
valid_image_sizes = set()

# 检查训练图像的大小和数量
for filename in os.listdir(train_images_path):
    if filename.endswith('.jpg'):
        num_train_images += 1
        image_path = os.path.join(train_images_path, filename)
        with Image.open(image_path) as img:
            train_image_sizes.add(img.size)

# 检查验证图像大小和数量
for filename in os.listdir(valid_images_path):
    if filename.endswith('.jpg'):
        num_valid_images += 1
        image_path = os.path.join(valid_images_path, filename)
        with Image.open(image_path) as img:
            valid_image_sizes.add(img.size)

# 打印结果
print(f"Number of training images: {num_train_images}")
print(f"Number of validation images: {num_valid_images}")

# 检查训练集中的所有图像是否具有相同的大小
if len(train_image_sizes) == 1:
    print(f"All training images have the same size: {train_image_sizes.pop()}")
else:
    print("Training images have varying sizes.")

# 检查验证集中的所有图像是否具有相同的大小
if len(valid_image_sizes) == 1:
    print(f"All validation images have the same size: {valid_image_sizes.pop()}")
else:
    print("Validation images have varying sizes.")

out:

Number of training images: 27934
Number of validation images: 2160
Training images have varying sizes.
Validation images have varying sizes.

```

验证图片数量和图片尺寸是否正确。 在 yolo 中图片尺寸可以不同，但是在 yolo 训练中图片尺寸必须是 32 的倍数。

#### 读取标签文件计算标签

```python
from collections import Counter
from pathlib import Path
import yaml

# 加载数据集配置文件
with open(yaml_file_path, 'r') as file:
    data_config = yaml.safe_load(file)

# 定义统计标签的函数
def count_labels(data_path):
    print('counting labels')
    label_counts = Counter()
    image_paths = list(Path(data_path).rglob('*.txt'))  # 假设图片是jpg格式
    print(len(image_paths))
    for image_path in image_paths:
        label_path = image_path.with_suffix('.txt')
        if label_path.exists():
            # print(label_path)
            with open(label_path, 'r') as file:
                labels = file.readlines()
                for label in labels:
                    label_class = int(label.split()[0])
                    label_counts[label_class] += 1
    return label_counts

print('data_config labels',data_config)
# print('data_config yaml_file_path+data_config',dataset_path+"/test")
# 统计每个数据集（训练、验证、测试）的标签
train_counts = count_labels(dataset_path+"/test/labels")
val_counts = count_labels(dataset_path+"/valid/labels")
test_counts = count_labels(dataset_path+"/train/labels")
print(train_counts,val_counts ,test_counts)
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

data_config labels {'train': '../train/images', 'val': '../valid/images', 'test': '../test/images', 'nc': 2, 'names': ['cigarette', 'Smoke']}
counting labels
1429
counting labels
2160
counting labels
27934
Counter({0: 1884, 1: 686}) Counter({0: 1456, 1: 1450}) Counter({0: 23803, 1: 11223})
total_counts Counter({0: 27143, 1: 13359})
类别 cigarette 有 27143 个标签
类别 Smoke 有 13359 个标签

```

可以看到我们有 27143 个标签，其中有 27143 个标签是香烟的, 13359 个标签是烟雾的。但是这些烟雾的标签是错误的，我们需要进行清洗。

### 数据清理

#### 剔除错误标签

首先我们需要先清理掉全部的 1 号标签。

```python
import os

def process_txt_files(folder_path):
    # 获取文件夹内的所有 txt 文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # 遍历每个 txt 文件
    for txt_file in txt_files:
        txt_path = os.path.join(folder_path, txt_file)

        with open(txt_path, 'r') as file:
            lines = file.readlines()

        # 处理每一行
        modified_lines = []
        for line in lines:
            if line and line[0] == '1':  # 如果首字符是数字 2
                continue  # 跳过该行
            modified_lines.append(line)  # 否则将此行保留

        # 将修改后的内容写回文件
        with open(txt_path, 'w') as file:
            file.writelines(modified_lines)
# 调用函数，传入文件夹路径
folder_path = f'{dataset_path}/train/labels'  # 修改为你的文件夹路径
process_txt_files(folder_path)
folder_path = f'{dataset_path}/valid/labels'  # 修改为你的文件夹路径
process_txt_files(folder_path)
folder_path = f'{dataset_path}/test/labels'  # 修改为你的文件夹路径
process_txt_files(folder_path)

```

随后我们需要将空的标签文件和对应的图片删除。

```python


import os
def delete_empty_txt_and_images(valid_folder, image_folder):
    """
    删除  文件夹中空的 .txt 文件，同时删除指定 image_folder 文件夹内同名的 .jpg 文件。

    Args:
        valid_folder (str): 存放 .txt 文件的文件夹路径。
        image_folder (str): 存放 .jpg 文件的文件夹路径。
    """
    # 遍历 valid 文件夹中的所有文件
    for txt_file in os.listdir(valid_folder):
        txt_path = os.path.join(valid_folder, txt_file)

        # 检查文件是否是 .txt 文件
        if txt_file.endswith('.txt'):
            # 检查 .txt 文件是否为空
            if os.path.getsize(txt_path) == 0:
                print(f"发现空的 .txt 文件: {txt_file}")

                # 删除空的 .txt 文件
                os.remove(txt_path)
                print(f"已删除: {txt_path}")

                # 构造对应的 .jpg 文件路径
                image_file = os.path.splitext(txt_file)[0] + '.jpg'
                image_path = os.path.join(image_folder, image_file)

                # 如果对应的 .jpg 文件存在，删除它
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"已删除对应的图片文件: {image_path}")

# 示例用法
valid_folder = f"{dataset_path}/train/labels"  # 替换为你的 train 文件夹路径
image_folder = f"{dataset_path}/train/images"  # 替换为你的图片文件夹路径
delete_empty_txt_and_images(valid_folder, image_folder)
# 示例用法
valid_folder = f"{dataset_path}/valid/labels"  # 替换为你的 valid 文件夹路径
image_folder = f"{dataset_path}/valid/images"  # 替换为你的图片文件夹路径
delete_empty_txt_and_images(valid_folder, image_folder)
# 示例用法
valid_folder = f"{dataset_path}/test/labels"  # 替换为你的 test 文件夹路径
image_folder = f"{dataset_path}/test/images"  # 替换为你的图片文件夹路径
delete_empty_txt_and_images(valid_folder, image_folder)






```

再次 读取标签文件计算

```python
输出:
data_config labels {'train': '../train/images', 'val': '../valid/images', 'test': '../test/images', 'nc': 2, 'names': ['cigarette', 'Smoke']}
counting labels
823
counting labels
922
counting labels
17842
Counter({0: 1884}) Counter({0: 1456}) Counter({0: 23803})
total_counts Counter({0: 27143})
类别 cigarette 有 27143 个标签
```

我们可以看到我们已经将全部的 1 号标签剔除了,但是我们的测试集数量有点少，我们需要进行扩充,又因为 yolo 训练时不需要 test 集，所以我们可以将 test 集的图片移动到 valid 集。

```python
输出:
data_config labels {'train': '../train/images', 'val': '../valid/images', 'test': '../test/images', 'nc': 2, 'names': ['cigarette', 'Smoke']}
counting labels
823
counting labels
1745
counting labels
17842
Counter({0: 1884}) Counter({0: 3340}) Counter({0: 23803})
total_counts Counter({0: 29027})
类别 cigarette 有 29027 个标签
```

这时我们已经拥有进 7:1 的训练集和测试集。

```python
# List all jpg images in the directory
image_files = [file for file in os.listdir(train_images_path) if file.endswith('.jpg')]

# Select 8 images at equal intervals
num_images = len(image_files)
selected_images = [image_files[i] for i in range(0, num_images, num_images // 13)]

# Create a 2x4 subplot
fig, axes = plt.subplots(3, 4, figsize=(20, 11))

# Display each of the selected images
for ax, img_file in zip(axes.ravel(), selected_images):
    img_path = os.path.join(train_images_path, img_file)
    image = Image.open(img_path)
    ax.imshow(image)
    ax.axis('off')

plt.suptitle('Sample Images from Training Dataset', fontsize=20)
plt.tight_layout()
plt.show()
```

![img.png](https://i-blog.csdnimg.cn/img_convert/0b2b44f88bb32fff244dda648b2ba442.png)

### 开始训练

### #训练参数

```python
# Train the model on our custom dataset
rdsss=random.randint(0,9900)
print(rdsss)
results = model.train(
    data=yaml_file_path,

    epochs=30,
    batch=20,

    imgsz=640,
    device=0,
    patience=10,
  optimizer='auto',
    lr0=0.005,
    lrf=0.01,
    dropout=0.1,
    seed= rdsss, )

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

#### 训练

```python

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      30/180      14.8G     0.5278       0.37      1.031          8        640: 100%|██████████| 133/133 [00:39<00:00,  3.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 7/7 [00:02<00:00,  2.96it/s]
                   all       1276       1568      0.909      0.783      0.882      0.732


YOLO11n summary (fused): 238 layers, 2,582,542 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 7/7 [00:25<00:00,  3.69s/it]
                   all       1183       1446      0.953      0.845       0.94      0.782
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

### 模型部署

#### 使用 `pose` 模型预推理人物,剪裁后使用 `smoke` 模型推理

```python

import cv2
import torch
from ultralytics import YOLO

sdssss = False


def mask(count, image, yolo_model, second_model):
    # YOLOv8模型进行目标检测
    results = yolo_model(image, imgsz=(1920, 1088), conf=0.7, iou=0.3)
    # print(len(results))
    detections = results[0].boxes  # 获取检测结果中的boxes
    class_names = results[0].names
    # clss = result
    # print("first model:", detections.shape, class_names)
    # 筛选出标签为人、sitting、standing的目标 (假设cls 0 为 person，4为sitting和standing)
    target_classes = [0, 1, 3, 4]  # 替换为你模型中对应的类别索引
    targets = [box for box in detections if int(box.cls) in target_classes]
    # print("targets:", targets)

    height_img, width_img, _ = image.shape

    cls = ['cigarette', 'Smoke']
    for i, target in enumerate(targets):
        # 获取目标区域的坐标
        x_center, y_center, width, height = target.xywh[0]
        # width, height = int(width) + 50, int(height)
        width, height = int(width) + int(width) / 5, int(height)
        # if width > height:
        #     height = width
        # else:
        #     width = height -50
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        # y2 = int(y_center + height / 2)
        y2 = int(y_center + height / 6)
        # clss = str(class_names[int(target.cls.item())])
        clss = "persons"
        clss_conf = f"{clss} {target.conf.item():.2f}"
        # clss_conf= f"{clss} "
        # print("clss:::::::",clss_conf)
        cv2.putText(image, clss_conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, [255, 50, 30], 1)
        cv2.rectangle(image, (x1, y1), (x2, y2), [255, 50, 30], 1)
        # asss=False
        # if asss==False:
        #     continue
        # 裁剪目标区域
        cropped_image = image[y1:y2, x1:x2]
        height_cropped_image, width_cropped_image, _ = cropped_image.shape
        # print(cropped_image.shape, width, height)

        if cropped_image is None or cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
            continue

        second_results = second_model(cropped_image, conf=0.3, imgsz=736, iou=0.3)
        second_detections = second_results[0].boxes  # 获取二次检测结果中的boxes

        # 绘制二次推理结果
        for index_cc in range(0, len(second_detections.conf)):
            print("index_cc++++++++++++++++", index_cc)

            second_x_center, second_y_center, second_width, second_height = second_detections.xywh[index_cc]
            conf_f = second_detections.conf[index_cc]
            # print("conf_f", conf_f)
            second_x1 = int(second_x_center - second_width / 2) + x1
            second_y1 = int(second_y_center - second_height / 2) + y1
            second_x2 = int(second_x_center + second_width / 2) + x1
            second_y2 = int(second_y_center + second_height / 2) + y1

            clss = str(cls[int(second_detections.cls[index_cc].item())])
            print('===============', clss)
            print('===============', int(second_detections.cls[index_cc].item()))

            if clss == 'smok' or clss == 'cigarette':
                rgb = (0, 0, 255)

                clss_conf = f"{clss} {conf_f.item():.2f}"
                # clss_conf = f"{clss} "
                cv2.putText(image, clss_conf, (second_x1, second_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rgb, 2)
                cv2.rectangle(image, (second_x1, second_y1), (second_x2, second_y2), rgb, 2)
            # elif clss == 'Smoke':
            else:
                rgb = (36, 255, 12)
                # print("second_detections",second_detections)

                clss_conf = f"{clss} {conf_f.item():.2f}"
                # clss_conf = f"{clss} "
                cv2.putText(image, clss_conf, (second_x1, second_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rgb, 2)
                cv2.rectangle(image, (second_x1, second_y1), (second_x2, second_y2), rgb, 2)

    # 保存或显示最终图像
    # output_path = 'path_to_save_result.jpg'
    # cv2.imwrite(output_path, image)
    return image


# 定义自定义操作函数
def process_frame(count, frame, yolo_model, second_model):
    processed_frame = mask(count, frame, yolo_model, second_model)
    return processed_frame


import cv2


# 定义自定义操作函数
def process_frame(count, frame, yolo_model, second_model):
    processed_frame = mask(count, frame, yolo_model, second_model)
    return processed_frame


cap = cv2.VideoCapture('/home/hhd/桌面/elevator_all/WeChat_time_10_m-Trim.mp4')

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
# 获取视频的帧宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义输出视频编写器
# 注意，如果处理后的帧是灰度图像，则 isColor 参数应设置为 False
out = cv2.VideoWriter('output_videos_13_2_now.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height),
                      isColor=True)

# 初始化 YOLO 模型
yolo_model = YOLO('/home/hhd/桌面/elevator_all/yolov8x-pose.pt')
second_model = YOLO('/home/hhd/桌面/elevator_all/datasets/smoke1s2k/best.pt')
frame_count = 0

while True:
    # 逐帧读取视频
    ret, frame = cap.read()

    # 如果帧读取成功
    if ret:
        print(frame_count)
        # if frame_count < 6140:
        # frame_count += 1
        # continue

        # 对帧进行处理
        processed_frame = process_frame(frame_count, frame, yolo_model, second_model)

        # 写入处理后的帧到输出视频
        out.write(processed_frame)

        frame_count += 1
        if frame_count > 600:
            break
    else:
        break

# 释放视频捕获器和编写器
cap.release()
out.release()

# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()

```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/25db8ed352e147adb9ec0a9ce5109513.png#pic_center)


1. **YOLO 模型加载和推理的性能开销**

    - **细节**：YOLO 模型的加载和推理是计算密集型的任务。每次调用 `yolo_model(image)` 时，都会进行一次完整的模型推理。在视频的每一帧中，都要进行两次推理（一次 YOLOv8 检测和一次烟雾检测模型），如果视频较长或帧率较高，可能会导致性能瓶颈，特别是在没有高性能硬件支持的情况下。
    - **建议**：可以考虑优化推理过程，例如使用更轻量级的模型（如 YOLO Nano 或 YOLOv5）或通过批量推理减少推理次数。

2. **目标区域裁剪（裁剪大小与位置）**

    - **细节**：在处理 YOLOv8 检测到的目标时，裁剪区域的大小是通过 `width, height` 的调整来决定的，这里直接对宽度和高度加上了一个固定的比例（`width = int(width) + int(width) / 5` 和 `height = int(height)`）。这可能导致某些目标框的裁剪区域过大或过小，尤其是当目标与其他对象相邻时。
    - **建议**：根据目标的特性，裁剪区域的尺寸可以动态调整，例如依据目标的类别、置信度等信息来决定裁剪区域的大小，避免裁剪区域太大或太小导致二次检测的效果不理想。

3. **YOLO 输出的处理**

    - **细节**：在 YOLO 的输出中，`results[0].boxes` 返回的是 `boxes` 的坐标信息，其中 `xywh` 是目标框的中心坐标和宽高，`cls` 是类别标签，`conf` 是置信度。在代码中，有些地方直接使用 `int()` 对坐标和宽高进行了转换，但需要注意的是，这种处理可能会导致某些数据精度的丢失，特别是在高分辨率图片中。
    - **建议**：确保在使用框坐标时进行适当的类型转换和四舍五入处理（比如：`int()`），同时在绘制框时也应注意坐标是否溢出图像范围。

4. **裁剪图像为空的判断**
    - **细节**：在裁剪目标区域后，代码中对 `cropped_image` 是否为空进行了判断 `if cropped_image is None or cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:`，防止图像为空或无效。但这段代码有些保守，实际情况可能不需要处理所有为零或 `None` 的情况，尤其是在图片尺寸较大时，这种判断可能是过多的计算。
    - **建议**：可以在裁剪区域的生成阶段就加上更细致的检查，比如仅当目标框的尺寸过小（小于某个阈值）时才跳过处理，这样避免不必要的判断和性能损失。

5) **视频帧率和输出视频帧大小**
    - **细节**：代码中获取视频的帧率和帧宽度 `fps = cap.get(cv2.CAP_PROP_FPS)` 和 `frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))`，然后将这些信息用于输出视频的配置。然而，输出视频的帧率和分辨率是否符合需求并未经过校验，可能会出现视频处理过程中丢帧或输出视频画面不一致的情况。
    - **建议**：可以根据实际需求调整输出视频的帧率和分辨率，确保处理的视频与输出的一致性。

完整项目代码实战操作，请进入“人工智能教学实训平台”[https://www.lswai.com](https://www.lswai.com) 体验完整操作流程。

[![rtx4090.jpeg](https://i-blog.csdnimg.cn/img_convert/65da1a9132aab51659e992634ef79d29.jpeg)](https://www.lswai.com)
