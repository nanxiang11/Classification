# 通用图像分类训练模板 (Universal Image Classification Training Pipeline)

这是一个 **通用的图像分类训练模板**，旨在为研究和工程提供一个高效、可复用的训练框架。模板支持多种主流分类网络，并附带一个树叶分类示例，方便快速上手和二次开发。

---
<img width="2520" height="1035" alt="image" src="https://github.com/user-attachments/assets/10a79276-cce5-48b5-835e-37d44ee67691" />
<img width="2433" height="852" alt="image" src="https://github.com/user-attachments/assets/3c31a63c-1378-45e6-9c80-4efd93940828" />



## 🔹 功能特性

* **通用训练框架**
  支持任意基于 PyTorch 的分类网络，只需更换模型即可训练。

* **多网络支持**
  内置多个主流分类模型示例，包括：

  * LeNet
  * ResNet
  * EfficientNet
  * VGG 系列
    你可以轻松扩展其他自定义模型。

* **自定义数据集加载**
  数据集按照类别名称文件夹自动识别标签，支持：

  * 图像增强 (旋转、翻转、亮度/对比度调节)
  * 自定义输入大小
  * 支持训练/验证分离

* **训练监控与日志**

  * 支持 [SwanLab](https://www.swanlab.com/) 可视化日志
  * 实时记录训练指标（loss、accuracy、precision、recall、F1）
  * 早停机制（EarlyStopping）自动终止训练

* **推理与可视化**

  * 支持随机可视化预测样本
  * 显示真实标签、预测标签及是否正确（√ / ×）
  * 自动计算图像排列行列数，布局整齐

* **树叶分类示例**

  * 使用公开叶片数据集，演示完整训练和推理流程
  * 可直接复用模板训练其他分类任务
    
<img width="1500" height="1200" alt="image" src="https://github.com/user-attachments/assets/d874289d-7574-4cfc-8d7e-2844169b7f3a" />

---

## 🔹 安装与依赖

```bash
git clone https://github.com/nanxiang11/Classification.git
cd Classification
conda create -n img_class python=3.11
conda activate img_class
pip install -r requirements.txt
```
<img width="480" height="633" alt="image" src="https://github.com/user-attachments/assets/fb8a98aa-bc89-444f-92a9-f10c655979a1" />
自行加入数据集文件夹
数据集链接：
通过网盘分享的文件：LeafClassification.zip
链接: https://pan.baidu.com/s/1XZXVf238K4FYD-b4JI_wuQ?pwd=j1n9 提取码: j1n9

依赖主要包括：

* torch
* torchvision
* albumentations
* timm
* matplotlib
* swanlab 

---


## 🔹 特性与优势

* 模板化训练流程，**支持快速替换模型和数据集**
* 完整训练 → 日志记录 → 可视化 → 推理示例
* 支持多种主流分类网络，可作为科研或项目基础模板
* 树叶分类示例展示真实可用的 end-to-end 流程

