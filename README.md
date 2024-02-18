# Multimodal Emotion Recognition

本仓库存放了多模态实验任务的代码，结合文本与图像的不同表示方法以及不同特征融合方法构建了多模态情感识别模型。

## Setup

你可以运行如下代码来安装所需依赖：

```shell
pip install -r requirements.txt
```

## Repository structure

```
|-- data # 存放了项目所需数据
    |-- data/ # 存放了原始的图像和文本
    |-- train.txt  # 原始训练数据
    |-- test_without_label.txt  # 原始测试数据
    |-- text_tag_train.csv # 使用notebook/process_data.ipynb整合后的训练数据
    |-- text_test.csv # 使用notebook/process_data.ipynb整合后的测试数据
|-- model # 模型定义
|-- notebook # 实验过程中用到的ipynb文件
    |-- process_data # 数据预处理的ipynb文件
|-- prediction # 每种模型的预测结果
    |-- BertResEarly # BertResEarly最佳模型的预测结果
|-- saved_model # 训练过程中保存的每种模型的最佳结果
    |-- BertResEarly # BertResEarly的最佳模型
|-- report # 辅助工具
    |-- report.docx # 本次项目的报告
|-- utils # 辅助工具
    |-- generate_dataset.py # 生成训练所需的Dataloader
    |-- train.py # 与训练有关的函数
|-- main.py # 模型训练评估及预测
|-- requiremens.txt # 需要的依赖
|-- test_without_label.txt # 预测后的测试集
```

## Run pipeline

直接python main.py即可，可选参数如下：

- text_encoder：指定文本的表示学习模型，可选项为bert和xlnet，默认为bert
- image_encoder：指定图像的表示学习模型，可选项为resnet和mobilenet，默认为resnet
- fusion_method：指定特征融合的方式，可选项为early，late和tensor，默认为early
- image_only：是否进行仅图像的消融实验
- text_only：是否进行仅文本的消融实验
- lr：指定学习率，默认为1e-5
- epoch：指定epoch数，默认为20

示例：

```shell
python main.py
```

使用BertResEarly模型进行训练，训练过程中的学习率为1e-5，进行20个epoch

```
python main.py --image_only
```

进行仅图像的消融实验，训练过程中的学习率为1e-5，进行20个epoch

## Attribution

[赛尔笔记 | 多模态情感分析简述 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/97170240?utm_psn=1734246687248314368)

Zhao S, Jia G, Yang J, et al. Emotion recognition from multiple modalities: Fundamentals and methodologies[J]. IEEE Signal Processing Magazine, 2021, 38(6): 59-73.

https://medium.com/haileleol-tibebu/data-fusion-78e68e65b2d1
