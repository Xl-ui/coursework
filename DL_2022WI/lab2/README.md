# Machine Learning Lab 2: MNIST Classification

本实验使用 PyTorch 完成 MNIST 手写数字分类，并比较不同模型结构和训练参数。

## 完成内容

- 构建全连接神经网络基线模型。
- 实现统一的训练、验证、测试和结果可视化流程。
- 比较 batch size、网络深度、隐藏层宽度、激活函数和 Dropout。
- 根据验证集结果选择模型，并在测试集上评估。
- 实现并比较 CNN、简化版ResNet-34 和 ResNet-50 模型。

## 主要文件

- `ml_lab2_pytorch_mnist.ipynb`：模型实现、调参实验、训练曲线与结果分析。
- `Res34_50.ipynb` :将其中使用的Res34与Res50(简化版本)代码独立出来