# Assignment 1: PyTorch Basics and k-NN

本目录包含两部分练习：PyTorch 基础操作，以及基于 NumPy 实现的 k-Nearest Neighbors（k-NN）分类实验。

## 完成内容

- 练习张量创建、索引、变形、广播和矩阵运算。
- 使用向量化方法减少 Python 循环，并尝试 GPU 计算。
- 使用双循环、单循环和无循环方法计算样本间的 L2 距离。
- 实现 k-NN 分类预测和交叉验证，并在 CIFAR-10 上选择合适的 k 值。

## 主要文件

- `pytorch101.ipynb`：PyTorch 基础练习与运行结果。
- `pytorch101.py`：PyTorch 练习中的函数实现。
- `knn_numpy.ipynb`：k-NN 实验、可视化和交叉验证。
- `cs231n/classifiers/k_nearest_neighbor.py`：NumPy 版本的 k-NN 实现。
- `cs231n/`、`eecs598/`：数据加载与课程辅助代码。

混杂了cs231n与eecs598两个版本的作业,二者内容是一致的,避免重复列出.
CIFAR-10 原始数据未包含在仓库中。