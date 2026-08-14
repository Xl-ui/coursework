# Assignment 6: Variational Autoencoders

本作业基于 MNIST 实现 Variational Autoencoder（VAE）和 Conditional VAE（CVAE）。

## 完成内容

- 构建全连接 VAE 的编码器与解码器。
- 实现重参数化技巧。
- 实现由重建误差和 KL 散度组成的损失函数。
- 训练 VAE 并进行随机生成和潜空间插值。
- 在类别标签条件下训练 CVAE，实现指定数字类别的生成。

## 主要文件

- `variational_autoencoders.ipynb`：实验过程与可视化结果。
- `vae.py`：VAE、CVAE 和损失函数实现。
- `vae_generation.jpg`：VAE 生成结果。
- `conditional_vae_generation.jpg`：CVAE 条件生成结果。
- `eecs598/`：课程提供的辅助代码。