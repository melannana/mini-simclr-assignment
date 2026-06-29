# Mini-SimCLR 图像表征学习复现实验报告

## 1. 论文信息

- **论文名称：** A Simple Framework for Contrastive Learning of Visual Representations
- **作者：** Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
- **论文地址：** https://arxiv.org/abs/2002.05709
- **官方代码参考：** https://github.com/google-research/simclr

---

# 2. 任务说明

本实验复现 SimCLR 自监督图像表征学习框架，并在 CIFAR-10 数据集上完成自监督预训练与 Linear Probe 评估。

```text
预训练输入：无标签图像

预训练目标：
使同一张图像经过两次随机增强后的特征更加接近，
不同图像之间的特征更加远离。

评估方式：
冻结 Encoder，
训练 Linear Probe，
最终在 CIFAR-10 Test Set 上计算分类准确率。
```

本实验仅复现论文核心思想，不追求 ImageNet 大规模实验，而是在个人电脑 CPU 环境下完成 Mini-SimCLR 的实现。

---

# 3. 数据集

- 数据集名称：CIFAR-10
- 数据集地址：https://www.cs.toronto.edu/~kriz/cifar.html
- 实际使用预训练图像数：3000
- 实际使用 Linear Probe 训练图像数：3000
- 实际使用测试图像数：10000
- 使用设备：CPU
- 总训练耗时：约（请填写你的实际时间，例如 25 分钟）

---

# 4. 数据增强

实验中采用 SimCLR 推荐的数据增强方式。

| 增强方法 | 参数设置 |
|---|---|
| RandomResizedCrop | 32×32 |
| RandomHorizontalFlip | p=0.5 |
| ColorJitter | Brightness=0.4 Contrast=0.4 Saturation=0.4 Hue=0.1 |
| RandomGrayscale | p=0.2 |
| GaussianBlur | 未使用（32×32 图像影响较小） |

请说明为什么这些增强适合 SimCLR：

```text
SimCLR 的核心思想是通过不同的数据增强构造同一图像的两个 View。

RandomResizedCrop 可以学习尺度不变性；

RandomHorizontalFlip 可以学习左右翻转不变性；

ColorJitter 可以增强颜色鲁棒性；

RandomGrayscale 可以减少颜色依赖。

这些增强共同提高了模型学习判别性表征的能力。
```

---

# 5. 模型结构

Mini-SimCLR 网络结构如下：

```text
Image
        │
        ▼
Two Random Augmented Views
        │
        ▼
Shared Encoder
        │
        ▼
Projection Head
        │
        ▼
L2 Normalize
        │
        ▼
NT-Xent Loss
```

---

## 5.1 Encoder

- encoder 类型：ResNet-18（适配 CIFAR-10）
- 输出特征维度：512
- 是否使用预训练权重：否

---

## 5.2 Projection Head

- MLP 层数：2
- hidden dimension：512
- output dimension：128
- 激活函数：ReLU
- BatchNorm：未使用

---

## 5.3 Linear Probe

- encoder 是否冻结：是
- linear classifier 输入维度：512
- 类别数：10

---

# 6. Loss 实现

实验采用论文中的 NT-Xent Loss。

- batch size：64
- 总样本数：2N = 128
- 正样本：同一张图片两次增强得到的 View
- 负样本：Batch 内其它所有样本
- temperature：0.5
- logits shape：(128,128)

实现步骤：

1. 拼接两个 View 得到 2N 个样本；
2. L2 Normalize；
3. 计算 Pairwise Cosine Similarity；
4. 去除自身相似度；
5. 提取 Positive Pair；
6. 使用 LogSumExp 构造分母；
7. 计算 NT-Xent Loss。

核心公式：

\[
L=-\left(s_{ij}-\log\sum_{k\neq i}e^{s_{ik}}\right)
\]

其中温度参数 τ=0.5。

---

# 7. 训练设置

## 7.1 自监督预训练

| 配置 | 数值 |
|---|---:|
| train images | 3000 |
| epochs | 10 |
| batch size | 64 |
| optimizer | Adam |
| learning rate | 0.001 |
| temperature | 0.5 |
| encoder | ResNet-18 |
| device | CPU |

---

## 7.2 Linear Probe

| 配置 | 数值 |
|---|---:|
| train images | 3000 |
| test images | 10000 |
| epochs | 10 |
| batch size | 64 |
| optimizer | Adam |
| learning rate | 0.001 |
| device | CPU |

---

# 8. 训练过程

Loss 曲线如下：

![](loss_curve.png)

训练 Loss：

| Epoch | Contrastive Loss |
|---|---:|
|1|4.42|
|2|4.22|
|3|4.12|
|4|4.08|
|5|4.03|
|6|3.96|
|7|3.98|
|8|3.93|
|9|3.92|
|10|3.89|

训练分析：

```text
Loss 从约 4.42 持续下降至 3.89，
整体呈下降趋势，仅第 7 个 Epoch 出现轻微波动。

说明 NT-Xent Loss 实现正确，
模型能够逐步学习有效的图像表征，
自监督训练过程稳定。
```

---

# 9. Linear Probe 结果

| 指标 | 结果 |
|---|---:|
| test accuracy | **33.89%** |
| random baseline | 10% |

分析：

```text
Linear Probe Accuracy 达到 33.89%，
明显高于随机猜测（10%）。

说明 Encoder 已经学习到一定的图像语义特征。

由于实验仅使用 3000 张训练图像，
训练 Epoch 较少，
且全部在 CPU 上完成，
因此准确率仍明显低于论文中使用 ImageNet 大规模训练得到的结果。

增加训练数据、提高 Batch Size、
增加 Epoch 或使用 GPU 都有望进一步提升准确率。
```

---

# 10. 预测结果展示

预测样例如下：

![](prediction_samples.png)

| 图片编号 | 真实类别 | 预测类别 | 是否正确 |
|---|---|---|---|
|1|cat|dog|×|
|2|cat|deer|×|
|3|bird|deer|×|
|4|cat|horse|×|
|5|frog|ship|×|

虽然部分预测错误，
但整体准确率已经明显高于随机猜测，
说明模型已经学习到一定的语义表示。

---

# 11. 问题与改进

实验过程中主要遇到以下问题：

1. NT-Xent Loss 正负样本索引容易出错；
2. CPU 训练速度较慢；
3. Batch Size 较小导致负样本数量不足；
4. 训练 Epoch 较少；
5. 部分类别（Cat、Deer）准确率较低。

改进方向：

- 增大 Batch Size；
- 使用完整 CIFAR-10 数据集；
- 增加预训练 Epoch；
- 加入 GaussianBlur；
- 使用 BatchNorm Projection Head；
- 使用 GPU 加速训练；
- 对比不同 Temperature 的影响；
- 可视化 Embedding（t-SNE）。

---

# 12. AI 对话过程记录

- 录制工具：ChatGPT
- 对话链接：(https://chatgpt.com/share/6a40b963-40e8-83ea-9c4f-d21c490ca3f1)
- 使用 AI 模型：ChatGPT GPT-5.5
- 累计对话时长：约 3 小时

AI 主要帮助内容：

```text
1. Mini-SimCLR 项目整体设计；
2. 数据增强实现；
3. ResNet Encoder；
4. Projection Head；
5. NT-Xent Loss；
6. Self-Supervised Pretraining；
7. Linear Probe；
8. Predict/Test 脚本；
9. 实验报告撰写。

所有代码均经过本人运行、调试和验证，
最终完成实验复现。
```

---

# 13. Git 提交记录

- 仓库地址：(https://github.com/melannana/mini-simclr-assignment)
- 总 Commit 数：9

git log --oneline：

```text
feat: load CIFAR-10 and data augmentation
feat: implement ResNet encoder
feat: add projection head
feat: implement NT-Xent Loss
feat: finish SimCLR pretraining
feat: implement linear probe
feat: add test script
feat: add prediction visualization
docs: complete experiment report
```