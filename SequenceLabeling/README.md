# 序列标注：中文命名实体识别

## 项目简介

本项目实现了基于BiLSTM-CRF的中文命名实体识别(NER)系统，并与BERT预训练模型进行对比。

## 作业内容

- **BIO标注体系**: 实现Begin-Inside-Outside标注格式
- **BiLSTM-CRF模型**: 双向LSTM特征提取 + CRF全局约束
- **模型优化**: LayerNorm、梯度裁剪、学习率调度、权重初始化
- **预训练对比**: 使用chinese-macbert-base进行对比实验
- **维特比解码**: 实现CRF层的动态规划解码算法

## 实验结果

| 模型 | 验证集F1 | 测试集F1 | 训练时间 |
|------|----------|----------|---------|
| BiLSTM-CRF (基线) | 88.64% | ~83% | ~30分钟 |
| BiLSTM-CRF (优化) | **90.40%** | 83.73% | ~35分钟 |
| BERT (预训练) | 96.33% | **91.52%** | ~14分钟 |

**核心发现**: 
- 优化后BiLSTM-CRF验证集提升1.76%，但测试集泛化有限
- BERT预训练模型性能显著优于从零训练，差距约8%

## 优化措施

- 隐藏层维度: 300 → 512
- Dropout: 0.3 → 0.5
- 优化器: Adam → AdamW (权重衰减0.01)
- 学习率调度: OneCycleLR
- 梯度裁剪: max_norm=5.0
- CRF Loss: Sum → Mean

## 项目结构

```
├── inference/          # 数据和CRF实现
├── save_model/        # 模型权重
├── logs/              # 训练日志
├── train.py           # BiLSTM-CRF训练
├── train_pretrained.py # BERT训练
├── test.py            # 模型测试
└── report.pdf         # 实验报告
```

## 运行方式

**训练BiLSTM-CRF:**
```bash
python train.py --hidden_dim 512 --dropout 0.5
```

**训练BERT:**
```bash
python train_pretrained.py --batch_size 16 --lr 2e-5
```

**模型测试:**
```bash
python test.py
```
