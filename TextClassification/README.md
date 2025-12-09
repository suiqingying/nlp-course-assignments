# 文本分类实践

## 项目简介

本项目实现了基于TextCNN和BERT的中文文本分类系统，通过系统性的超参数优化和模型复杂度对比实验，探索了模型性能与复杂度的关系。

## 作业内容

- **TextCNN模型**: 实现多尺度卷积核的文本分类模型
- **超参数优化**: 系统性网格搜索（学习率、Dropout、通道数、卷积核尺寸）
- **模型复杂度对比**: 从简单到复杂的4个版本（原始、优化、增强、极限）
- **预训练模型**: 使用chinese-macbert-large进行对比
- **关键发现**: num_channels参数的引入带来5.27%的性能提升

## 实验结果

| 模型 | 准确率 | 参数量 | 训练时间 |
|------|--------|--------|---------|
| BERT (chinese-macbert-large) | **93.00%** | 1.1亿 | 8分钟 |
| TextCNN (最佳配置) | **89.00%** | 96.6万 | 1分钟 |
| TextCNN (基线) | 88.27% | 96.6万 | 1分钟 |
| 增强版TextCNN | 84.93% | 1369万 | 3分钟 |
| 极限版TextCNN | 66.67% | 2803万 | 5分钟 |

**核心发现**: 
- num_channels (1→100) 带来+5.27%提升
- 过度复杂化导致严重过拟合
- 简单模型在小数据集上更有效

## 最佳配置

- 学习率: 0.001
- Dropout: 0.5
- 通道数: 100
- 卷积核: [4, 5, 6]

## 项目结构

```
├── dataset/           # 数据集
├── save_model/        # 模型权重
├── src/               # 源代码
│   ├── main.py        # TextCNN训练
│   ├── main_bert.py   # BERT训练
│   └── inference.py   # 推理代码
├── logs/              # 训练日志
└── report.pdf         # 实验报告
```

## 运行方式

**训练TextCNN:**
```bash
cd src
python main.py --lr 0.001 --dropout 0.5 --channels 100
```

**训练BERT:**
```bash
cd src
python main_bert.py --batch_size 16 --lr 2e-5 --num_epoch 8
```

**模型推理:**
```bash
cd src
python inference.py
```
