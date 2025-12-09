# 神经机器翻译实践

## 项目简介

本项目实现了英译中神经机器翻译系统，包含从零训练的Transformer和预训练mBART模型微调两种方案。

## 作业内容

- **Transformer架构**: 从零实现Encoder-Decoder结构，包括多头注意力、位置编码、掩码机制
- **预训练模型**: 使用mBART-50多语言模型进行翻译任务
- **数据处理**: 处理BPE分词格式（`@@`分隔符）
- **评估指标**: BLEU分数计算

## 实验结果

| 模型 | 验证集Loss | BLEU | 训练时间 |
|------|-----------|------|---------|
| mBART-50 (预训练) | 2.90 | **0.51** | 无需训练 |
| Transformer (从零) | 7.93 | 0.03 | ~10分钟 |

**核心发现**: 预训练模型BLEU是从零训练的17倍，充分说明了预训练的重要性。

## 项目结构

```
├── mbart_finetune.py      # mBART微调脚本
├── train_transformer.py   # Transformer从零实现
├── model_best.pth         # 训练好的权重
├── data/                  # 训练/验证/测试数据
└── report.pdf             # 实验报告
```

## 运行方式

**从零训练Transformer:**
```bash
python train_transformer.py --epochs 10 --batch_size 64
```

**mBART预训练模型评估:**
```bash
python mbart_finetune.py --predict_only
```

**交互式翻译:**
```bash
python train_transformer.py --interactive
```

## 硬件要求

- 推荐: NVIDIA RTX 4090 (24GB)
- mBART需要更多显存，建议使用bf16混合精度
