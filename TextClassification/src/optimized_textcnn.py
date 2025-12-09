import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import logging
import time
import json
import os
from common import SentimentDataset, collate_fn, evaluate, train

class OptimizedTextCNN(nn.Module):
    """
    优化版TextCNN - 在原始基础上做小幅有效改进
    """
    def __init__(self, config):
        super(OptimizedTextCNN, self).__init__()

        # 基础参数（与原始保持兼容）
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim if hasattr(config, 'embedding_dim') else 300
        self.num_classes = config.num_classes
        self.dropout = config.dropout

        # 增强参数
        self.filter_sizes = config.filter_sizes if hasattr(config, 'filter_sizes') else [3, 4, 5]
        self.num_channels = config.num_channels if hasattr(config, 'num_channels') else 100

        # 词嵌入层 - 保持与原始相同
        self.embedding = nn.Embedding(self.vocab_size + 1, self.embedding_dim, padding_idx=8019)

        # 改进的卷积层 - 使用不同数量的通道
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.num_channels, (fs, self.embedding_dim))
            for fs in self.filter_sizes
        ])

        # 批归一化 - 提升训练稳定性
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(self.num_channels)
            for _ in self.filter_sizes
        ])

        # 残差连接投影层
        total_channels = len(self.filter_sizes) * self.num_channels
        self.residual_proj = nn.Linear(total_channels, total_channels)

        # 改进的分类器 - 更深但不过于复杂
        self.classifier = nn.Sequential(
            nn.Linear(total_channels, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout * 0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout * 0.3),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        # 词嵌入
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x.unsqueeze(1)     # [batch, 1, seq_len, embed_dim]

        # 多尺度卷积
        conv_outputs = []
        for i, conv in enumerate(self.convs):
            # 卷积
            conv_out = conv(x)
            # 激活
            conv_out = F.relu(conv_out.squeeze(3))  # [batch, channels, seq_len]
            # 批归一化
            conv_out = self.batch_norms[i](conv_out)
            # 最大池化
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            pooled = pooled.squeeze(2)  # [batch, channels]
            conv_outputs.append(pooled)

        # 特征拼接
        x = torch.cat(conv_outputs, 1)  # [batch, total_channels]

        # 残差连接
        residual = self.residual_proj(x)
        x = x + 0.1 * residual  # 轻微的残差连接

        # 分类
        output = self.classifier(x)

        return output

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

def create_optimized_config(vocab_size):
    """创建优化配置"""
    config = {
        'vocab_size': vocab_size,
        'embedding_dim': 300,
        'num_classes': 2,
        'dropout': 0.4,  # 稍微降低dropout
        'filter_sizes': [3, 4, 5, 6],  # 增加一个尺度
        'num_channels': 120,  # 稍微增加通道数
        'lr': 0.0005,
        'num_epoch': 30,  # 增加训练轮数
        'eval_interval': 40,
        'save_path': '../save_model/optimized_textcnn_best.pt',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'log_steps': 40,
        'batch_size': 100  # 适中的batch size
    }
    return config

def train_optimized_model():
    """训练优化版TextCNN"""
    print("=" * 60)
    print("开始训练优化版TextCNN模型")
    print("=" * 60)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optimized_training.log', mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    print("\n1. 加载数据集...")
    # 加载数据
    train_dataset = SentimentDataset('../dataset/train.jsonl', '../dataset/vocab.json')
    val_dataset = SentimentDataset('../dataset/val.jsonl', '../dataset/vocab.json')
    test_dataset = SentimentDataset('../dataset/test.jsonl', '../dataset/vocab.json')

    print(f"   训练集样本数: {len(train_dataset)}")
    print(f"   验证集样本数: {len(val_dataset)}")
    print(f"   测试集样本数: {len(test_dataset)}")

    print("\n2. 创建数据加载器...")
    # DataLoader配置
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=100,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=100, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    print("   数据加载器创建完成")

    print("\n3. 创建模型...")
    # 创建模型
    vocab_size = len(json.load(open('../dataset/vocab.json', 'r', encoding='utf-8')))
    config_dict = create_optimized_config(vocab_size)
    config = namedtuple('config', config_dict.keys())(**config_dict)

    device = config.device
    model = OptimizedTextCNN(config).to(device)

    print(f"   词汇表大小: {vocab_size}")
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   使用设备: {device}")
    print(f"   嵌入维度: {config.embedding_dim}")
    print(f"   卷积核大小: {config.filter_sizes}")
    print(f"   通道数: {config.num_channels}")
    print(f"   学习率: {config.lr}")
    print(f"   Batch Size: {config.batch_size}")

    # 显示CUDA信息
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    print("\n4. 开始训练...")
    print("-" * 60)
    logging.info("=== 优化版TextCNN训练开始 ===")

    # 使用train函数进行训练（与common.py兼容）
    start_time = time.time()
    train_loss_history, val_acc_history = train(model, config, train_loader, val_loader)
    end_time = time.time()

    print("-" * 60)
    print(f"\n训练完成！")
    print(f"总训练时间: {end_time - start_time:.2f}秒 ({(end_time - start_time)/60:.1f}分钟)")
    print(f"最佳验证准确率: {max(val_acc_history):.4f}")

    print("\n5. 在测试集上评估...")
    print("-" * 60)

    # 加载最佳模型
    if os.path.exists(config.save_path):
        print(f"加载最佳模型: {config.save_path}")
        checkpoint = torch.load(config.save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        # 评估测试集
        test_acc = evaluate(model, test_loader, config)
        print(f"\n测试集准确率: {test_acc:.4f}")

        # 与原始TextCNN比较
        baseline_acc = 0.8827  # 原始TextCNN的88.27%
        improvement = test_acc - baseline_acc
        print(f"\n性能对比:")
        print(f"   原始TextCNN: {baseline_acc:.4f}")
        print(f"   优化版TextCNN: {test_acc:.4f}")
        print(f"   性能提升: {improvement:+.4f} ({improvement/baseline_acc*100:+.2f}%)")

        if test_acc > baseline_acc:
            print("\n✅ 成功提升性能！")
        else:
            print("\n⚠️  性能未能超越原始模型")

        return test_acc
    else:
        print(f"错误：未找到保存的模型文件 {config.save_path}")
        return 0.0

if __name__ == '__main__':
    # 确保保存目录存在
    os.makedirs('save_model', exist_ok=True)

    # 运行训练
    test_acc = train_optimized_model()

    print(f'\n优化版TextCNN最终测试准确率: {test_acc:.4f}')