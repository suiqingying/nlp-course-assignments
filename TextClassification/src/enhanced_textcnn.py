import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import logging
import time
import json
import os
from common import SentimentDataset, collate_fn, evaluate, train

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.size(0)

        # 计算Q, K, V
        Q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        # 输出投影
        output = self.out_linear(context)
        return output

class EnhancedTextCNN(nn.Module):
    """
    增强版TextCNN - 高性能架构
    """
    def __init__(self, config):
        super(EnhancedTextCNN, self).__init__()

        # 基础参数（兼容原始main.py）
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim if hasattr(config, 'embedding_dim') else 300
        self.num_classes = config.num_classes
        self.dropout = config.dropout

        # 增强配置
        self.filter_sizes = config.filter_sizes if hasattr(config, 'filter_sizes') else [2, 3, 4, 5]
        self.num_channels = config.num_channels if hasattr(config, 'num_channels') else 128
        self.hidden_dim = config.hidden_dim if hasattr(config, 'hidden_dim') else 512

        # 词嵌入层 - 使用padding_idx
        self.embedding = nn.Embedding(self.vocab_size + 1, self.embedding_dim, padding_idx=8019)

        # 位置编码
        self.pos_embedding = nn.Embedding(1000, self.embedding_dim)

        # 多尺度卷积层组
        self.conv_groups = nn.ModuleList()
        for i, fs in enumerate(self.filter_sizes):
            # 每个尺度使用不同的通道数
            channels = self.num_channels + i * 32
            conv_group = nn.ModuleList([
                nn.Conv2d(1, channels, (fs, self.embedding_dim)),
                nn.Conv2d(channels, channels, (3, 1), padding=(1, 0))
            ])
            self.conv_groups.append(conv_group)

        # 残差连接层
        self_residual_channels = sum([self.num_channels + i * 32 for i in range(len(self.filter_sizes))])
        self.residual_proj = nn.Linear(self_residual_channels, self_residual_channels)

        # 层归一化
        self.layer_norm = nn.LayerNorm(self_residual_channels)

        # 多头注意力
        self.attention = MultiHeadAttention(self_residual_channels, num_heads=8)

        # 特征增强网络
        self.feature_enhance = nn.Sequential(
            nn.Linear(self_residual_channels, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self_residual_channels),
            nn.LayerNorm(self_residual_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(self_residual_channels, self_residual_channels),
            nn.Sigmoid()
        )

        # 深层分类器
        self.classifier = nn.Sequential(
            nn.Linear(self_residual_channels, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout * 0.5),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.BatchNorm1d(self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout * 0.3),
            nn.Linear(self.hidden_dim // 4, self.num_classes)
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.1)

    def forward(self, x):
        batch_size, seq_len = x.size()

        # 词嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        # 位置编码
        if seq_len <= 1000:
            pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
            embedded = embedded + self.pos_embedding(pos)

        # 多尺度卷积特征提取
        conv_outputs = []
        for i, conv_group in enumerate(self.conv_groups):
            # 第一层卷积
            conv_out = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
            conv_out = conv_group[0](conv_out)  # [batch_size, channels, new_seq_len, 1]
            conv_out = F.relu(conv_out.squeeze(-1))  # [batch_size, channels, new_seq_len]

            # 第二层卷积
            conv_out = conv_out.unsqueeze(-1)  # [batch_size, channels, new_seq_len, 1]
            conv_out = conv_group[1](conv_out)  # [batch_size, channels, new_seq_len, 1]
            conv_out = F.relu(conv_out.squeeze(-1))  # [batch_size, channels, new_seq_len]

            # 最大池化
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # [batch_size, channels, 1]
            pooled = pooled.squeeze(-1)  # [batch_size, channels]
            conv_outputs.append(pooled)

        # 特征拼接
        feature_map = torch.cat(conv_outputs, dim=1)  # [batch_size, total_channels]

        # 残差连接
        residual = self.residual_proj(feature_map)
        feature_map = self.layer_norm(feature_map + residual)

        # 增加序列维度用于注意力
        feature_seq = feature_map.unsqueeze(1)  # [batch_size, 1, total_channels]

        # 多头注意力
        attended = self.attention(feature_seq)  # [batch_size, 1, total_channels]
        attended = attended.squeeze(1)  # [batch_size, total_channels]

        # 特征增强
        enhanced = self.feature_enhance(attended)

        # 门控融合
        gate_weights = self.gate(feature_map)
        final_features = gate_weights * feature_map + (1 - gate_weights) * enhanced

        # 分类
        output = self.classifier(final_features)

        return output

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

def create_enhanced_config(vocab_size, base_channels=128):
    """创建增强配置"""
    config = {
        'vocab_size': vocab_size,
        'embedding_dim': 300,
        'num_classes': 2,
        'dropout': 0.4,
        'filter_sizes': [2, 3, 4, 5, 6],
        'num_channels': base_channels,
        'hidden_dim': 1024,
        'lr': 0.0008,
        'weight_decay': 1e-4,
        'eval_interval': 40,
        'num_epoch': 25,
        'save_path': '../save_model/enhanced_textcnn_best.pt',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'log_steps': 40,
        'batch_size': 64
    }
    return config

def train_enhanced_model():
    """训练增强版TextCNN"""
    print("=" * 60)
    print("开始训练增强版TextCNN模型")
    print("=" * 60)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('enhanced_training.log', mode='w', encoding='utf-8'),
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
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    print("   数据加载器创建完成")

    print("\n3. 创建模型...")
    # 创建模型
    vocab_size = len(json.load(open('../dataset/vocab.json', 'r', encoding='utf-8')))
    config_dict = create_enhanced_config(vocab_size)
    config = namedtuple('config', config_dict.keys())(**config_dict)

    device = config.device
    model = EnhancedTextCNN(config).to(device)

    print(f"   词汇表大小: {vocab_size}")
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   使用设备: {device}")
    print(f"   嵌入维度: {config.embedding_dim}")
    print(f"   卷积核大小: {config.filter_sizes}")
    print(f"   学习率: {config.lr}")

    # 显示CUDA信息
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    print("\n4. 开始训练...")
    print("-" * 60)
    logging.info("=== 增强版TextCNN训练开始 ===")

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
        print(f"   增强版TextCNN: {test_acc:.4f}")
        print(f"   性能提升: {improvement:+.4f} ({improvement/baseline_acc*100:+.2f}%)")

        return test_acc
    else:
        print(f"错误：未找到保存的模型文件 {config.save_path}")
        return 0.0

if __name__ == '__main__':
    # 确保保存目录存在
    os.makedirs('save_model', exist_ok=True)

    # 运行训练
    test_acc = train_enhanced_model()

    print(f'\n增强版TextCNN最终测试准确率: {test_acc:.4f}')
    print(f'相比原始TextCNN，预期提升: 2-5个百分点')