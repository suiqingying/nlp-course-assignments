import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import logging
import time
import json
import os
from common import SentimentDataset, collate_fn, evaluate

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch, channels, length]
        b, c, _ = x.size()

        # 平均池化分支
        avg_out = self.fc(self.avg_pool(x).view(b, c))

        # 最大池化分支
        max_out = self.fc(self.max_pool(x).view(b, c))

        # 融合
        out = avg_out + max_out
        out = out.view(b, c, 1)
        return x * out.expand_as(x)

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch, channels, length]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention

class DualAttentionBlock(nn.Module):
    """双重注意力块"""
    def __init__(self, channels):
        super(DualAttentionBlock, self).__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        # x shape: [batch, channels, length]
        residual = x

        # 通道注意力
        x = self.channel_att(x)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2) + residual

        # 空间注意力
        residual = x
        x = self.spatial_att(x)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2) + residual

        return x

class UltraTextCNN(nn.Module):
    """
    极限版TextCNN - 最先进架构结合
    """
    def __init__(self, config):
        super(UltraTextCNN, self).__init__()

        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim if hasattr(config, 'embedding_dim') else 400
        self.num_classes = config.num_classes
        self.num_heads = config.num_heads if hasattr(config, 'num_heads') else 8

        # 1. 词嵌入层 - 高维度
        self.embedding = nn.Embedding(self.vocab_size + 1, self.embedding_dim, padding_idx=8019)

        # 2. 位置编码
        self.pos_embedding = nn.Embedding(2048, self.embedding_dim)

        # 3. 段差编码
        self.segment_embedding = nn.Embedding(4, self.embedding_dim)

        # 4. 分层卷积架构
        self.conv_layers = nn.ModuleDict()
        filter_sizes = [2, 3, 4, 5, 7]  # 更多的尺度
        channels_list = [128, 256, 384, 512, 256]

        for i, (fs, ch) in enumerate(zip(filter_sizes, channels_list)):
            # 每层包含多级卷积
            layer_name = f'conv_{i}'
            self.conv_layers[layer_name] = nn.ModuleDict({
                'conv1': nn.Conv2d(1, ch, (fs, self.embedding_dim), padding=(fs//2, 0)),
                'conv2': nn.Conv2d(ch, ch*2, (3, 1), padding=(1, 0)),
                'conv3': nn.Conv2d(ch*2, ch*2, (3, 1), padding=(1, 0)),
                'bn1': nn.BatchNorm2d(ch),
                'bn2': nn.BatchNorm2d(ch*2),
                'bn3': nn.BatchNorm2d(ch*2),
                'attention': DualAttentionBlock(ch*2),
                'dropout': nn.Dropout(0.2 + i*0.05)
            })

        # 5. 密集连接模块
        total_channels = sum(channels_list)*2  # 每层输出通道数翻倍
        self.dense_layers = nn.ModuleList([
            nn.Linear(total_channels, total_channels // 2),
            nn.BatchNorm1d(total_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(total_channels // 2, total_channels // 4),
            nn.BatchNorm1d(total_channels // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(total_channels // 4, total_channels // 8),
            nn.BatchNorm1d(total_channels // 8),
            nn.ReLU(),
            nn.Dropout(0.2)
        ])

        # 6. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=total_channels // 8,
            nhead=self.num_heads,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 7. 自注意力池化
        self.attention_pooling = nn.Sequential(
            nn.Linear(total_channels // 8, total_channels // 16),
            nn.Tanh(),
            nn.Linear(total_channels // 16, 1)
        )

        # 8. 多层分类器
        self.classifier = nn.Sequential(
            nn.Linear(total_channels // 8, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes)
        )

        # 9. 辅助损失头
        self.aux_classifier = nn.Sequential(
            nn.Linear(total_channels // 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
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
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, x):
        batch_size, seq_len = x.size()

        # 1. 词嵌入 + 位置编码
        embedded = self.embedding(x)

        # 位置编码
        if seq_len <= 2048:
            pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
            embedded = embedded + self.pos_embedding(pos)

        # 段差编码（模拟句子的不同部分）
        segment_ids = torch.zeros_like(x)
        segment_ids[:, seq_len//2:] = 1
        if seq_len // 3 < seq_len:
            segment_ids[:, seq_len//3:] = 2
        if seq_len // 4 < seq_len:
            segment_ids[:, seq_len//4:] = 3
        embedded = embedded + self.segment_embedding(segment_ids)

        # 2. 分层卷积特征提取
        conv_outputs = []
        for layer_name in self.conv_layers:
            layer_dict = self.conv_layers[layer_name]
            # 第一层卷积
            x_conv = embedded.unsqueeze(1)
            x_conv = layer_dict.conv1(x_conv)
            x_conv = layer_dict.bn1(x_conv)
            x_conv = F.relu(x_conv)
            x_conv = layer_dict.dropout(x_conv)

            # 第二层卷积
            x_conv = layer_dict.conv2(x_conv)
            x_conv = layer_dict.bn2(x_conv)
            x_conv = F.relu(x_conv)
            x_conv = layer_dict.dropout(x_conv)

            # 第三层卷积
            x_conv = layer_dict.conv3(x_conv)
            x_conv = layer_dict.bn3(x_conv)
            x_conv = F.relu(x_conv)

            # 重塑以适应注意力
            x_conv = x_conv.squeeze(-1)  # [batch, channels, seq_len]
            x_conv = layer_dict.attention(x_conv)

            # 全局最大池化
            pooled = F.max_pool1d(x_conv, x_conv.size(2))  # [batch, channels, 1]
            pooled = pooled.squeeze(-1)  # [batch, channels]
            conv_outputs.append(pooled)

        # 3. 特征融合
        feature_map = torch.cat(conv_outputs, dim=1)  # [batch, total_channels]

        # 4. 密集连接
        for dense_layer in self.dense_layers:
            feature_map = dense_layer(feature_map)

        # 5. Transformer处理
        x_trans = feature_map.unsqueeze(1)  # [batch, 1, features]
        x_trans = self.transformer_encoder(x_trans)
        x_trans = x_trans.squeeze(1)  # [batch, features]

        # 6. 自注意力池化
        attention_weights = self.attention_pooling(x_trans)
        attention_weights = F.softmax(attention_weights, dim=1)
        pooled_features = attention_weights * x_trans

        # 7. 主分类输出
        main_output = self.classifier(pooled_features)

        # 8. 辅助输出（训练时使用）
        aux_output = self.aux_classifier(pooled_features)

        if self.training:
            return main_output, aux_output
        else:
            return main_output

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            return torch.argmax(logits, dim=-1)

def create_ultra_config(vocab_size):
    """创建极限配置"""
    config = {
        'vocab_size': vocab_size,
        'embedding_dim': 400,
        'num_classes': 2,
        'filter_sizes': [2, 3, 4, 5, 7],
        'num_heads': 8,
        'dropout': 0.3,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'eval_interval': 30,
        'num_epoch': 25,
        'save_path': '../save_model/ultra_textcnn_best.pt',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'log_steps': 30,
        'batch_size': 32
    }
    return config

def train_ultra_model():
    """训练极限版TextCNN"""
    print("=" * 60)
    print("开始训练极限版TextCNN模型")
    print("=" * 60)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ultra_training.log', mode='w', encoding='utf-8'),
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
    # DataLoader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # 更小的batch size支持更大模型
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    print("   数据加载器创建完成")

    print("\n3. 创建模型...")
    # 创建模型配置
    vocab_size = len(json.load(open('../dataset/vocab.json', 'r', encoding='utf-8')))
    config_dict = create_ultra_config(vocab_size)
    config = namedtuple('config', config_dict.keys())(**config_dict)

    device = config.device
    model = UltraTextCNN(config).to(device)

    print(f"   词汇表大小: {vocab_size}")
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   使用设备: {device}")
    print(f"   嵌入维度: {config.embedding_dim}")
    print(f"   卷积核大小: {config.filter_sizes}")
    print(f"   注意力头数: {config.num_heads}")
    print(f"   学习率: {config.lr}")
    print(f"   Batch Size: {config.batch_size}")

    # 显示CUDA信息
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"   当前GPU使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    print("\n4. 开始训练...")
    print("-" * 60)
    logging.info("=== 极限版TextCNN训练开始 ===")

    # 自定义训练函数以支持辅助损失
    def train_ultra_custom(model, config, train_loader, val_loader):
        criterion = nn.CrossEntropyLoss()
        aux_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=1e-5,
            betas=(0.9, 0.95)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.7,
            patience=3
        )

        logging.info("开始训练...")
        model.train()
        best_acc = 0.0
        train_loss_history = []
        val_acc_history = []
        step = 0

        for epoch in range(config.num_epoch):
            logging.info(f"Epoch {epoch + 1}/{config.num_epoch}")
            epoch_loss = 0
            model.train()

            for i, data in enumerate(train_loader):
                step += 1
                inputs = data[0].to(config.device)
                labels = data[1].to(config.device)

                optimizer.zero_grad()
                outputs, aux_outputs = model(inputs)

                # 主损失 + 辅助损失
                main_loss = criterion(outputs, labels)
                aux_loss = aux_criterion(aux_outputs, labels)
                total_loss = main_loss + 0.3 * aux_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += total_loss.item()

                if step % config.eval_interval == 0:
                    model.eval()
                    correct = 0
                    total = 0

                    with torch.no_grad():
                        for data in val_loader:
                            inputs = data[0].to(config.device)
                            labels = data[1].to(config.device)
                            outputs = model(inputs)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]

                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                    val_acc = correct / total
                    avg_loss = epoch_loss / (i + 1)

                    logging.info(f'Step {step}: Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
                    val_acc_history.append(val_acc)

                    # 保存最佳模型
                    if val_acc > best_acc:
                        best_acc = val_acc
                        logging.info(f'🎉 新的最佳验证集准确率: {best_acc:.4f}')
                        save_dict = {
                            'model_state_dict': model.state_dict(),
                            'config': config._asdict() if hasattr(config, '_asdict') else config.__dict__
                        }
                        torch.save(save_dict, config.save_path)

                    scheduler.step(val_acc)
                    model.train()

            train_loss_history.append(epoch_loss / len(train_loader))
            logging.info(f'Epoch {epoch + 1} 完成，平均损失: {epoch_loss / len(train_loader):.4f}')

        return train_loss_history, val_acc_history

    # 训练模型
    print("\n   训练进度:")
    start_time = time.time()
    train_loss_history, val_acc_history = train_ultra_custom(model, config, train_loader, val_loader)
    end_time = time.time()

    # 使用训练历史记录避免未使用变量警告
    if len(train_loss_history) > 0:
        print(f"   最终训练损失: {train_loss_history[-1]:.4f}")

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
        print(f"   极限版TextCNN: {test_acc:.4f}")
        print(f"   性能提升: {improvement:+.4f} ({improvement/baseline_acc*100:+.2f}%)")

        # 检查是否达到90%目标
        if test_acc >= 0.90:
            print("\n🎉 恭喜！成功突破90%准确率大关！")
        else:
            gap = 0.90 - test_acc
            print(f"\n📈 距离90%目标还差 {gap:.4f}")

        return test_acc
    else:
        print(f"错误：未找到保存的模型文件 {config.save_path}")
        return 0.0

if __name__ == '__main__':
    os.makedirs('save_model', exist_ok=True)

    test_acc = train_ultra_model()

    print(f'\n极限版TextCNN最终测试准确率: {test_acc:.4f}')
    print('目标：突破90%大关！')