import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import json
import time
import logging
import os
import argparse
from collections import namedtuple

class SentimentDataset(Dataset):
    def __init__(self, data_path, vocab_path) -> None:
        self.vocab = json.load(open(vocab_path, 'r', encoding='utf-8'))
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = [json.loads(line) for line in f]
        random.shuffle(raw_data)
        data = []
        for item in raw_data:
            text = item['text']
            text_id = [self.vocab[t] if t in self.vocab.keys() else self.vocab['UNK'] for t in text]
            label = int(item['label'])
            data.append([text_id, label])
        return data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def collate_fn(data):
    pad_idx = 8019
    texts = [d[0] for d in data]
    label = [d[1] for d in data]
    batch_size = len(texts)
    max_length = max([len(t) for t in texts])
    text_ids = torch.ones((batch_size, max_length)).long().fill_(pad_idx)
    label_ids = torch.tensor(label).long()
    for idx, text in enumerate(texts):
        text_ids[idx, :len(text)] = torch.tensor(text)
    return text_ids, label_ids

class TextCNN(nn.Module):
    """Original TextCNN architecture that achieved 88.27% accuracy"""
    def __init__(self, config) -> None:
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size + 1, config.embedding_dim, padding_idx=8019)
        self.filter_sizes = sorted(list(set(config.filter_sizes)))
        self.num_channels = config.num_channels
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.num_channels, (k, config.embedding_dim))
            for k in self.filter_sizes
        ])

        # 计算总特征维度
        total_dim = len(self.filter_sizes) * self.num_channels
        self.fc = nn.Linear(total_dim, config.num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.unsqueeze(1)     # (batch, 1, seq_len, embed_dim)

        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x)
            conv_out = F.relu(conv_out.squeeze(3))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        x = torch.cat(conv_outputs, 1)
        x = self.fc(x)
        return x

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

# 为了向后兼容
OptimizedTextCNN = TextCNN
TextCNN = TextCNN

def cal_acc(pred, golden):
    correct = sum([int(x == y) for x, y in zip(pred, golden)])
    return correct / len(pred) if len(pred) > 0 else 0.0

def train(model, config, train_dataset, eval_dataset):
    # 使用简单的交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 使用原始的Adam优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=1e-4
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

        for i, data in enumerate(train_dataset):
            step += 1
            inputs = data[0].to(config.device)
            labels = data[1].to(config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 每N步验证一次
            if step % config.eval_interval == 0:
                model.eval()
                val_loss = 0
                correct = 0
                total = 0

                with torch.no_grad():
                    for data in eval_dataset:
                        inputs = data[0].to(config.device)
                        labels = data[1].to(config.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                val_acc = correct / total
                avg_epoch_loss = epoch_loss / (i + 1)

                logging.info(f'Step {step}: Loss: {avg_epoch_loss:.4f}, Val Acc: {val_acc:.4f}')
                # 添加验证准确率到历史记录
                val_acc_history.append(val_acc)

                # 保存最佳模型
                if val_acc > best_acc:
                    best_acc = val_acc
                    logging.info(f'🎉 新的最佳验证集准确率: {best_acc:.4f}')
                    # 确保保存目录存在
                    save_dir = os.path.dirname(config.save_path)
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    # 保存模型状态和配置
                    save_dict = {
                        'model_state_dict': model.state_dict(),
                        'config': config._asdict() if hasattr(config, '_asdict') else config.__dict__
                    }
                    torch.save(save_dict, config.save_path)

                model.train()

        # 记录每个epoch的平均损失
        train_loss_history.append(epoch_loss / len(train_dataset))
        logging.info(f'Epoch {epoch + 1} 完成，平均损失: {epoch_loss / len(train_dataset):.4f}')

    # 训练结束后，做最后一次完整验证
    logging.info("训练完成，进行最终验证...")
    model.eval()
    final_val_loss = 0
    final_correct = 0
    final_total = 0

    # 使用评估时的损失函数（不使用标签平滑）
    eval_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in eval_dataset:
            inputs = data[0].to(config.device)
            labels = data[1].to(config.device)
            outputs = model(inputs)
            loss = eval_criterion(outputs, labels)
            final_val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            final_total += labels.size(0)
            final_correct += (predicted == labels).sum().item()

    final_val_acc = final_correct / final_total
    val_acc_history.append(final_val_acc)  # 添加最终验证准确率
    logging.info(f'最终验证集准确率: {final_val_acc:.4f}, 平均损失: {final_val_loss/len(eval_dataset):.4f}')

    logging.info(f'🏆 训练完成！最佳验证集准确率: {best_acc:.4f}')
    return train_loss_history, val_acc_history

def evaluate(model, dataset, config):
    model.eval()
    pred = []
    golden = []
    total_loss = 0

    # 使用评估时的损失函数（不使用标签平滑）
    eval_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in dataset:
            inputs = data[0].to(config.device)
            labels = data[1].to(config.device)

            logits = model(inputs)
            loss = eval_criterion(logits, labels)
            total_loss += loss.item()

            pred.extend(model.predict(inputs).cpu().numpy().tolist())
            golden.extend(labels.cpu().numpy().tolist())

    acc = cal_acc(pred, golden)
    avg_loss = total_loss / len(dataset)
    logging.info(f'验证集准确率: {acc:.4f}, 平均损失: {avg_loss:.4f}')
    return acc