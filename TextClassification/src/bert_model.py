import os
import sys
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer, AutoModel

class OptimizedRobertaClassifier(nn.Module):
    """Maximum performance RoBERTa classifier with advanced features"""
    def __init__(self, pretrained_model_name='hfl/chinese-roberta-wwm-ext-large', num_classes=2):
        super(OptimizedRobertaClassifier, self).__init__()

        # Cache directory management
        cache_dir = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE') or os.path.abspath(os.path.dirname(__file__))

        # Load pretrained model
        try:
            self.bert = AutoModel.from_pretrained(pretrained_model_name, cache_dir=cache_dir)
        except Exception as e:
            logging.error(f"Failed to load AutoModel '{pretrained_model_name}': {e}")
            raise

        hidden_size = self.bert.config.hidden_size

        # 根据hidden_size自动计算合适的num_heads
        # 确保embed_dim能被num_heads整除
        possible_heads = [1, 2, 4, 8, 16]
        num_heads = max([h for h in possible_heads if hidden_size % h == 0])
        if num_heads == 0:
            num_heads = 1  # 如果没有合适的，使用1

        print(f"BERT hidden_size: {hidden_size}, using num_heads: {num_heads}")

        # 高级特征提取架构
        # 1. 多头自注意力池化
        self.multihead_attention = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, dropout=0.1, batch_first=True
        )

        # 2. 双向LSTM增强序列建模
        self.lstm = nn.LSTM(
            hidden_size, hidden_size // 2,
            num_layers=2, bidirectional=True,
            dropout=0.2, batch_first=True
        )

        # 3. 门控融合单元
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # 4. 层归一化和dropout
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)

        # 5. 深层分类器
        self.classifier = nn.Sequential(
            # 第一层
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),

            # 第二层
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.15),

            # 第三层
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),

            # 输出层
            nn.Linear(hidden_size // 4, num_classes)
        )

        # 6. 权重初始化
        self._init_weights()

    def _init_weights(self):
        """自定义权重初始化"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            output_attentions=True
        )

        hidden_states = outputs.hidden_states  # 所有层的隐藏状态
        last_hidden_state = hidden_states[-1]  # 最后一层

        # 1. CLS token表示
        cls_representation = last_hidden_state[:, 0, :]

        # 2. 多头注意力池化
        attention_output, attention_weights = self.multihead_attention(
            last_hidden_state, last_hidden_state, last_hidden_state,
            key_padding_mask=~attention_mask.bool()
        )
        pooled_attention = attention_output[:, 0, :]

        # 3. LSTM序列建模
        lstm_output, _ = self.lstm(last_hidden_state)
        pooled_lstm = lstm_output[:, 0, :]

        # 4. 多层特征融合
        # 使用最后4层的BERT输出
        bert_fused = torch.stack(hidden_states[-4:]).mean(dim=0)
        bert_pooled = bert_fused[:, 0, :]

        # 门控融合
        combined_representations = torch.cat([pooled_attention, pooled_lstm], dim=-1)
        gate_weights = self.gate(combined_representations)

        # 加权融合所有特征
        final_features = (gate_weights * cls_representation +
                         (1 - gate_weights) * pooled_attention +
                         0.2 * pooled_lstm +
                         0.1 * bert_pooled)

        # 层归一化和dropout
        final_features = self.layer_norm1(final_features)
        final_features = self.dropout1(final_features)
        final_features = self.layer_norm2(final_features)
        final_features = self.dropout2(final_features)

        # 深层分类
        logits = self.classifier(final_features)

        return logits

    def predict(self, input_ids, attention_mask, token_type_ids):
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.argmax(logits, dim=-1)

# 为了向后兼容
RobertaClassifier = OptimizedRobertaClassifier

class RobertaDataset(Dataset):
    def __init__(self, file_path):
        self.texts, self.labels = self.load_data(file_path)

    def load_data(self, file_path):
        texts, labels = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                texts.append(item['text'])
                labels.append(int(item['label']))
        return texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn_factory(tokenizer, max_len):
    def collate_fn(batch):
        texts, labels = zip(*batch)

        # 确保文本格式正确
        normalized_texts = [
            t if isinstance(t, str) else " ".join(t) if isinstance(t, (list, tuple)) else str(t)
            for t in texts
        ]

        # 高级tokenization设置
        inputs = tokenizer(
            normalized_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True,
            return_overflowing_tokens=False
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs.get('token_type_ids', torch.zeros_like(input_ids))

        labels = torch.tensor(labels, dtype=torch.long)

        return input_ids, attention_mask, token_type_ids, labels
    return collate_fn