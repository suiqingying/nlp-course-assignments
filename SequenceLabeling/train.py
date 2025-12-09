import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import json
from inference.CRF import CRF
import argparse
from collections import namedtuple
import os
from tqdm import tqdm
import time
import logging
import csv
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False

class BiLSTM_CRF(nn.Module):
    def __init__(self, config):
        super(BiLSTM_CRF, self).__init__()
        self.target_size = config.tag_size
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout
        self.embedding_dim = config.embedding_dim
        self.num_layers = getattr(config, 'num_layers', 2)
        
        # 词嵌入层
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.drop = nn.Dropout(self.dropout)
        
        # BiLSTM层 - 增加层间dropout
        self.lstm = nn.LSTM(
            self.embedding_dim, 
            self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0  # 层间dropout
        )
        
        # 添加LayerNorm提升稳定性
        self.layer_norm = nn.LayerNorm(self.hidden_dim * 2)
        
        self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.target_size + 2)
        self.crf = CRF(self.target_size, config.gpu)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，提升训练稳定性"""
        # 初始化embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # 初始化LSTM权重
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # 设置遗忘门偏置为1，帮助长期记忆
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        # 初始化线性层
        nn.init.xavier_uniform_(self.hidden2tag.weight)
        nn.init.zeros_(self.hidden2tag.bias)
    
    def get_tags(self, word_input):
        embeddings = self.drop(self.embedding(word_input))
        hidden = None
        lstm_feat, hidden = self.lstm(embeddings, hidden)
        # 添加LayerNorm
        lstm_feat = self.layer_norm(lstm_feat)
        feats = self.drop(lstm_feat)
        feats = self.hidden2tag(feats)
        return feats
    
    def forward(self, word_input, mask, tags):
        feats = self.get_tags(word_input)
        loss = self.crf(feats, mask, tags)
        return loss
    
    def decode(self, word_input, mask):
        feats = self.get_tags(word_input)
        tag_seq = self.crf.decode(feats, mask)
        return tag_seq
    
class NER_dataset(Dataset):
    def __init__(self, data_path, chr_vocab_path, tag_vocab_path):
        self.data = self.load_data(data_path)
        self.chr_vocab = json.load(open(chr_vocab_path, 'r', encoding='utf-8'))
        self.tag_vocab = json.load(open(tag_vocab_path, 'r', encoding='utf-8'))
        print(len(self.data))
        
    def load_data(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            sentence = []
            tags = []
            for line in f:
                # 兼容 Windows 的 \r\n 以及带空格的空行
                if not line.strip():
                    data.append([sentence, tags])
                    sentence = []
                    tags = []
                else:
                    word, tag = line.strip().split('\t')
                    sentence.append(word)
                    tags.append(tag)
            if sentence:
                data.append([sentence, tags])
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence, tags = self.data[index]
        char_ids = [self.chr_vocab[char] for char in sentence]
        tag_ids = [self.tag_vocab[tag] for tag in tags]
        return torch.tensor(char_ids), torch.tensor(tag_ids)

def collate_fn(data):
    words = [item[0] for item in data]
    tags = [item[1] for item in data]
    max_seq_len = max([t.shape[0] for t in words])
    batch_size = len(words)
    word_ids = torch.zeros(batch_size, max_seq_len).long()
    mask = torch.zeros(batch_size, max_seq_len).long()
    tag_ids = torch.zeros(batch_size, max_seq_len).long()
    for idx, (word, tag) in enumerate(zip(words, tags)):
        word_ids[idx, :word.shape[0]] = word
        tag_ids[idx, :tag.shape[0]] = tag
        mask[idx, :word.shape[0]] = 1
    return word_ids, mask.bool(), tag_ids

def extract_BIO(seq):
    """
    Extracts the BIO tag from a sequence.
    Returns a list of tuples (tag, chunk_start, chunk_end)
    """
    res = []
    chunk_start = -1
    chunk_type = ''
    for i, tag in enumerate(seq):
        if tag.startswith('B-'):
            if chunk_start != -1:
                res.append((chunk_type, chunk_start, i - 1))
            chunk_type = tag[2:]
            chunk_start = i
        elif tag.startswith('I-'):
            continue
        else:
            if chunk_start != -1:
                res.append((chunk_type, chunk_start, i - 1))
                chunk_start = -1
                chunk_type = ''
    if chunk_start != -1:
        res.append((chunk_type, chunk_start, len(seq) - 1))
    return res

def cal_F1(pre_seq, golden_seq):
    """
    Calculate precision, recall, and F1 score based on predicted and gold sequences.
    """
    pre_chunks = extract_BIO(pre_seq)
    gold_chunks = extract_BIO(golden_seq)
    
    pre_set = set(pre_chunks)
    gold_set = set(gold_chunks)
    
    true_positives = len(pre_set & gold_set)
    precision = true_positives / len(pre_set) if pre_set else 0
    recall = true_positives / len(gold_set) if gold_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    return precision, recall, f1

def train(model, config, train_dataset, eval_dataset, tag_vocab, logger=None, history=None):
    device = next(model.parameters()).device  # 获取模型所在的设备
    lr = getattr(config, "lr", 0.001)
    weight_decay = getattr(config, "weight_decay", 0.01)
    max_grad_norm = getattr(config, "max_grad_norm", 5.0)
    
    # 使用AdamW优化器，添加权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 添加学习率调度器
    total_steps = len(train_dataset) * config.num_epoch
    warmup_steps = int(total_steps * 0.1)  # 10% warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,  # warmup比例
        anneal_strategy='cos'
    )
    
    model.train()
    global_step = 0
    best_f1 = 0
    total_batches = len(train_dataset)
    patience = 5  # 早停耐心值
    no_improve_count = 0

    # 初始化history如果为空
    if history is None:
        history = {
            'epoch_losses': [],
            'epoch_val_f1s': [],
            'epoch_times': [],
            'steps': [],
            'train_losses': [],
            'val_f1s': [],
            'epochs': []
        }

    print(f"开始训练，总共 {config.num_epoch} 个epoch，每个epoch有 {total_batches} 个batch")
    if logger:
        logger.info(f"开始训练，总共 {config.num_epoch} 个epoch，每个epoch有 {total_batches} 个batch")
    print("-" * 80)

    for epoch in range(config.num_epoch):
        epoch_start_time = time.time()
        total_loss = 0
        loss_count = 0

        # 创建进度条
        pbar = tqdm(train_dataset, desc=f'Epoch {epoch+1}/{config.num_epoch}',
                    leave=False, ncols=100)

        for data in pbar:
            # 将数据移动到GPU
            if config.gpu:
                data = [d.to(device) if torch.is_tensor(d) else d for d in data]

            loss = model(*data)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()  # 更新学习率
            model.zero_grad()
            global_step += 1

            total_loss += loss.item()
            loss_count += 1

            # 更新进度条显示
            if global_step % 10 == 0:
                avg_loss = total_loss / loss_count
                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                 'avg_loss': f'{avg_loss:.4f}',
                                 'step': global_step})

            if global_step % config.eval_interval == 0:
                print(f'\n评估时间点 - Epoch: {epoch+1}, Global Step: {global_step}, Loss: {loss.item():.4f}')
                f1 = evaluate(model, eval_dataset, tag_vocab)

                # 记录到历史
                history['steps'].append(global_step)
                history['train_losses'].append(loss.item())
                history['val_f1s'].append(f1)
                history['epochs'].append(epoch + 1)

                if logger:
                    logger.info(f'Step {global_step} - Loss: {loss.item():.4f}, Val F1: {f1:.4f}')

                if f1 > best_f1:
                    best_f1 = f1
                    save_dir = os.path.dirname(config.save_path) or '.'
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(model.state_dict(), config.save_path)
                    print(f'新的最佳模型已保存，F1分数: {best_f1:.4f}')
                    if logger:
                        logger.info(f'新的最佳模型已保存，F1分数: {best_f1:.4f}')
                model.train()  # 切换回训练模式

        # Epoch结束后的统计
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = total_loss / loss_count
        print(f'\nEpoch {epoch+1}/{config.num_epoch} 完成')
        print(f'  - 平均损失: {avg_epoch_loss:.4f}')
        print(f'  - 训练时间: {epoch_time:.2f}秒')
        print(f'  - 总步数: {global_step}')

        # 每个epoch结束后评估一次
        print(f'\nEpoch {epoch+1} 评估:')
        f1 = evaluate(model, eval_dataset, tag_vocab)

        # 记录epoch级别的指标
        history['epoch_losses'].append(avg_epoch_loss)
        history['epoch_val_f1s'].append(f1)
        history['epoch_times'].append(epoch_time)

        if logger:
            logger.info(f'Epoch {epoch+1} 完成 - 平均损失: {avg_epoch_loss:.4f}, 训练时间: {epoch_time:.2f}秒, Val F1: {f1:.4f}')

        if f1 > best_f1:
            best_f1 = f1
            save_dir = os.path.dirname(config.save_path) or '.'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), config.save_path)
            print(f'新的最佳模型已保存，F1分数: {best_f1:.4f}')
            if logger:
                logger.info(f'新的最佳模型已保存，F1分数: {best_f1:.4f}')
        model.train()  # 切换回训练模式

        print("-" * 80)

    print(f'\n训练完成！最佳F1分数: {best_f1:.4f}')
    if logger:
        logger.info(f'训练完成！最佳F1分数: {best_f1:.4f}')

    return best_f1, history

def evaluate(model, dataset, tag_vocab):
    device = next(model.parameters()).device  # 获取模型所在的设备
    model.eval()
    pred = []
    golden = []

    # 为评估添加进度条
    pbar = tqdm(dataset, desc='评估中', leave=False, ncols=80)

    with torch.no_grad():
        for data in pbar:
            # data: (word_ids, mask, tag_ids)
            word_ids, mask, tag_ids = data
            
            # 将数据移动到GPU
            word_ids = word_ids.to(device)
            mask = mask.to(device)
            tag_ids = tag_ids.to(device)
            
            # 解码预测
            decoded = model.decode(word_ids, mask)  # (batch_size, seq_len)
            
            # 按样本处理，使用mask过滤padding
            batch_size = word_ids.size(0)
            for i in range(batch_size):
                seq_mask = mask[i]  # (seq_len,)
                seq_len = seq_mask.sum().item()  # 实际序列长度
                
                # 获取有效部分的预测和真实标签
                pred_seq = decoded[i, :seq_len].tolist()
                gold_seq = tag_ids[i, :seq_len].tolist()
                
                pred.extend(pred_seq)
                golden.extend(gold_seq)

            # 更新进度条
            pbar.set_postfix({'已处理': f'{len(pbar)} batches'})

    pred = [tag_vocab[i] for i in pred]
    golden = [tag_vocab[i] for i in golden]
    precision, recall, f1 = cal_F1(pred, golden)

    print(f'  - 精确率(Precision): {precision:.4f}')
    print(f'  - 召回率(Recall): {recall:.4f}')
    print(f'  - F1分数: {f1:.4f}')

    return f1
 
# 添加保存结果的辅助函数
def save_predictions(model, dataset, tag_vocab, output_path):
    """保存预测结果"""
    device = next(model.parameters()).device
    model.eval()
    predictions = []

    with open(output_path, 'w', encoding='utf-8') as f:
        with torch.no_grad():
            for i, data in enumerate(dataset):
                if torch.is_tensor(data[0]):
                    data = [d.to(device) if torch.is_tensor(d) else d for d in data]

                # 获取预测
                pred_ids = model.decode(*data[:-1]).squeeze().int().tolist()
                pred_tags = [tag_vocab[id] for id in pred_ids if id >= 0]

                # 获取真实标签
                true_ids = data[-1].squeeze().int().tolist()
                true_tags = [tag_vocab[id] for id in true_ids if id >= 0]

                # 写入文件
                f.write(f"样本 {i+1}:\n")
                f.write(f"预测: {' '.join(pred_tags)}\n")
                f.write(f"真实: {' '.join(true_tags)}\n")
                f.write("-" * 50 + "\n")

    print(f"预测结果已保存到: {output_path}")

def plot_training_curves(history, save_dir):
    """绘制并保存训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    ax1.plot(history['epoch_losses'], 'b-', label='训练损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.set_title('训练损失曲线')
    ax1.legend()
    ax1.grid(True)

    # F1分数曲线
    ax2.plot(history['epoch_val_f1s'], 'r-', label='验证集F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1分数')
    ax2.set_title('验证集F1分数曲线')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制step级别的F1曲线
    if len(history['steps']) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(history['steps'], history['val_f1s'], 'g-', alpha=0.7, label='验证F1（每100步）')
        plt.xlabel('训练步数')
        plt.ylabel('F1分数')
        plt.title('验证集F1分数变化（训练步数）')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'f1_by_steps.png'), dpi=300, bbox_inches='tight')
        plt.close()

def save_results_to_csv(history, config, save_dir, batch_size):
    """保存结果到CSV文件"""
    # 保存训练历史
    with open(os.path.join(save_dir, 'training_history.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Val F1', 'Time (s)'])
        for i in range(len(history['epoch_losses'])):
            writer.writerow([
                i + 1,
                history['epoch_losses'][i],
                history['epoch_val_f1s'][i],
                history['epoch_times'][i]
            ])

    # 保存配置和最佳结果
    with open(os.path.join(save_dir, 'experiment_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['参数', '值'])
        writer.writerow(['模型类型', 'BiLSTM-CRF'])
        writer.writerow(['学习率', config.lr])
        writer.writerow(['批次大小', batch_size])
        writer.writerow(['隐藏层维度', config.hidden_dim])
        writer.writerow(['Dropout', config.dropout])
        writer.writerow(['训练轮数', config.num_epoch])
        writer.writerow(['最佳F1分数', max(history['epoch_val_f1s'])])
        writer.writerow(['最佳F1 Epoch', history['epoch_val_f1s'].index(max(history['epoch_val_f1s'])) + 1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BiLSTM-CRF序列标注模型训练')
    parser.add_argument('--save_path', default='save_model/best.pt', help='模型保存路径')
    parser.add_argument('--train', default='inference/train.txt', help='训练数据路径')
    parser.add_argument('--test', default='inference/test.txt', help='测试数据路径')
    parser.add_argument('--val', default='inference/val.txt', help='验证数据路径')
    parser.add_argument('--chr_vocab', default='inference/chr_vocab.json', help='字符词汇表路径')
    parser.add_argument('--tag_vocab', default='inference/tag_vocab.json', help='标签词汇表路径')
    parser.add_argument('--num_epoch', default=25, type=int, help='训练轮数')
    parser.add_argument('--lr', default=0.001, type=float, help='学习率')  # 降低学习率
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout比率')  # 增加dropout
    parser.add_argument('--hidden_dim', default=512, type=int, help='隐藏层维度')  # 增大隐藏层
    parser.add_argument('--embedding_dim', default=256, type=int, help='词嵌入维度')  # 独立的embedding维度
    parser.add_argument('--num_layers', default=2, type=int, help='LSTM层数')
    parser.add_argument('--batch_size', default=64, type=int, help='批次大小')
    parser.add_argument('--eval_interval', default=100, type=int, help='评估间隔（步数）')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='权重衰减')
    parser.add_argument('--max_grad_norm', default=5.0, type=float, help='梯度裁剪阈值')
    parser.add_argument('--log_dir', default='logs', help='日志保存目录')
    arg = parser.parse_args()

    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(arg.log_dir, f'bilstm_crf_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # 初始化训练历史记录
    history = {
        'epoch_losses': [],
        'epoch_val_f1s': [],
        'epoch_times': [],
        'steps': [],
        'train_losses': [],
        'val_f1s': [],
        'epochs': []
    }

    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    print(f"使用设备: {device}")

    # 加载数据集
    logger.info("正在加载数据集...")
    print("正在加载数据集...")
    train_dataset = NER_dataset(arg.train, arg.chr_vocab, arg.tag_vocab)
    val_dataset = NER_dataset(arg.val, arg.chr_vocab, arg.tag_vocab)
    test_dataset = NER_dataset(arg.test, arg.chr_vocab, arg.tag_vocab)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=arg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 加载词汇表
    chr_vocab = json.load(open(arg.chr_vocab, 'r', encoding='utf-8'))
    tag_vocab = json.load(open(arg.tag_vocab, 'r', encoding='utf-8'))
    tag_vocab = {v:k for k,v in tag_vocab.items()}

    # 记录数据集信息
    logger.info("="*80)
    logger.info("数据集信息:")
    logger.info(f"  - 训练集样本数: {len(train_dataset)}")
    logger.info(f"  - 验证集样本数: {len(val_dataset)}")
    logger.info(f"  - 测试集样本数: {len(test_dataset)}")
    logger.info(f"  - 字符词汇表大小: {len(chr_vocab)}")
    logger.info(f"  - 标签词汇表大小: {len(tag_vocab)}")

    print("\n" + "="*80)
    print("数据集信息:")
    print(f"  - 训练集样本数: {len(train_dataset)}")
    print(f"  - 验证集样本数: {len(val_dataset)}")
    print(f"  - 测试集样本数: {len(test_dataset)}")
    print(f"  - 字符词汇表大小: {len(chr_vocab)}")
    print(f"  - 标签词汇表大小: {len(tag_vocab)}")
    print("="*80 + "\n")

    # 模型配置
    config = {
        "vocab_size": len(chr_vocab),
        "hidden_dim": arg.hidden_dim,
        "dropout": arg.dropout,
        "embedding_dim": arg.embedding_dim,
        "num_layers": arg.num_layers,
        "tag_size": len(tag_vocab),
        "gpu": (device.type == 'cuda'),
        "lr": arg.lr,
        "weight_decay": arg.weight_decay,
        "max_grad_norm": arg.max_grad_norm,
        "num_epoch": arg.num_epoch,
        "eval_interval": arg.eval_interval,
        "save_path": arg.save_path
    }
    config = namedtuple('Config', config.keys())(*config.values())

    # 记录训练参数
    logger.info("="*80)
    logger.info("训练参数:")
    logger.info(f"  - 学习率: {config.lr}")
    logger.info(f"  - 批次大小: {arg.batch_size}")
    logger.info(f"  - 隐藏层维度: {config.hidden_dim}")
    logger.info(f"  - 词嵌入维度: {config.embedding_dim}")
    logger.info(f"  - LSTM层数: {config.num_layers}")
    logger.info(f"  - Dropout: {config.dropout}")
    logger.info(f"  - 权重衰减: {config.weight_decay}")
    logger.info(f"  - 梯度裁剪: {config.max_grad_norm}")
    logger.info(f"  - 训练轮数: {config.num_epoch}")
    logger.info(f"  - 评估间隔: {config.eval_interval} 步")
    logger.info(f"  - 模型保存路径: {config.save_path}")
    logger.info(f"  - 使用GPU: {config.gpu}")

    # 打印训练参数
    print("训练参数:")
    print(f"  - 学习率: {config.lr}")
    print(f"  - 批次大小: {arg.batch_size}")
    print(f"  - 隐藏层维度: {config.hidden_dim}")
    print(f"  - 词嵌入维度: {config.embedding_dim}")
    print(f"  - LSTM层数: {config.num_layers}")
    print(f"  - Dropout: {config.dropout}")
    print(f"  - 权重衰减: {config.weight_decay}")
    print(f"  - 梯度裁剪: {config.max_grad_norm}")
    print(f"  - 训练轮数: {config.num_epoch}")
    print(f"  - 评估间隔: {config.eval_interval} 步")
    print(f"  - 模型保存路径: {config.save_path}")
    print(f"  - 使用GPU: {config.gpu}")
    print("\n" + "="*80 + "\n")

    # 创建并开始训练
    model = BiLSTM_CRF(config)
    model = model.to(device)  # 将模型移动到GPU

    # 保存模型结构
    with open(os.path.join(log_dir, 'model_architecture.txt'), 'w', encoding='utf-8') as f:
        f.write(str(model))

    best_f1, history = train(model, config, train_loader, val_loader, tag_vocab, logger, history)

    # 训练完成后在测试集上评估
    logger.info("\n在测试集上评估最终模型:")
    print("\n在测试集上评估最终模型:")
    model.load_state_dict(torch.load(config.save_path))
    test_f1 = evaluate(model, test_loader, tag_vocab)

    # 保存测试集预测结果
    test_pred_path = os.path.join(log_dir, 'test_predictions.txt')
    save_predictions(model, test_loader[:100], tag_vocab, test_pred_path)  # 只保存前100个样本

    logger.info(f"\n测试集最终F1分数: {test_f1:.4f}")
    print(f"\n测试集最终F1分数: {test_f1:.4f}")

    # 保存训练历史
    with open(os.path.join(log_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 绘制并保存训练曲线
    plot_training_curves(history, log_dir)

    # 保存结果到CSV
    save_results_to_csv(history, config, log_dir, arg.batch_size)

    logger.info(f"\n所有训练记录已保存到: {log_dir}")
    print(f"\n所有训练记录已保存到: {log_dir}")