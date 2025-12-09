import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import logging
import os
import argparse
import json

from bert_model import RobertaClassifier, RobertaDataset, collate_fn_factory

# Setup logging
import os
from datetime import datetime

# 创建logs目录
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 生成日志文件名，包含时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"bert_training_{timestamp}.log")

# 配置logging同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # 控制台输出
        logging.StreamHandler(),
        # 文件输出
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)

logging.info(f"日志将同时保存到文件: {log_file}")

def train(model, train_loader, eval_loader, config):
    """
    Training loop for the RoBERTa model.
    """
    logging.info("=" * 60)
    logging.info("开始训练BERT模型...")
    logging.info("=" * 60)
    logging.info(f"训练参数:")
    logging.info(f"  - 学习率: {config.lr}")
    logging.info(f"  - Batch Size: {config.batch_size}")
    logging.info(f"  - 最大序列长度: {config.max_len}")
    logging.info(f"  - 训练轮数: {config.num_epoch}")
    logging.info(f"  - 预热步数: {config.warmup_steps}")
    logging.info(f"  - 保存路径: {config.save_path}")
    logging.info("=" * 60)

    # Optimizer and Scheduler
    # 新版本transformers中移除了AdamW，使用torch.optim.AdamW
    try:
        from transformers import get_linear_schedule_with_warmup
        # 使用torch.optim.AdamW
        import torch.optim as optim
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01, eps=1e-8)
    except ImportError:
        # 旧版本可能有不同的路径
        try:
            from transformers.optimization import AdamW, get_linear_schedule_with_warmup
            optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=0.01, eps=1e-8)
        except ImportError:
            # 最后的备选方案
            import torch.optim as optim
            from transformers import get_linear_schedule_with_warmup
            optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01, eps=1e-8)
    total_steps = len(train_loader) * config.num_epoch
    warmup_steps = getattr(config, 'warmup_steps', 500)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Loss function (最新版本支持label_smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.to(config.device)

    global_step = 0
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(config.num_epoch):
        logging.info(f"Epoch {epoch + 1}/{config.num_epoch}")
        model.train()

        epoch_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, token_type_ids, labels = [b.to(config.device) for b in batch]

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()

            if global_step % config.log_steps == 0:
                logging.info(f"Epoch {epoch + 1}, Step {global_step}, Loss: {loss.item():.4f}")

                # Evaluate on validation set
                model.eval()
                val_acc = evaluate(model, eval_loader, config)
                model.train()

                logging.info(f"验证集准确率: {val_acc:.4f}")

                if val_acc > best_acc:
                    best_acc = val_acc
                    logging.info(f"新的最佳验证集准确率: {best_acc:.4f}")
                    torch.save(model.state_dict(), config.save_path)

                if val_acc > 0.95:  # Early stopping if very high accuracy
                    logging.info("Reached very high accuracy, stopping training early")
                    break

        avg_epoch_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

    total_time = time.time() - start_time
    logging.info(f"训练完成，总耗时: {total_time:.2f}秒")
    logging.info(f"最佳验证集准确率: {best_acc:.4f}")

    return best_acc

def evaluate(model, eval_loader, config):
    """
    Evaluation function for the RoBERTa model.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids, attention_mask, token_type_ids, labels = [b.to(config.device) for b in batch]

            logits = model(input_ids, attention_mask, token_type_ids)
            predictions = torch.argmax(logits, dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

def test(model, test_loader, config):
    """
    Test function for the RoBERTa model.
    """
    logging.info("开始在测试集上评估...")
    test_acc = evaluate(model, test_loader, config)
    logging.info(f"测试集准确率: {test_acc:.4f}")
    return test_acc

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name', default='hfl/chinese-roberta-wwm-ext', help='Pretrained model name or path')
    parser.add_argument('--train_path', default='./dataset/train.jsonl', help='Path to training data')
    parser.add_argument('--val_path', default='./dataset/val.jsonl', help='Path to validation data')
    parser.add_argument('--test_path', default='./dataset/test.jsonl', help='Path to test data')
    parser.add_argument('--save_path', default='./save_model/bert_best.pt', help='Path to save the model')
    parser.add_argument('--num_epoch', default=6, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--lr', default=3e-5, type=float, help='Learning rate')
    parser.add_argument('--max_len', default=256, type=int, help='Maximum sequence length')
    parser.add_argument('--grad_accumulation_steps', default=1, type=int, help='Gradient accumulation steps for larger effective batch size')
    parser.add_argument('--log_steps', default=25, type=int, help='Log frequency')
    parser.add_argument('--warmup_steps', default=500, type=int, help='Number of warmup steps')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Operation mode: train or test')

    config = parser.parse_args()

    # Enhanced configuration
    # 推荐的中文文本分类模型（按优先级排序）：
    # 1. 'hfl/chinese-macbert-large' - MacBERT专门优化了分类任务，large版本参数更多
    # 2. 'hfl/chinese-roberta-wwm-ext-large' - 当前使用的，通用中文RoBERTa
    # 3. 'hfl/chinese-electra-180g-large-discriminator' - ELECTRA架构，通常在分类任务上表现优异
    # 4. 'uer/roberta-base-finetuned-chinanews-chinese' - 在中文新闻分类上微调过
    # 5. 'Langboat/mengzi-bert-large' - 澜舟科技的Mengzi模型，在多项中文任务上表现优异

    config.pretrained_model_name = 'hfl/chinese-macbert-large'  # 更适合文本分类
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.weight_decay = 0.01

    logging.info(f"Using device: {config.device}")
    logging.info(f"Pretrained model: {config.pretrained_model_name}")
    logging.info("推荐模型说明：")
    logging.info("  - MacBERT: 改进的MLM策略，在分类任务上通常优于RoBERTa")
    logging.info("  - Large版本: 更多的参数，更强的表达能力")
    logging.info("  - 专为中文优化: 更好的中文理解和表达能力")
    logging.info("=" * 60)
    logging.info("硬件配置信息：")
    logging.info(f"  - GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    logging.info(f"  - 可用显存: {torch.cuda.memory_reserved(0) / 1024**3:.1f}GB")
    logging.info(f"  - 训练配置：Batch Size={config.batch_size}, Max Length={config.max_len}")
    logging.info("=" * 60)

    # Setup tokenizer
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name, local_files_only=True)
    except:
        logging.info(f"Local tokenizer not found, downloading {config.pretrained_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)

    # Prepare datasets
    if config.mode == 'train':
        train_dataset = RobertaDataset(config.train_path)
        val_dataset = RobertaDataset(config.val_path)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn_factory(tokenizer, config.max_len)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn_factory(tokenizer, config.max_len)
        )

        # Setup model
        model = RobertaClassifier(config.pretrained_model_name, num_classes=2)

        logging.info(f"训练数据集大小: {len(train_dataset)}")
        logging.info(f"验证数据集大小: {len(val_dataset)}")
        logging.info(f"训练批次数: {len(train_loader)}")
        logging.info(f"验证批次数: {len(val_loader)}")
        logging.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

        # Create save directory
        save_dir = os.path.dirname(config.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        logging.info("=" * 60)
        logging.info("准备就绪，即将开始训练...")
        logging.info("=" * 60)

        # Start training
        best_val_acc = train(model, train_loader, val_loader, config)

        # 训练完成后，自动在测试集上评估
        logging.info("=" * 60)
        logging.info("训练完成，开始在测试集上评估...")
        logging.info("=" * 60)

        # 加载测试数据
        test_dataset = RobertaDataset(config.test_path)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn_factory(tokenizer, config.max_len)
        )

        # 重新加载最佳模型
        model = RobertaClassifier(config.pretrained_model_name, num_classes=2)
        model.load_state_dict(torch.load(config.save_path, map_location=config.device))
        model.to(config.device)

        logging.info(f"测试数据集大小: {len(test_dataset)}")
        logging.info(f"已加载最佳验证模型: {config.save_path}")

        # 在测试集上评估
        test_acc = test(model, test_loader, config)

        logging.info("=" * 60)
        logging.info("最终结果总结:")
        logging.info(f"  - 最佳验证集准确率: {best_val_acc:.4f}")
        logging.info(f"  - 测试集准确率: {test_acc:.4f}")
        logging.info("=" * 60)

    elif config.mode == 'test':
        logging.info("=" * 60)
        logging.info("测试模式")
        logging.info("=" * 60)

        if not os.path.exists(config.save_path):
            logging.error(f"模型文件未找到: {config.save_path}")
            return

        logging.info("正在加载测试数据...")
        test_dataset = RobertaDataset(config.test_path)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn_factory(tokenizer, config.max_len)
        )

        logging.info("正在加载模型...")
        # Load model
        model = RobertaClassifier(config.pretrained_model_name, num_classes=2)
        model.load_state_dict(torch.load(config.save_path, map_location=config.device))
        model.to(config.device)

        logging.info(f"模型已加载: {config.save_path}")
        logging.info(f"测试数据集大小: {len(test_dataset)}")
        logging.info(f"测试批次数: {len(test_loader)}")
        logging.info("=" * 60)

        # Test
        test_acc = test(model, test_loader, config)

if __name__ == '__main__':
    main()