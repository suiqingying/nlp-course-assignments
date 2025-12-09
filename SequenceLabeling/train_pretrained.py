import argparse
import json
import os
import random
from typing import List, Tuple
import time
import logging

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from train import cal_F1


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_bio(data_path: str) -> List[Tuple[List[str], List[str]]]:
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        sentence, tags = [], []
        for line in f:
            if line == "\n":
                if sentence:
                    data.append((sentence, tags))
                sentence, tags = [], []
            else:
                word, tag = line.strip().split("\t")
                sentence.append(word)
                tags.append(tag)
        if sentence:
            data.append((sentence, tags))
    return data


class HFNERDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        tag_vocab_path: str,
        max_length: int = 128,
    ):
        self.samples = load_bio(data_path)
        self.tokenizer = tokenizer
        tag_vocab = json.load(open(tag_vocab_path, "r", encoding="utf-8"))
        self.tag2id = {k: int(v) for k, v in tag_vocab.items()}
        self.id2tag = {v: k for k, v in self.tag2id.items()}
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        words, tags = self.samples[idx]
        tag_ids = [self.tag2id[t] for t in tags]
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        labels = [-100] * len(encoding["input_ids"])
        word_ids = encoding.word_ids()
        for i, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx >= len(tag_ids):
                continue
            labels[i] = tag_ids[word_idx]

        encoding["labels"] = labels
        return {k: torch.tensor(v) for k, v in encoding.items()}


def train_one_epoch(model, dataloader, optimizer, scheduler, device, max_grad_norm, epoch, num_epochs, logger=None):
    model.train()
    total_loss = 0.0

    # 创建进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False, ncols=100)

    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        # 更新进度条显示
        avg_loss = total_loss / (step + 1)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{avg_loss:.4f}'})

    return total_loss / max(len(dataloader), 1)


def evaluate(model, dataloader, id2tag, device, logger=None):
    model.eval()
    pred, golden = [], []

    # 为评估添加进度条
    pbar = tqdm(dataloader, desc='评估中', leave=False, ncols=80)

    with torch.no_grad():
        for batch in pbar:
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            input_ids = batch["input_ids"].to(device)
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = labels.detach().cpu().numpy()
            mask = attention_mask.detach().cpu().numpy()
            pred_ids = logits.argmax(axis=-1)
            for l_seq, p_seq, m_seq in zip(label_ids, pred_ids, mask):
                for l, p, m in zip(l_seq, p_seq, m_seq):
                    if m == 0 or l == -100:
                        continue
                    golden.append(id2tag[int(l)])
                    pred.append(id2tag[int(p)])

            # 更新进度条
            pbar.set_postfix({'已处理': f'{len(pbar)} batches'})

    precision, recall, f1 = cal_F1(pred, golden)

    # 记录评估结果
    if logger:
        logger.info(f'  - 精确率(Precision): {precision:.4f}')
        logger.info(f'  - 召回率(Recall): {recall:.4f}')
        logger.info(f'  - F1分数: {f1:.4f}')

    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description='基于预训练模型的BiLSTM-CRF序列标注训练')
    parser.add_argument("--model_name", default="hfl/chinese-macbert-base", help="预训练模型名称")
    parser.add_argument("--train", default="inference/train.txt", help="训练数据路径")
    parser.add_argument("--val", default="inference/val.txt", help="验证数据路径")
    parser.add_argument("--tag_vocab", default="inference/tag_vocab.json", help="标签词汇表路径")
    parser.add_argument("--test", default=None, help="测试数据路径(可选，提供则在训练后评估测试集)")
    parser.add_argument("--eval_only", action="store_true", help="只评估已训练模型，跳过训练过程")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--num_epoch", type=int, default=5, help="训练轮数")
    parser.add_argument("--lr", type=float, default=3e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--save_path", default="save_model/bert_best.pt", help="模型保存路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default="auto", help="设备选择 (auto/cpu/cuda:0)")
    args = parser.parse_args()

    # 设置日志
    log_dir = os.path.join("logs", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'bert_training.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    set_seed(args.seed)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    logger.info(f"使用设备: {device}")
    logger.info("正在加载数据集和模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = HFNERDataset(args.train, tokenizer, args.tag_vocab, args.max_length)
    val_ds = HFNERDataset(args.val, tokenizer, args.tag_vocab, args.max_length)
    id2tag = train_ds.id2tag

    # 打印数据集信息
    logger.info("="*80)
    logger.info("数据集和模型信息:")
    logger.info(f"  - 预训练模型: {args.model_name}")
    logger.info(f"  - 训练集样本数: {len(train_ds)}")
    logger.info(f"  - 验证集样本数: {len(val_ds)}")
    logger.info(f"  - 标签类别数: {len(train_ds.tag2id)}")
    logger.info(f"  - 设备: {device}")
    logger.info("="*80)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(train_ds.tag2id),
        id2label=train_ds.id2tag,
        label2id=train_ds.tag2id,
    )
    model.to(device)

    # 如仅评估，提前加载权重并跳过训练
    if args.eval_only:
        if os.path.exists(args.save_path):
            logger.info(f"加载已训练模型权重: {args.save_path}")
            model.load_state_dict(torch.load(args.save_path, map_location=device))
        else:
            logger.warning(f"未找到已保存模型，将使用当前初始化权重: {args.save_path}")

        # 如果提供了测试集，则优先在测试集评估，否则在验证集评估
        if args.test:
            if not os.path.exists(args.test):
                logger.error(f"测试集不存在: {args.test}")
                return
            test_ds = HFNERDataset(args.test, tokenizer, args.tag_vocab, args.max_length)
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator
            )
            logger.info("开始在测试集上评估...")
            precision, recall, f1 = evaluate(model, test_loader, id2tag, device, logger)
            logger.info("测试集结果:")
            logger.info(f"  - 精确率(Precision): {precision:.4f}")
            logger.info(f"  - 召回率(Recall): {recall:.4f}")
            logger.info(f"  - F1分数: {f1:.4f}")
        else:
            logger.info("未提供测试集，改在验证集上评估...")
            precision, recall, f1 = evaluate(model, val_loader, id2tag, device, logger)
            logger.info("验证集结果:")
            logger.info(f"  - 精确率(Precision): {precision:.4f}")
            logger.info(f"  - 召回率(Recall): {recall:.4f}")
            logger.info(f"  - F1分数: {f1:.4f}")
        logger.info("评估结束，未执行训练。")
        logger.info(f"\n所有训练记录已保存到: {log_dir}")
        return

    # 打印训练参数
    logger.info("="*80)
    logger.info("训练参数:")
    logger.info(f"  - 学习率: {args.lr}")
    logger.info(f"  - 批次大小: {args.batch_size}")
    logger.info(f"  - 最大序列长度: {args.max_length}")
    logger.info(f"  - 权重衰减: {args.weight_decay}")
    logger.info(f"  - 预热比例: {args.warmup_ratio}")
    logger.info(f"  - 训练轮数: {args.num_epoch}")
    logger.info(f"  - 梯度裁剪: {args.max_grad_norm}")
    logger.info(f"  - 模型保存路径: {args.save_path}")
    logger.info("="*80)

    num_training_steps = len(train_loader) * args.num_epoch
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    best_f1 = 0.0
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    logger.info(f"开始训练，总共 {args.num_epoch} 个epoch，每个epoch有 {len(train_loader)} 个batch")
    logger.info("-" * 80)

    for epoch in range(args.num_epoch):
        epoch_start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, args.max_grad_norm, epoch, args.num_epoch, logger
        )
        precision, recall, f1 = evaluate(model, val_loader, id2tag, device, logger)

        epoch_time = time.time() - epoch_start_time

        logger.info(f'\nEpoch {epoch+1}/{args.num_epoch} 完成')
        logger.info(f'  - 平均损失: {train_loss:.4f}')
        logger.info(f'  - 训练时间: {epoch_time:.2f}秒')
        logger.info(f'  - 精确率(Precision): {precision:.4f}')
        logger.info(f'  - 召回率(Recall): {recall:.4f}')
        logger.info(f'  - F1分数: {f1:.4f}')

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.save_path)
            logger.info(f'  - 新的最佳F1分数: {best_f1:.4f}，模型已保存到 {args.save_path}')
        else:
            logger.info(f'  - 当前最佳F1分数: {best_f1:.4f}')

        logger.info("-" * 80)

    logger.info(f'\n训练完成！最佳F1分数: {best_f1:.4f}')

    # 可选：加载最佳权重并在测试集上评估
    if args.test:
        if not os.path.exists(args.test):
            logger.warning(f"测试集不存在，跳过测试: {args.test}")
        elif not os.path.exists(args.save_path):
            logger.warning(f"未找到已保存的最佳模型，跳过测试: {args.save_path}")
        else:
            logger.info("加载最佳模型权重并在测试集上评估...")
            model.load_state_dict(torch.load(args.save_path, map_location=device))
            model.to(device)
            test_ds = HFNERDataset(args.test, tokenizer, args.tag_vocab, args.max_length)
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator
            )
            test_precision, test_recall, test_f1 = evaluate(model, test_loader, id2tag, device, logger)
            logger.info("测试集结果:")
            logger.info(f"  - 精确率(Precision): {test_precision:.4f}")
            logger.info(f"  - 召回率(Recall): {test_recall:.4f}")
            logger.info(f"  - F1分数: {test_f1:.4f}")

    logger.info(f"\n所有训练记录已保存到: {log_dir}")


if __name__ == "__main__":
    print("Starting training...")
    main()

