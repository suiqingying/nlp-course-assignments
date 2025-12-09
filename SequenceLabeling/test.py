import os
import json
import argparse
import torch
import time
import logging
from collections import namedtuple

# 从 train.py 复用组件
from train import BiLSTM_CRF, NER_dataset, collate_fn, evaluate

def build_config(chr_vocab_path, tag_vocab_path, hidden_dim, embedding_dim, num_layers, dropout):
    chr_vocab = json.load(open(chr_vocab_path, 'r', encoding='utf-8'))
    tag_vocab = json.load(open(tag_vocab_path, 'r', encoding='utf-8'))
    config = {
        "vocab_size": len(chr_vocab),
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "tag_size": len(tag_vocab),
        "gpu": torch.cuda.is_available()
    }
    return namedtuple('Config', config.keys())(*config.values()), tag_vocab

def main():
    parser = argparse.ArgumentParser(description='BiLSTM-CRF模型测试')
    parser.add_argument('--model_path', default='save_model/best.pt', help='模型文件路径')
    parser.add_argument('--test', default='inference/test.txt', help='测试数据路径')
    parser.add_argument('--chr_vocab', default='inference/chr_vocab.json', help='字符词汇表路径')
    parser.add_argument('--tag_vocab', default='inference/tag_vocab.json', help='标签词汇表路径')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--hidden_dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--embedding_dim', type=int, default=256, help='词嵌入维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--device', default='auto', help='设备选择 (auto/cpu/cuda)')
    args = parser.parse_args()

    # 设置日志
    log_dir = os.path.join("logs", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'test.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # 检查模型文件
    if not os.path.exists(args.model_path):
        logger.error(f"模型文件不存在: {args.model_path}")
        return

    # 设置设备
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"使用设备: {device}")

    # 打开词汇表
    logger.info("正在加载词汇表...")
    config, raw_tag_vocab = build_config(
        args.chr_vocab, args.tag_vocab, 
        args.hidden_dim, args.embedding_dim, args.num_layers, args.dropout
    )
    # 构造 id->tag 映射（evaluate 期望）
    id2tag = {int(v): k for k, v in raw_tag_vocab.items()}

    # 加载数据
    logger.info("正在加载测试数据...")
    test_dataset = NER_dataset(args.test, args.chr_vocab, args.tag_vocab)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 创建模型
    logger.info("正在创建模型...")
    model = BiLSTM_CRF(config)
    model.to(device)

    # 加载模型权重
    logger.info(f"正在加载模型权重: {args.model_path}")
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    # 显示测试信息
    logger.info("="*80)
    logger.info("测试信息:")
    logger.info(f"  - 模型路径: {args.model_path}")
    logger.info(f"  - 测试数据路径: {args.test}")
    logger.info(f"  - 测试样本数: {len(test_dataset)}")
    logger.info(f"  - 批次大小: {args.batch_size}")
    logger.info(f"  - 隐藏层维度: {args.hidden_dim}")
    logger.info(f"  - 词嵌入维度: {args.embedding_dim}")
    logger.info(f"  - LSTM层数: {args.num_layers}")
    logger.info(f"  - Dropout: {args.dropout}")
    logger.info(f"  - 使用设备: {device}")
    logger.info("="*80)

    # 开始测试
    logger.info("\n开始测试...")
    start_time = time.time()

    # 复用 evaluate（会打印并返回 f1）
    f1 = evaluate(model, test_loader, id2tag)

    test_time = time.time() - start_time

    logger.info(f"\n测试完成!")
    logger.info(f"  - 测试时间: {test_time:.2f}秒")
    logger.info(f"  - Test F1: {f1:.4f}")

    logger.info(f"\n测试记录已保存到: {log_dir}")

if __name__ == '__main__':
    main()