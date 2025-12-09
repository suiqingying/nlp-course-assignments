import torch
from torch.utils.data import DataLoader
import time
import logging
import os
import argparse
from collections import namedtuple
import json

from common import SentimentDataset, collate_fn, TextCNN, evaluate, train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='../save_model/best.pt')
    parser.add_argument('--train', default='../dataset/train.jsonl')
    parser.add_argument('--test', default='../dataset/test.jsonl')
    parser.add_argument('--val', default='../dataset/val.jsonl')
    parser.add_argument('--num_epoch', default=25, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate - Best: 0.001')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size - Best: 128')
    parser.add_argument('--eval_interval', default='50', type=str, help='Validation interval: "epoch" or number of steps (e.g., 50)')
    parser.add_argument('--vocab', default='../dataset/vocab.json')
    parser.add_argument('--hidden_dim', default=300, type=int, help='Embedding dimension')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout rate - Best: 0.5')
    parser.add_argument('--filter_sizes', default='[3,4,5]', type=str, help='Filter sizes - Best: [3,4,5]')
    parser.add_argument('--num_channels', default=100, type=int, help='Number of output channels for each CNN filter - Best: 100')
    parser.add_argument('--log_file', default='training.log', help='Path to save the log file.')
    parser.add_argument('--plot_file', default='training_comparison.png', help='Path to save the training plot.')
    parser.add_argument('--mode', default='train', choices=['train', 'eval'], help='Mode: train or eval')
    parser.add_argument('--device_mode', default='auto', choices=['auto', 'cpu', 'gpu'], help='Device mode: auto (use GPU if available), cpu only, or gpu only')
    parser.add_argument('--log_steps', type=int, help='Number of steps between logging')
    arg = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(arg.log_file, mode='w', encoding='utf-8'),  # 添加 encoding 参数
            logging.StreamHandler()
        ]
    )

    train_dataset = SentimentDataset(arg.train, arg.vocab)
    val_dataset = SentimentDataset(arg.val, arg.vocab)
    test_dataset = SentimentDataset(arg.test, arg.vocab)
    train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=arg.batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    logging.info(f"数据加载完成，训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}, 测试集样本数: {len(test_dataset)}")

    # 根据device_mode参数选择设备
    if arg.device_mode == 'gpu':
        if not torch.cuda.is_available():
            logging.error("GPU模式被指定，但未找到可用的CUDA设备。程序将退出。")
            exit()
        device = torch.device('cuda')
    elif arg.device_mode == 'cpu':
        device = torch.device('cpu')
    else:  # auto mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f"使用设备模式: {arg.device_mode}, 实际设备: {device}")

    # CUDA状态检查
    if device.type == 'cuda':
        logging.info(f"CUDA版本: {torch.version.cuda}")
        logging.info(f"GPU型号: {torch.cuda.get_device_name(0)}")
        logging.info(f"当前GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f}GB 已使用 / {torch.cuda.memory_reserved()/1024**3:.2f}GB 总计")

    # 确保保存目录存在
    save_dir = os.path.dirname(arg.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"创建模型保存目录: {save_dir}")

    chr_vocab = json.load(open(arg.vocab, 'r', encoding='utf-8'))
    # 处理eval_interval和log_steps
    if arg.log_steps:
        log_steps = arg.log_steps
    elif arg.eval_interval == 'epoch':
        log_steps = len(train_dataset) // arg.batch_size  # 每epoch一次
    else:
        log_steps = int(arg.eval_interval)

    config = {
        'dropout': arg.dropout,
        'num_classes': 2,
        'vocab_size': len(chr_vocab),
        'embedding_dim': arg.hidden_dim,
        'filter_sizes': json.loads(arg.filter_sizes),
        'num_channels': arg.num_channels,
        'lr': arg.lr,
        'num_epoch': arg.num_epoch,
        'eval_interval': log_steps,
        'save_path': arg.save_path,
        'device': device,
        'log_steps': log_steps,
        'batch_size': arg.batch_size  # 添加batch_size参数
    }
    config = namedtuple('config', config.keys())(**config)
    logging.info(f"模型配置: {config}")
    model = TextCNN(config)
    logging.info(f"使用设备: {device.type.upper()}")
    if arg.mode == 'train':
        logging.info("--- 开始训练模式 ---")

        results = {}
        logging.info(f"--- 使用 {device.type.upper()} 开始训练 ---")

        model = TextCNN(config)
        model.to(device)

        start_time = time.time()
        train_loss_history, val_acc_history = train(model, config, train_loader, val_loader)
        end_time = time.time()

        training_time = end_time - start_time
        # 获取最佳验证准确率（更符合实际需求）
        final_acc = max(val_acc_history) if val_acc_history else 0

        results[str(device)] = {
            'time': training_time,
            'final_acc': final_acc,
            'train_loss_history': train_loss_history,
            'val_acc_history': val_acc_history
        }

        logging.info(f"--- {device.type.upper()} 训练完成 ---")
        logging.info(f"总耗时: {training_time:.2f} 秒")
        logging.info(f"最终验证集准确率: {final_acc:.4f}")

        logging.info("\n--- 在测试集上评估模型 ---")
        if results:
            best_device_name = max(results, key=lambda k: results[k]['final_acc'])

            # 直接使用保存的模型路径
            if os.path.exists(arg.save_path):
                logging.info(f"加载最佳模型: {arg.save_path}")
                checkpoint = torch.load(arg.save_path, map_location=device)

                # 从检查点中读取原始配置
                if isinstance(checkpoint, dict) and 'config' in checkpoint:
                    original_config_dict = checkpoint['config']
                    # 如果是字典，重建config对象
                    if isinstance(original_config_dict, dict):
                        temp_config = namedtuple('config', original_config_dict.keys())(**original_config_dict)
                    else:
                        temp_config = original_config_dict
                    temp_config = temp_config._replace(device=device)  # 更新device
                    model_state = checkpoint['model_state_dict']
                else:
                    # 如果checkpoint不是字典，直接使用
                    temp_config = namedtuple('config', [
                        'device', 'vocab_size', 'embedding_dim', 'num_classes',
                        'filter_sizes', 'num_channels', 'dropout'
                    ])(device, len(chr_vocab), 300, 2, [3,4,5], 100, 0.5)
                    model_state = checkpoint

                model = TextCNN(temp_config)
                model.load_state_dict(model_state)
                model.to(device)

                test_acc = evaluate(model, test_loader, temp_config)
                logging.info(f"最佳模型在测试集上的准确率: {test_acc:.4f}")
            else:
                logging.warning(f"未找到模型文件: {arg.save_path}，跳过测试集评估。")
        else:
            logging.error("训练结果为空，无法评估最佳模型。")

    elif arg.mode == 'eval':
        logging.info("--- 开始仅评估模式 ---")
        if not os.path.exists(arg.save_path):
            logging.error(f"模型文件未找到: {arg.save_path}，程序将退出。")
            exit()

        logging.info(f"正在从 {arg.save_path} 加载模型")
        checkpoint = torch.load(arg.save_path, map_location=device)
        config = namedtuple('Config', checkpoint['config'].keys())(**checkpoint['config'])
        model = TextCNN(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        logging.info("在测试集上进行评估...")
        # 临时创建config用于评估
        temp_config = namedtuple('config', ['device'])(device)
        test_acc = evaluate(model, test_loader, temp_config)
        logging.info(f"模型在测试集上的准确率为: {test_acc:.4f}")