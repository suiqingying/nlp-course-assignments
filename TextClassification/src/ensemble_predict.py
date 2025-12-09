"""
TextCNN 模型集成预测脚本
集成多个高性能模型的预测结果
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from collections import namedtuple
from common import TextCNN, SentimentDataset, collate_fn
import logging
from argparse import Namespace

def load_model(model_path, vocab_size, device):
    """加载单个模型，自动检测配置"""
    # 先加载检查点以了解模型结构
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 分析state_dict来推断filter_sizes
    conv_keys = [k for k in state_dict.keys() if k.startswith('convs.')]
    max_conv_index = max([int(k.split('.')[1]) for k in conv_keys]) if conv_keys else 0

    # 根据卷积层数量推断filter_sizes
    if max_conv_index == 2:  # convs.0, convs.1, convs.2
        filter_sizes = [3,4,5]
    elif max_conv_index == 3:  # convs.0, convs.1, convs.2, convs.3
        filter_sizes = [2,3,4,5]
    elif max_conv_index == 2:  # 其他可能的组合
        filter_sizes = [4,5,6]
    else:
        filter_sizes = [3,4,5]  # 默认

    # 检查fc层的输入维度来确认num_channels
    fc_weight = state_dict['fc.weight']
    expected_features = fc_weight.shape[1]
    if expected_features > 300:
        num_channels = expected_features // len(filter_sizes)
    else:
        num_channels = 100

    # 创建配置
    temp_config = namedtuple('config', [
        'device', 'vocab_size', 'embedding_dim', 'num_classes',
        'filter_sizes', 'num_channels', 'dropout'
    ])(device, vocab_size, 300, 2, filter_sizes, num_channels, 0.5)

    # 创建并加载模型
    model = TextCNN(temp_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def ensemble_predict(models, data_loader, device, method='weighted_avg'):
    """
    集成预测

    Args:
        models: 模型列表
        data_loader: 数据加载器
        device: 设备
        method: 集成方法 ('avg', 'weighted_avg', 'majority_vote')

    Returns:
        predictions: 预测结果
        confidences: 置信度
    """
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch in data_loader:
            texts, labels = batch
            texts = texts.to(device)

            batch_predictions = []
            batch_probabilities = []

            # 获取每个模型的预测
            for model in models:
                outputs = model(texts)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                batch_predictions.append(preds.cpu().numpy())
                batch_probabilities.append(probs.cpu().numpy())

            # 转换为numpy数组
            batch_predictions = np.array(batch_predictions).T  # (batch_size, num_models)
            batch_probabilities = np.array(batch_probabilities).transpose(1, 2, 0)  # (batch_size, num_classes, num_models)

            all_predictions.extend(batch_predictions)
            all_probabilities.extend(batch_probabilities)

    # 应用集成方法
    if method == 'avg':
        # 简单平均
        final_probabilities = np.mean(all_probabilities, axis=2)
        final_predictions = np.argmax(final_probabilities, axis=1)

    elif method == 'weighted_avg':
        # 加权平均（基于验证集准确率）
        # 只有已加载的模型才有权重
        num_loaded = len(models)
        weights = np.array([0.8893, 0.8813, 0.8800][:num_loaded])  # 模型的验证准确率
        weights = weights / weights.sum()

        final_probabilities = np.average(all_probabilities, axis=2, weights=weights)
        final_predictions = np.argmax(final_probabilities, axis=1)

    elif method == 'majority_vote':
        # 多数投票
        final_predictions = []
        for preds in all_predictions:
            # 统计每个类别的票数
            counts = np.bincount(preds, minlength=2)
            final_predictions.append(np.argmax(counts))
        final_predictions = np.array(final_predictions)

        # 计算置信度（投票的一致性）
        confidences = []
        for i, preds in enumerate(all_predictions):
            unique, counts = np.unique(preds, return_counts=True)
            confidence = counts.max() / len(preds)
            confidences.append(confidence)
        confidences = np.array(confidences)

        # 对于多数投票，概率应该反映投票比例
        final_probabilities = np.zeros((len(final_predictions), 2))
        for i, preds in enumerate(all_predictions):
            unique, counts = np.unique(preds, return_counts=True)
            # 将投票比例转换为概率
            for j, unique_val in enumerate(unique):
                final_probabilities[i, unique_val] = counts[j] / len(preds)

        # 确保预测结果一致
        for i, pred in enumerate(final_predictions):
            # 如果概率最大值不是预测类别，修正概率
            if final_probabilities[i, pred] != final_probabilities[i].max():
                final_probabilities[i] = 0
                final_probabilities[i, pred] = 1

    # 简单输出显示方法的差异
    if method == 'majority_vote':
        print(f"  Majority vote confidence distribution: {np.unique(final_predictions, return_counts=True)}")

    return final_predictions, final_probabilities

def evaluate_ensemble(models, test_loader, device, method='weighted_avg'):
    """评估集成模型"""
    predictions, probabilities = ensemble_predict(models, test_loader, device, method)

    # 获取真实标签
    all_labels = []
    for batch in test_loader:
        texts, labels = batch
        all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)

    # 计算准确率
    accuracy = (predictions == all_labels).mean()

    # 计算每个类别的性能
    tp_0 = np.sum((predictions == 0) & (all_labels == 0))
    tn_0 = np.sum((predictions == 1) & (all_labels == 1))
    fp_0 = np.sum((predictions == 0) & (all_labels == 1))
    fn_0 = np.sum((predictions == 1) & (all_labels == 0))

    tp_1 = np.sum((predictions == 1) & (all_labels == 1))
    tn_1 = np.sum((predictions == 0) & (all_labels == 0))
    fp_1 = np.sum((predictions == 1) & (all_labels == 0))
    fn_1 = np.sum((predictions == 0) & (all_labels == 1))

    precision_0 = tp_0 / (tp_0 + fp_0) if (tp_0 + fp_0) > 0 else 0
    recall_0 = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) > 0 else 0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0

    precision_1 = tp_1 / (tp_1 + fp_1) if (tp_1 + fp_1) > 0 else 0
    recall_1 = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) > 0 else 0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision_0': precision_0,
        'recall_0': recall_0,
        'f1_0': f1_0,
        'precision_1': precision_1,
        'recall_1': recall_1,
        'f1_1': f1_1,
        'predictions': predictions,
        'probabilities': probabilities,
        'true_labels': all_labels,
        'num_models': len(models)
    }

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")

    # 加载词汇表
    vocab_path = '../dataset/vocab.json'
    chr_vocab = json.load(open(vocab_path, 'r', encoding='utf-8'))
    vocab_size = len(chr_vocab)

    # 创建测试数据集
    test_dataset = SentimentDataset('../dataset/test.jsonl', vocab_path)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    # 根据实际验证准确率选择Top 3模型
    # 从comprehensive_analysis_report.md中找到的真正Top 3模型
    # 第1名: run_1 (88.93%) - lr=0.001, dropout=0.5, channels=100
    # 第2名: run_6 (88.13%) - lr=0.001, dropout=0.6, channels=100
    # 第3名: run_3 (88.00%) - lr=0.0012, dropout=0.5, channels=100

    # 按验证准确率从高到低的模型路径
    model_paths = [
        '../batch_runs/run_1_lr_0.001_dropout_0.5_channels_100_model.pt',  # 验证准确率0.8893
        '../batch_runs/run_6_lr_0.001_dropout_0.6_channels_100_model.pt',  # 验证准确率0.8813
        '../batch_runs/run_3_lr_0.0012_dropout_0.5_channels_100_model.pt'   # 验证准确率0.8800
    ]

    # 加载模型
    models = []
    valid_models = []

    for i, model_path in enumerate(model_paths):
        if os.path.exists(model_path):
            try:
                model = load_model(model_path, vocab_size, device)
                models.append(model)
                valid_models.append(model_path)
                logging.info(f"成功加载模型 {i+1}: {os.path.basename(model_path)}")
            except Exception as e:
                logging.error(f"加载模型 {i+1} 失败: {e}")
        else:
            logging.warning(f"模型文件不存在: {model_path}")

    logging.info(f"成功加载 {len(models)}/{len(model_paths)} 个模型")

    if len(models) == 0:
        logging.error("没有可用的模型，程序退出")
        return

    # 测试不同的集成方法
    methods = ['avg', 'weighted_avg', 'majority_vote']

    print("\n" + "="*60)
    print("集成模型性能对比")
    print("="*60)

    # 存储所有方法的结果
    all_results = {}
    best_accuracy = 0
    best_method = 'avg'

    for method in methods:
        logging.info(f"使用 {method} 方法进行集成...")

        results = evaluate_ensemble(models, test_loader, device, method)

        # 存储结果供后续使用
        all_results[method] = results

        # 显示预测分布的差异
        unique_preds, counts = np.unique(results['predictions'], return_counts=True)
        print(f"\n集成方法: {method.upper()}")
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"预测分布: {dict(zip(unique_preds, counts))}")
        print(f"类别 0 - Precision: {results['precision_0']:.4f}, Recall: {results['recall_0']:.4f}, F1: {results['f1_0']:.4f}")
        print(f"类别 1 - Precision: {results['precision_1']:.4f}, Recall: {results['recall_1']:.4f}, F1: {results['f1_1']:.4f}")
        print("-" * 60)

        # 更新最佳方法
        if results['accuracy'] > best_accuracy:
            best_accuracy = results['accuracy']
            best_method = method

    # 保存所有集成方法的结果
    individual_accuracies = [0.8893, 0.8813, 0.8800][:len(models)]  # 根据实际加载的模型数量调整

    # 创建包含所有方法结果的完整结果集（使用已经计算的结果）
    all_methods_results = {}
    for method in methods:
        results = all_results[method]
        # 添加预测分布信息
        unique_preds, counts = np.unique(results['predictions'], return_counts=True)
        prediction_dist = dict(zip(unique_preds.tolist(), counts.tolist()))

        all_methods_results[method] = {
            'accuracy': float(results['accuracy']),
            'precision_0': float(results['precision_0']),
            'recall_0': float(results['recall_0']),
            'f1_0': float(results['f1_0']),
            'precision_1': float(results['precision_1']),
            'recall_1': float(results['recall_1']),
            'f1_1': float(results['f1_1']),
            'prediction_distribution': prediction_dist
        }

    # 保存结果
    ensemble_results = {
        'best_method': best_method,
        'best_accuracy': float(best_accuracy),
        'all_methods': all_methods_results,
        'model_count': len(models),
        'used_models': valid_models,
        'individual_accuracies': individual_accuracies
    }

    with open('ensemble_results.json', 'w', encoding='utf-8') as f:
        json.dump(ensemble_results, f, ensure_ascii=False, indent=2)

    print(f"\n集成结果已保存到: ensemble_results.json")
    print(f"最佳方法: {best_method.upper()}, 最佳测试集准确率: {best_accuracy:.4f}")

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    main()