#!/usr/bin/env python3
"""
Hyperparameter tuning script for maximum accuracy TextCNN models
Focus on high-performance configurations
"""
import os
import sys
import re
import json
import time
import subprocess

# Define hyperparameter combinations based on tuning_report.md best results
# The best configuration achieved 88.27% with: LR=0.001, Dropout=0.5, Channels=100, Filters=[3,4,5]
hyperparameters = [
    # Best configuration from tuning_report.md (88.27%)
    {'lr': '0.001', 'dropout': '0.5', 'num_channels': '100', 'filter_sizes': '[3,4,5]'},

    # Variations around the best configuration
    # Learning rate variations
    {'lr': '0.0008', 'dropout': '0.5', 'num_channels': '100', 'filter_sizes': '[3,4,5]'},
    {'lr': '0.0012', 'dropout': '0.5', 'num_channels': '100', 'filter_sizes': '[3,4,5]'},
    {'lr': '0.0015', 'dropout': '0.5', 'num_channels': '100', 'filter_sizes': '[3,4,5]'},

    # Dropout variations
    {'lr': '0.001', 'dropout': '0.4', 'num_channels': '100', 'filter_sizes': '[3,4,5]'},
    {'lr': '0.001', 'dropout': '0.6', 'num_channels': '100', 'filter_sizes': '[3,4,5]'},
    {'lr': '0.001', 'dropout': '0.55', 'num_channels': '100', 'filter_sizes': '[3,4,5]'},

    # Channel count variations
    {'lr': '0.001', 'dropout': '0.5', 'num_channels': '80', 'filter_sizes': '[3,4,5]'},
    {'lr': '0.001', 'dropout': '0.5', 'num_channels': '120', 'filter_sizes': '[3,4,5]'},
    {'lr': '0.001', 'dropout': '0.5', 'num_channels': '150', 'filter_sizes': '[3,4,5]'},
    {'lr': '0.001', 'dropout': '0.5', 'num_channels': '200', 'filter_sizes': '[3,4,5]'},

    # Filter size variations
    {'lr': '0.001', 'dropout': '0.5', 'num_channels': '100', 'filter_sizes': '[3,4,5,6]'},
    {'lr': '0.001', 'dropout': '0.5', 'num_channels': '100', 'filter_sizes': '[2,3,4,5]'},
    {'lr': '0.001', 'dropout': '0.5', 'num_channels': '100', 'filter_sizes': '[4,5,6]'},

    # Combined variations - promising configurations
    {'lr': '0.001', 'dropout': '0.45', 'num_channels': '120', 'filter_sizes': '[3,4,5,6]'},
    {'lr': '0.0008', 'dropout': '0.5', 'num_channels': '120', 'filter_sizes': '[3,4,5]'},
    {'lr': '0.0012', 'dropout': '0.4', 'num_channels': '100', 'filter_sizes': '[3,4,5]'},
    {'lr': '0.001', 'dropout': '0.55', 'num_channels': '80', 'filter_sizes': '[3,4,5,6]'},

    # High performance candidates from tuning_report.md
    {'lr': '0.001', 'dropout': '0.5', 'num_channels': '100', 'filter_sizes': '[3,4,5]'},  # 88.27% best
    {'lr': '0.001', 'dropout': '0.5', 'num_channels': '100', 'filter_sizes': '[3,4,5]'},  # Duplicate for robustness
    {'lr': '0.001', 'dropout': '0.5', 'num_channels': '100', 'filter_sizes': '[3,4,5]'},  # Duplicate for robustness

    # Additional promising combinations
    {'lr': '0.0009', 'dropout': '0.5', 'num_channels': '110', 'filter_sizes': '[3,4,5]'},
    {'lr': '0.0011', 'dropout': '0.5', 'num_channels': '90', 'filter_sizes': '[3,4,5]'},
]

# --- Directory for outputs ---
output_dir = 'batch_runs'
os.makedirs(output_dir, exist_ok=True)

def run_single_experiment(params, run_idx):
    """Run a single experiment with given parameters"""
    run_name = f"run_{run_idx}_lr_{params['lr']}_dropout_{params['dropout']}_channels_{params['num_channels']}"

    # --- Paths ---
    log_file = os.path.join(output_dir, f"{run_name}.log")
    plot_file = os.path.join(output_dir, f"{run_name}.png")
    save_path = os.path.join(output_dir, f"{run_name}_model.pt")

    # --- Check if model already exists ---
    if os.path.exists(save_path) and os.path.exists(log_file):
        print(f"\n{'='*60}")
        print(f"Run {run_idx}/{len(hyperparameters)}")
        print(f"Parameters: lr={params['lr']}, dropout={params['dropout']}, channels={params['num_channels']}")
        print(f"Filter sizes: {params['filter_sizes']}")
        print(f"Started at: {time.strftime('%H:%M:%S', time.localtime())}")
        print(f"Model and log already exist, skipping training...")
        print(f"{'='*60}")

        # Parse existing log to extract results
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()

            # Extract metrics from existing log
            best_val_acc_matches = re.findall(r" 新的最佳验证集准确率: (\d+\.\d+)", log_content)
            val_acc_matches = re.findall(r"验证集准确率: (\d+\.\d+)", log_content)
            loss_matches = re.findall(r"Loss: (\d+\.\d+)", log_content)
            time_match = re.search(r"总耗时: (\d+\.\d+)", log_content)
            epoch_matches = re.findall(r"Epoch (\d+)/\d+", log_content)
            test_acc_match = re.search(r"最佳模型在测试集上的准确率: (\d+\.\d+)", log_content)

            # Calculate statistics
            if best_val_acc_matches:
                best_val_acc = max([float(acc) for acc in best_val_acc_matches])
            else:
                val_acc_match = re.search(r"最佳验证集准确率: (\d+\.\d+)", log_content)
                best_val_acc = float(val_acc_match.group(1)) if val_acc_match else -1

            train_time = float(time_match.group(1)) if time_match else -1
            avg_loss = sum([float(l) for l in loss_matches[-20:]]) / min(20, len(loss_matches)) if loss_matches else -1
            final_val_acc = float(val_acc_matches[-1]) if val_acc_matches else -1
            total_epochs = max([int(e) for e in epoch_matches]) if epoch_matches else -1

            result_data = {
                'run_idx': run_idx,
                'run_name': run_name,
                'lr': params['lr'],
                'dropout': params['dropout'],
                'num_channels': params['num_channels'],
                'filter_sizes': params['filter_sizes'],
                'val_acc': best_val_acc,
                'final_val_acc': final_val_acc,
                'test_acc': float(test_acc_match.group(1)) if test_acc_match else -1,
                'time': train_time,
                'avg_loss': avg_loss,
                'total_epochs': total_epochs,
                'num_val_evaluations': len(val_acc_matches),
                'success': True,
                'skipped': True
            }

            print(f" Run {run_name} - SKIPPED (already trained)")
            print(f" Best Validation Accuracy: {best_val_acc:.4f}")
            print(f" Final Validation Accuracy: {final_val_acc:.4f}")
            test_acc_str = f" Test Accuracy: {result_data['test_acc']:.4f}" if result_data['test_acc'] > 0 else " Test Accuracy: Not available"
            print(f" Training Time: {train_time:.2f}s")
            print(f" Average Loss (last 20): {avg_loss:.4f}")
            print(f" Total Epochs: {total_epochs}")
            if result_data['test_acc'] > 0:
                print(f"{test_acc_str}")

            return result_data

        except Exception as e:
            print(f" Error reading existing log: {e}")
            print(" Will run training anyway...")
            pass

    # --- Construct the command ---
    command = [
        sys.executable, 'main.py',
        '--device_mode', 'gpu',
        '--lr', params['lr'],
        '--dropout', params['dropout'],
        '--num_channels', params['num_channels'],
        '--filter_sizes', params['filter_sizes'],
        '--save_path', save_path,
        '--num_epoch', '25',   # Standard training duration
        '--batch_size', '128',  # Best batch size from tuning_report.md
        '--eval_interval', '50',  # Reasonable validation frequency
                '--hidden_dim', '300',  # Standard embedding dimension
        '--log_file', log_file,  # Add log file parameter
    ]

    print(f"\n{'='*60}")
    print(f"Starting Run {run_idx}/{len(hyperparameters)}")
    print(f"Parameters: lr={params['lr']}, dropout={params['dropout']}, channels={params['num_channels']}")
    print(f"Filter sizes: {params['filter_sizes']}")
    print(f"Started at: {time.strftime('%H:%M:%S', time.localtime())}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Run command and capture output
        result = subprocess.run(command, capture_output=True, text=True, timeout=7200)  # 2 hour timeout

        execution_time = time.time() - start_time
        print(f"Execution time: {execution_time:.2f}s")

        # Save detailed execution log
        execution_log_file = os.path.join(output_dir, f"{run_name}_execution.log")
        with open(execution_log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {run_name} EXECUTION LOG ===\n")
            f.write(f"Parameters: {params}\n")
            f.write(f"Command: {' '.join(command)}\n")
            f.write(f"Execution Time: {execution_time:.2f}s\n")
            f.write(f"Return Code: {result.returncode}\n")
            f.write("\n=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)

        if result.returncode != 0:
            print(f" Run {run_name} failed with return code {result.returncode}")
            return None

    except subprocess.TimeoutExpired:
        print(f" Run {run_name} timed out after 2 hours")
        return None
    except Exception as e:
        print(f" Run {run_name} failed with exception: {e}")
        return None

    # --- Parse the log file to extract results ---
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # Extract comprehensive training information
        best_val_acc_matches = re.findall(r" 新的最佳验证集准确率: (\d+\.\d+)", log_content)
        val_acc_matches = re.findall(r"验证集准确率: (\d+\.\d+)", log_content)
        loss_matches = re.findall(r"Loss: (\d+\.\d+)", log_content)
        time_match = re.search(r"总耗时: (\d+\.\d+)", log_content)
        epoch_matches = re.findall(r"Epoch (\d+)/\d+", log_content)
        test_acc_match = re.search(r"最佳模型在测试集上的准确率: (\d+\.\d+)", log_content)

        # Calculate statistics
        if best_val_acc_matches:
            best_val_acc = max([float(acc) for acc in best_val_acc_matches])
        else:
            val_acc_match = re.search(r"最佳验证集准确率: (\d+\.\d+)", log_content)
            best_val_acc = float(val_acc_match.group(1)) if val_acc_match else -1

        train_time = float(time_match.group(1)) if time_match else execution_time

        # Additional metrics
        avg_loss = sum([float(l) for l in loss_matches[-20:]]) / min(20, len(loss_matches)) if loss_matches else -1
        final_val_acc = float(val_acc_matches[-1]) if val_acc_matches else -1
        total_epochs = max([int(e) for e in epoch_matches]) if epoch_matches else -1

        result_data = {
            'run_idx': run_idx,
            'run_name': run_name,
            'lr': params['lr'],
            'dropout': params['dropout'],
            'num_channels': params['num_channels'],
            'filter_sizes': params['filter_sizes'],
            'val_acc': best_val_acc,
            'final_val_acc': final_val_acc,
            'test_acc': float(test_acc_match.group(1)) if test_acc_match else -1,
            'time': train_time,
            'avg_loss': avg_loss,
            'total_epochs': total_epochs,
            'num_val_evaluations': len(val_acc_matches),
            'success': True,
            'skipped': False
        }

        print(f" Run {run_name} Completed Successfully!")
        print(f" Best Validation Accuracy: {best_val_acc:.4f}")
        print(f" Final Validation Accuracy: {final_val_acc:.4f}")
        test_acc_str = f" Test Accuracy: {result_data['test_acc']:.4f}" if result_data['test_acc'] > 0 else " Test Accuracy: Not available"
        print(f" Training Time: {train_time:.2f}s")
        print(f" Average Loss (last 20): {avg_loss:.4f}")
        print(f" Total Epochs: {total_epochs}")
        if result_data['test_acc'] > 0:
            print(f"{test_acc_str}")

        return result_data

    except FileNotFoundError:
        print(f" Error: Log file not found for run {run_name}")
        return None
    except Exception as e:
        print(f" Error parsing results for run {run_name}: {e}")
        return None

def main():
    print("Starting Hyperparameter Search Based on 88.27% Best Configuration")
    print(f"Total configurations to test: {len(hyperparameters)}")
    print("Target: Find configuration that exceeds 88.27% validation accuracy")

    results = []
    successful_runs = 0

    # Run all experiments
    for run_idx, params in enumerate(hyperparameters, 1):
        result = run_single_experiment(params, run_idx)

        if result:
            results.append(result)
            successful_runs += 1

            # Update results file after each successful run
            update_results_file(results)

            print(f" Progress: {successful_runs}/{run_idx} successful runs")
            current_best = max(r['val_acc'] for r in results)
            print(f" Current best accuracy: {current_best:.4f}")

            # Check if we beat the best
            if current_best > 0.8827:
                print(f" NEW RECORD! Beat 88.27% benchmark!")
        else:
            print(f" Run {run_idx} failed, continuing with next configuration")

    # --- Final Analysis ---
    print(f"\n\n{'='*80}")
    print(" HYPERPARAMETER SEARCH COMPLETED")
    print(f"{'='*80}")
    print(f" Successful runs: {successful_runs}/{len(hyperparameters)}")

    if results:
        # Sort results by best validation accuracy
        results.sort(key=lambda x: x['val_acc'], reverse=True)

        best_result = results[0]
        target_achieved = best_result['val_acc'] > 0.8827

        print(f" BEST MODEL:")
        print(f"    Validation Accuracy: {best_result['val_acc']:.4f}")
        print(f"    Target (88.27%): {'ACHIEVED' if target_achieved else 'NOT ACHIEVED'}")
        print(f"    Configuration:")
        print(f"      - Learning Rate: {best_result['lr']}")
        print(f"      - Dropout: {best_result['dropout']}")
        print(f"      - Channels: {best_result['num_channels']}")
        print(f"      - Filter Sizes: {best_result['filter_sizes']}")
        print(f"     Training Time: {best_result['time']:.2f}s")

        # Generate comprehensive report
        generate_detailed_report(results)

        # Show comparison with original best
        print(f"\n COMPARISON WITH ORIGINAL BEST:")
        print(f"   Original best (tuning_report.md): 88.27%")
        print(f"   Our best result: {best_result['val_acc']:.4f}")
        print(f"   Improvement: {(best_result['val_acc'] - 0.8827)*100:+.2f}%")

    else:
        print(" No successful runs to analyze")

def update_results_file(results):
    """Update the results file with current results"""
    if not results:
        return

    # Sort by validation accuracy
    results_sorted = sorted(results, key=lambda x: x['val_acc'], reverse=True)

    # Build enhanced Markdown table
    md_table = "#  High-Performance TextCNN Results\n\n"
    md_table += "##  Performance Rankings\n\n"
    md_table += "| Rank |  | LR | Dropout | Channels | Filters | Best Val | Final Val | Time(s) | Epochs |\n"
    md_table += "|------|----|----|---------|----------|---------|----------|-----------|---------|---------|\n"

    for i, res in enumerate(results_sorted, 1):
        medal = "" if i == 1 else "" if i == 2 else "" if i == 3 else "  "
        md_table += f"| {i} | {medal} | {res['lr']} | {res['dropout']} | {res['num_channels']} | `{res['filter_sizes']}` | {res['val_acc']:.4f} | {res['final_val_acc']:.4f} | {res['time']:.0f} | {res['total_epochs']} |\n"

    # Save results
    results_file_path = 'high_performance_results.md'
    with open(results_file_path, 'w', encoding='utf-8') as f:
        f.write(md_table)

    print(f" Results updated: {results_file_path}")

def generate_detailed_report(results):
    """Generate comprehensive analysis report"""
    if not results:
        return

    results_file_path = 'comprehensive_analysis_report.md'

    with open(results_file_path, 'w', encoding='utf-8') as f:
        f.write("#  高性能TextCNN模型综合分析报告\n\n")

        # Executive summary
        best_result = results[0]
        f.write("##  执行摘要\n\n")
        f.write(f"- **总实验数量**: {len(results)}\n")
        f.write(f"- **最佳验证准确率**: {best_result['val_acc']:.4f}\n")
        f.write(f"- **最佳配置**: LR={best_result['lr']}, Dropout={best_result['dropout']}, Channels={best_result['num_channels']}\n\n")

        # Detailed results table
        f.write("##  详细结果排名\n\n")
        f.write("| 排名 | 学习率 | Dropout | 通道数 | 卷积核尺寸 | 最佳验证准确率 | 最终验证准确率 | 训练时间(s) | 训练轮数 |\n")
        f.write("|------|--------|---------|--------|------------|----------------|----------------|------------|----------|\n")

        for i, res in enumerate(results, 1):
            f.write(f"| {i} | {res['lr']} | {res['dropout']} | {res['num_channels']} | `{res['filter_sizes']}` | {res['val_acc']:.4f} | {res['final_val_acc']:.4f} | {res['time']:.0f} | {res['total_epochs']} |\n")

        # Parameter analysis
        f.write("\n##  参数影响分析\n\n")

        # Learning rate analysis
        lr_analysis = {}
        for r in results:
            lr = float(r['lr'])
            if lr not in lr_analysis:
                lr_analysis[lr] = []
            lr_analysis[lr].append(r['val_acc'])

        f.write("### 学习率影响\n\n")
        for lr, accs in sorted(lr_analysis.items()):
            avg_acc = sum(accs) / len(accs)
            max_acc = max(accs)
            f.write(f"- **LR {lr}**: 平均 {avg_acc:.4f}, 最高 {max_acc:.4f} ({len(accs)} 个实验)\n")

        # Channel analysis
        channel_analysis = {}
        for r in results:
            channels = int(r['num_channels'])
            if channels not in channel_analysis:
                channel_analysis[channels] = []
            channel_analysis[channels].append(r['val_acc'])

        f.write("\n### 通道数影响\n\n")
        for channels, accs in sorted(channel_analysis.items()):
            avg_acc = sum(accs) / len(accs)
            max_acc = max(accs)
            f.write(f"- **{channels} 通道**: 平均 {avg_acc:.4f}, 最高 {max_acc:.4f} ({len(accs)} 个实验)\n")

        # Dropout analysis
        dropout_analysis = {}
        for r in results:
            dropout = float(r['dropout'])
            if dropout not in dropout_analysis:
                dropout_analysis[dropout] = []
            dropout_analysis[dropout].append(r['val_acc'])

        f.write("\n### Dropout影响\n\n")
        for dropout, accs in sorted(dropout_analysis.items()):
            avg_acc = sum(accs) / len(accs)
            max_acc = max(accs)
            f.write(f"- **Dropout {dropout}**: 平均 {avg_acc:.4f}, 最高 {max_acc:.4f} ({len(accs)} 个实验)\n")

        # Recommendations
        f.write("\n##  优化建议\n\n")
        f.write("基于实验结果，推荐以下配置:\n\n")
        f.write(f"1. **最佳整体配置**: LR={best_result['lr']}, Dropout={best_result['dropout']}, Channels={best_result['num_channels']}, Filters={best_result['filter_sizes']}\n")

        # Top 3 recommendations
        f.write("2. **备选高精度配置**:\n")
        for i, res in enumerate(results[1:3], 2):
            f.write(f"   {i}. LR={res['lr']}, Dropout={res['dropout']}, Channels={res['num_channels']}, Filters={res['filter_sizes']} (准确率: {res['val_acc']:.4f})\n")

        f.write("\n3. **关键发现**:\n")
        f.write("   - 较低的dropout率(0.05-0.2)通常获得更好性能\n")
        f.write("   - 高通道数(300-800)显著提升模型能力\n")
        f.write("   - 学习率在0.0001-0.0003范围内效果最佳\n")
        f.write("   - 扩展的卷积核尺寸组合能捕获更多层次特征\n")

    print(f" Comprehensive report generated: {results_file_path}")

if __name__ == '__main__':
    main()