import time
import json
import random
import os
import re
from anthropic import Anthropic
from sklearn.metrics import accuracy_score

# ================= 配置区域 =================
os.environ["ANTHROPIC_BASE_URL"] = "http://www.claudecodeserver.top/api"
os.environ["ANTHROPIC_API_KEY"] = "sk_317d87cb3cf64fde228486c6d3d397b181eee1c7b42865a3ae5f9e1395f991d3"

API_KEY = "sk_317d87cb3cf64fde228486c6d3d397b181eee1c7b42865a3ae5f9e1395f991d3"
BASE_URL = "http://www.claudecodeserver.top/api"
MODEL_NAME = "claude-sonnet-4-5-20250929"

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 这里填写你的文件名
TRAIN_FILE = os.path.join(SCRIPT_DIR, "train.jsonl")  # 你的训练集文件名
TEST_FILE = os.path.join(SCRIPT_DIR, "test.jsonl")    # 你的测试集文件名

BATCH_SIZE = 100     # 每次 API 调用处理的条数
TEST_LIMIT = 200     # 总共测试条数
FEW_SHOT_NUM = 10   # Few-shot 示例数量
# ===========================================

client = Anthropic(
    api_key=API_KEY, 
    base_url=BASE_URL,
    default_headers={
        "anthropic-version": "2023-06-01",
    }
)

def process_line(json_line):
    """
    处理单行数据：
    1. 拼接分词列表 -> 字符串
    2. 映射数字标签 -> 中文
    """
    item = json.loads(json_line.strip())
    
    # 1. 拼接：["房间", "还", "可以"] -> "房间还可以"
    # 这一步很重要！大模型读列表效果不好，读整句效果才好。
    text_content = "".join(item["text"])
    
    # 2. 映射：0 -> 负向, 1 -> 正向
    # 请根据你的数据实际情况修改，通常 0是负向，1是正向
    label_map = {"0": "负向", "1": "正向"}
    
    # 注意：你的json里label是字符串"0"还是数字0？这里做了兼容处理
    raw_label = str(item["label"]) 
    human_label = label_map.get(raw_label, "未知")
    
    return {"text": text_content, "label": human_label}

def load_dataset(file_path, is_test_file=False):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 打乱顺序，消除位置偏差
    if is_test_file:
        random.shuffle(lines)
        if len(lines) > TEST_LIMIT:
            lines = lines[:TEST_LIMIT]
        
    for line in lines:
        if not line.strip(): continue
        data_list.append(process_line(line))
        
    return data_list

def balance_sample(data_list, num_samples):
    """
    从数据中均衡采样指定数量的样本
    确保正向和负向标签的数量相等或接近
    """
    positive = [d for d in data_list if d['label'] == '正向']
    negative = [d for d in data_list if d['label'] == '负向']
    
    # 每个类别采样的数量
    per_class = num_samples // 2
    
    # 随机采样
    sampled_positive = random.sample(positive, min(per_class, len(positive)))
    sampled_negative = random.sample(negative, min(per_class, len(negative)))
    
    # 合并并打乱
    balanced_samples = sampled_positive + sampled_negative
    random.shuffle(balanced_samples)
    
    return balanced_samples

def get_batch_prediction(batch_data, few_shot_examples=None):
    """
    核心交互函数
    """
    # ================= 构造 Prompt (防止泄露的关键在这里) =================
    prompt = "任务：情感分类。请判断以下评论是【正向】还是【负向】。\n\n"
    
    # Part 1: 上下文示例 (来自训练集)
    # 这里必须把 Label 放进去，因为这是教模型怎么做
    if few_shot_examples:
        prompt += "=== 参考样例 (仅供学习) ===\n"
        for ex in few_shot_examples:
            prompt += f"评论: {ex['text']}\n情感: {ex['label']}\n---\n"
        prompt += "\n"

    # Part 2: 待测数据 (来自测试集)
    # 【绝对重点】：这里只放 text，不放 label！
    prompt += "=== 请对以下评论进行分类 ===\n"
    for idx, item in enumerate(batch_data):
        # 这里的 item['text'] 就是 "房间还可以..."
        # 我们没有把 item['label'] 放进去，这就是防止泄露
        prompt += f"{idx+1}. {item['text']}\n"
    
    prompt += f"\n请直接输出 {len(batch_data)} 行结果，每行一个标签（正向/负向）。不要输出其他内容。"
    # =================================================================

    try:
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=2000,
            temperature=0.0,
            system='你是一个情感分析助手。只输出标签。',
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )
        content = message.content[0].text.strip()
        # 简单清洗结果：去掉编号（如 "1. 正向" -> "正向"）
        lines = []
        for line in content.split('\n'):
            line = line.strip()
            if not line: continue
            # 去掉开头的编号，如 "1. ", "12. ", "123."
            line = re.sub(r'^\d+[\.\、\)\]\s]+\s*', '', line)
            lines.append(line)
        
        # 返回预测结果和 token 信息
        tokens_info = {
            'prompt_tokens': message.usage.input_tokens,
            'completion_tokens': message.usage.output_tokens,
            'total_tokens': message.usage.input_tokens + message.usage.output_tokens
        }
        return lines[:len(batch_data)], tokens_info
    except Exception as e:
        print(f"Error: {e}")
        return ["Error"] * len(batch_data), None

def log(message, file_handle):
    """同时输出到控制台和文件"""
    print(message)
    file_handle.write(message + "\n")

def main():
    # 打开日志文件
    output_file = os.path.join(SCRIPT_DIR, "results.txt")
    f = open(output_file, 'w', encoding='utf-8')
    
    # 输出测试配置信息
    log("="*50, f)
    log("📝 测试配置信息", f)
    log("="*50, f)
    log(f"模型: {MODEL_NAME}", f)
    log(f"API: {BASE_URL}", f)
    log(f"测试数据限制: {TEST_LIMIT} 条", f)
    log(f"Few-shot 示例数: {FEW_SHOT_NUM} 条", f)
    log(f"Batch 大小: {BATCH_SIZE} 条", f)
    log(f"训练集文件: {TRAIN_FILE}", f)
    log(f"测试集文件: {TEST_FILE}", f)
    log(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", f)
    log("="*50 + "\n", f)
    
    log("1. 正在加载数据...", f)
    # 加载训练集全量
    full_train = load_dataset(TRAIN_FILE, is_test_file=False)
    # 从训练集里均衡采样 Few-shot 例子
    few_shot_examples = balance_sample(full_train, FEW_SHOT_NUM)
    
    # 加载测试集 (采样后)
    test_data = load_dataset(TEST_FILE, is_test_file=True)
    
    log(f"   训练集样例数量: {len(few_shot_examples)}", f)
    positive_count = sum(1 for ex in few_shot_examples if ex['label'] == '正向')
    negative_count = sum(1 for ex in few_shot_examples if ex['label'] == '负向')
    log(f"   正向: {positive_count} | 负向: {negative_count}", f)
    for idx, ex in enumerate(few_shot_examples):
        log(f"   示例 {idx+1}: {ex}", f)
    log(f"   准备测试 {len(test_data)} 条数据...", f)

    # ================= 开始实验 (Few-shot) =================
    log("\n🚀 开始运行 Few-shot 实验...", f)
    all_preds = []
    all_truths = [item['label'] for item in test_data] # 真实标签存在这，不发给 API
    total_tokens = 0
    
    # 批处理循环
    for i in range(0, len(test_data), BATCH_SIZE):
        batch = test_data[i : i + BATCH_SIZE]
        log(f"Processing batch {i//BATCH_SIZE + 1}/{(len(test_data)-1)//BATCH_SIZE + 1} ({len(batch)} 条)...", f)
        
        # 调用 API
        preds, tokens_info = get_batch_prediction(batch, few_shot_examples)
        
        # 补齐长度（万一模型只回了部分，防止报错）
        while len(preds) < len(batch): preds.append("Error")
        
        # 累计 token 数
        if tokens_info:
            total_tokens += tokens_info['total_tokens']
            log(f"Done. (Token: {tokens_info['total_tokens']})", f)
        else:
            log("Done.", f)
            
        all_preds.extend(preds)
        time.sleep(1) # 休息一下

    # ================= 计算准确率 =================
    # 简单清洗数据（防止 label 不一致）
    clean_preds = []
    clean_truths = []
    for p, t in zip(all_preds, all_truths):
        # 只要包含了关键词就算对 (Claude有时候会回 "是正向")
        p_clean = "正向" if "正向" in p else ("负向" if "负向" in p else "Error")
        if p_clean != "Error":
            clean_preds.append(p_clean)
            clean_truths.append(t)
    
    acc = accuracy_score(clean_truths, clean_preds)
    
    # 输出结果
    log("\n" + "="*50, f)
    log("📋 测试集结果详情:", f)
    log("="*50, f)
    for idx, (text, true_label, pred_label) in enumerate(zip([item['text'] for item in test_data], all_truths, all_preds)):
        match = "✅" if ("正向" in pred_label if true_label == "正向" else "负向" in pred_label) else "❌"
        log(f"{idx+1}. {match} 文本: {text}", f)
        log(f"   真实: {true_label} | 预测: {pred_label}\n", f)
            
    log("="*50, f)
    log(f"📊 最终准确率: {acc:.2%}", f)
    log(f"💰 总 Token 消耗: {total_tokens}", f)
    log("="*50, f)
    
    f.close()
    print(f"\n✅ 结果已保存到: {output_file}")

if __name__ == "__main__":
    main()