import os
import re
import jieba
import requests
import nltk
from tqdm import tqdm

# data/raw/ 目录下的文件
DATA_RAW_DIR = os.path.join('data', 'raw')
EN_RAW_FILE = os.path.join(DATA_RAW_DIR, "english_data.txt")
ZH_RAW_FILE = os.path.join(DATA_RAW_DIR, "chinese_data.txt")
ZH_TECH_RAW_FILE = os.path.join(DATA_RAW_DIR, "chinese_data_tech.txt")


# data/processed/ 目录下的文件
DATA_PROCESSED_DIR = os.path.join('data', 'processed')
EN_PROCESSED_FILE = os.path.join(DATA_PROCESSED_DIR, "english_data_processed.txt")
ZH_PROCESSED_FILE = os.path.join(DATA_PROCESSED_DIR, "chinese_data_processed.txt")
ZH_TECH_PROCESSED_FILE = os.path.join(DATA_PROCESSED_DIR, "chinese_data_tech_processed.txt")


def prepare_dependencies():
    """
    检查和下载所有外部数据依赖项。
    """
    print("--- 正在检查并准备所有必要的依赖数据... ---")

    # 检查并下载 NLTK 数据包
    required_nltk_packages = {
        'punkt': 'tokenizers/punkt',
    }
    all_packages_found = True
    for pkg_name, pkg_path in required_nltk_packages.items():
        try:
            nltk.data.find(pkg_path)
        except LookupError:
            all_packages_found = False
            print(f"NLTK 数据包 '{pkg_name}' 缺失，将在检查后统一自动下载...")
            
    if all_packages_found:
        print("所有必需的 NLTK 数据包均已存在。")
    else:
        print("\n正在下载缺失的 NLTK 数据包...")
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        for pkg_name in required_nltk_packages:
            if not nltk.data.find(required_nltk_packages[pkg_name], quiet=True):
                 print(f"  - 下载 {pkg_name}...")
                 nltk.download(pkg_name, quiet=True)
        print("NLTK 数据下载完成。")
    
    print("--- 所有依赖数据准备就绪。---\n")


def preprocess_english():
    """处理英文文本。"""
    if not os.path.exists(EN_RAW_FILE):
        print(f"错误: 找不到原始英文文件 {EN_RAW_FILE}，跳过处理。")
        return

    print(f"开始处理英文文件: {EN_RAW_FILE}")
    with open(EN_RAW_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    print("  - 步骤 1: 正在进行分词...")
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)

    print("  - 步骤 2: 转换为小写并清理 (只保留字母)...")
    words = [word.lower() for word in tokens if word.isalpha()]

    print(f"  - 步骤 3: 保存处理后的文本到 {EN_PROCESSED_FILE}...")
    with open(EN_PROCESSED_FILE, 'w', encoding='utf-8') as f:
        f.write(' '.join(words))
    
    print("英文文件处理完成！\n")


def preprocess_chinese():
    """处理中文文本。"""
    if not os.path.exists(ZH_RAW_FILE):
        print(f"错误: 找不到原始中文文件 {ZH_RAW_FILE}，跳过处理。")
        return

    print(f"开始处理中文文件: {ZH_RAW_FILE}")
    
    with open(ZH_RAW_FILE, 'r', encoding='utf-8') as f_in, \
         open(ZH_PROCESSED_FILE, 'w', encoding='utf-8') as f_out:
        
        print("  - 正在逐行去除标点、分词...")
        total_size = os.path.getsize(ZH_RAW_FILE)
        # 只保留中文字符的正则表达式
        chinese_only = re.compile(r'[^\u4e00-\u9fa5]')

        with tqdm(total=total_size, unit='B', unit_scale=True, desc="中文处理进度") as pbar:
            for line in f_in:
                current_bytes = len(line.encode('utf-8'))
                line = line.strip()
                if not line:
                    pbar.update(current_bytes)
                    continue
                
                # 去除标点和非中文字符
                cleaned_line = chinese_only.sub('', line)
                
                words = jieba.cut(cleaned_line)
                # 直接使用分词结果，不过滤停用词
                filtered_words = [word for word in words if word.strip()]

                if filtered_words:
                    f_out.write(' '.join(filtered_words) + ' ')
                pbar.update(current_bytes)
    
    print("中文文件处理完成！")


def preprocess_chinese_tech():
    """处理中文科技新闻文本。"""
    if not os.path.exists(ZH_TECH_RAW_FILE):
        print(f"错误: 找不到原始中文科技新闻文件 {ZH_TECH_RAW_FILE}，跳过处理。")
        return

    print(f"开始处理中文科技新闻文件: {ZH_TECH_RAW_FILE}")
    
    with open(ZH_TECH_RAW_FILE, 'r', encoding='utf-8') as f_in, \
         open(ZH_TECH_PROCESSED_FILE, 'w', encoding='utf-8') as f_out:
        
        print("  - 正在逐行去除标点、分词...")
        total_size = os.path.getsize(ZH_TECH_RAW_FILE)
        # 只保留中文字符的正则表达式
        chinese_only = re.compile(r'[^\u4e00-\u9fa5]')

        with tqdm(total=total_size, unit='B', unit_scale=True, desc="中文科技新闻处理进度") as pbar:
            for line in f_in:
                current_bytes = len(line.encode('utf-8'))
                line = line.strip()
                if not line:
                    pbar.update(current_bytes)
                    continue
                
                # 去除标点和非中文字符
                cleaned_line = chinese_only.sub('', line)
                
                words = jieba.cut(cleaned_line)
                # 直接使用分词结果，不过滤停用词
                filtered_words = [word for word in words if word.strip()]

                if filtered_words:
                    f_out.write(' '.join(filtered_words) + ' ')
                pbar.update(current_bytes)
    
    print("中文科技新闻文件处理完成！")

def extract_dates_and_numbers():

    # 从原始语料中抽取日期和数字，并保存结果。
    print("\n--- 开始执行基本要求：抽取日期和数字... ---")
    
    # 定义正则表达式
    # 数字: 匹配整数、小数、百分比
    number_pattern = re.compile(r'\d+\.?\d*%?')
    # 日期: 匹配 YYYY-MM-DD, YYYY/MM/DD, YYYY年MM月DD日, MM月DD日
    date_pattern = re.compile(r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?|\d{1,2}月\d{1,2}日')

    output_dir = "output"
    output_file = os.path.join(output_dir, "extractions.txt")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write("="*20 + " 日期和数字抽取结果 " + "="*20 + "\n\n")
        
        for raw_file, name in [(EN_RAW_FILE, "英文语料"), (ZH_RAW_FILE, "中文语料")]:
            if not os.path.exists(raw_file):
                print(f"未找到原始文件 {raw_file}，跳过抽取。")
                continue
            
            print(f"正在从 {name} ({raw_file}) 中抽取...")
            with open(raw_file, 'r', encoding='utf-8') as f_in:
                text = f_in.read()
            
            numbers = number_pattern.findall(text)
            dates = date_pattern.findall(text)
            
            f_out.write(f"--- {name} ---\n")
            f_out.write(f"找到数字 {len(numbers)} 个 (示例前50个): {numbers[:50]}\n")
            f_out.write(f"找到日期 {len(dates)} 个 (示例前50个): {dates[:50]}\n")
            f_out.write("-"*(40 + len(name)) + "\n\n")

    print(f"抽取结果已保存到: {output_file}")


if __name__ == "__main__":
    prepare_dependencies()
    
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    preprocess_english()
    preprocess_chinese()
    preprocess_chinese_tech()
    extract_dates_and_numbers()