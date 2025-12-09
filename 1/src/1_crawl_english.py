import requests
import os

# --- 配置 ---
# 使用 os.path.join 确保跨平台兼容性
OUTPUT_DIR = os.path.join('data', 'raw')
EN_FILENAME = os.path.join(OUTPUT_DIR, "english_data.txt")

EN_URLS = [
    "https://www.gutenberg.org/files/98/98-0.txt",
    "https://www.gutenberg.org/files/2701/2701-0.txt",
    "https://www.gutenberg.org/files/4300/4300-0.txt",
    "https://www.gutenberg.org/files/1342/1342-0.txt",
    "https://www.gutenberg.org/files/84/84-0.txt",
    "https://www.gutenberg.org/files/2600/2600-0.txt"
]

def crawl_english_data():
    # 自动创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if os.path.exists(EN_FILENAME):
        os.remove(EN_FILENAME)
    print("开始下载英文数据...")

    for i, url in enumerate(EN_URLS):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(EN_FILENAME, 'a', encoding='utf-8') as f:
                f.write(response.text + "\n\n" + "="*80 + "\n\n")
            print(f"  ({i+1}/{len(EN_URLS)}) 成功下载并追加来自 {url}")
        except requests.exceptions.RequestException as e:
            print(f"  下载 {url} 时出错: {e}")

    if os.path.exists(EN_FILENAME):
        file_size = os.path.getsize(EN_FILENAME) / (1024 * 1024)
        print(f"\n成功合并所有英文数据到 {EN_FILENAME}")
        print(f"文件总大小: {file_size:.2f} MB")
    else:
        print("\n未能创建英文数据文件。")

if __name__ == "__main__":
    crawl_english_data()