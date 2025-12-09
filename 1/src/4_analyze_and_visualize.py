import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

# --- 配置 ---
DATA_PROCESSED_DIR = os.path.join('data', 'processed')
OUTPUT_CHARTS_DIR = os.path.join('output', 'charts')
EN_PROCESSED_FILE = os.path.join(DATA_PROCESSED_DIR, "english_data_processed.txt")
ZH_PROCESSED_FILE = os.path.join(DATA_PROCESSED_DIR, "chinese_data_processed.txt")
FONT_PATH_FOR_WORDCLOUD = 'C:/Windows/Fonts/simhei.ttf' # 用于词云

# --- 中文显示配置 ---
def configure_matplotlib_for_chinese():
    print("正在配置 Matplotlib 以支持中文显示...")
    try:
        from matplotlib.font_manager import fontManager
        font_names = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
        for font_name in font_names:
            if font_name in [f.name for f in fontManager.ttflist]:
                plt.rcParams['font.sans-serif'] = [font_name]
                print(f"已将 Matplotlib 字体设置为 '{font_name}'。")
                plt.rcParams['axes.unicode_minus'] = False
                return
        print("警告：未找到常用中文字体。图表中的中文可能无法显示。")
        plt.rcParams['font.sans-serif'] = ['SimHei'] # Fallback
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"配置中文字体时出错: {e}")

# --- 分析主函数 ---
def analyze_corpus(file_path, language_name):
    # 对给定的语料文件进行全面的分析和可视化。
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 '{file_path}'，无法进行分析。")
        return

    print(f"\n{'='*20} 开始分析: {language_name} 语料 {'='*20}")
    with open(file_path, 'r', encoding='utf-8') as f:
        words = f.read().split()

    # 1. 基本统计
    total_words = len(words)
    unique_words = len(set(words))
    print(f"  - 总词数: {total_words:,}")
    print(f"  - 独立词汇量: {unique_words:,}")

    # 2. 词频统计
    print("  - 正在统计词频...")
    word_counts = Counter(words)
    print("  - Top 20 高频词:")
    for word, count in word_counts.most_common(20):
        print(f"    - {word:<15} {count:,}")

    # 3. 齐夫定律验证
    print("  - 正在生成齐夫定律（Zipf's Law）验证图...")
    frequencies = [count for _, count in word_counts.most_common()]
    ranks = range(1, len(frequencies) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, marker=".", linestyle='none', label=f'{language_name} 词频分布')
    
    # 拟合直线
    coeffs = np.polyfit(np.log(ranks), np.log(frequencies), 1)
    fit_line = np.exp(coeffs[1] + coeffs[0] * np.log(ranks))
    plt.plot(ranks, fit_line, 'r--', label=f'拟合线 (斜率: {coeffs[0]:.2f})')
    
    plt.title(f'{language_name} 词频的齐夫定律验证 (Log-Log Plot)')
    plt.xlabel('排名 (Rank) - 对数刻度')
    plt.ylabel('频率 (Frequency) - 对数刻度')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    zipf_chart_path = os.path.join(OUTPUT_CHARTS_DIR, f"zipf_law_{language_name.lower()}.png")
    plt.savefig(zipf_chart_path)
    print(f"  - 齐夫定律图已保存到: {zipf_chart_path}")
    plt.show()

    # 4. 生成词云
    print(f"  - 正在生成 {language_name} 词云...")
    font_path = FONT_PATH_FOR_WORDCLOUD if language_name == "Chinese" else None
    if language_name == "Chinese" and not os.path.exists(font_path):
        print(f"  - 警告：找不到中文字体 '{font_path}'，词云可能乱码。")
        return # 如果没字体，中文词云就没意义了
        
    wordcloud = WordCloud(
        font_path=font_path,
        width=1000, height=500,
        background_color='white',
        max_words=150,
        collocations=False # 避免词语组合
    ).generate(' '.join(words))

    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{language_name} 高频词词云')
    wordcloud_path = os.path.join(OUTPUT_CHARTS_DIR, f"word_cloud_{language_name.lower()}.png")
    plt.savefig(wordcloud_path)
    print(f"  - 词云图已保存到: {wordcloud_path}")
    plt.show()

if __name__ == "__main__":

    os.makedirs(OUTPUT_CHARTS_DIR, exist_ok=True)
    configure_matplotlib_for_chinese()
    analyze_corpus(EN_PROCESSED_FILE, "English")
    analyze_corpus(ZH_PROCESSED_FILE, "Chinese")