import os
from gensim.models import Word2Vec

# --- 文件和参数 ---
EN_MODEL_PATH = "output/models/word2vec_english.model"
ZH_MODEL_PATH = "output/models/word2vec_chinese.model"
TOP_K = 10

# --- 查询词列表 ---
# 给定 20 个英文单词
ENGLISH_QUERY_WORDS = [
    "time", "love", "war", "city", "night",
    "friend", "power", "world", "heart", "light",
    "king", "house", "hand", "water", "woman",
    "story", "death", "truth", "life", "fear"
]

# 给定 20 个中文词
CHINESE_QUERY_WORDS = [
    "中国", "政府", "经济", "市场", "公司",
    "技术", "人工智能", "芯片", "手机", "互联网",
    "投资", "产业", "应用", "平台", "数据",
    "安全", "研究", "高校", "美国", "欧洲"
]

def find_and_print_similar(model_path, query_words, lang_name):
    if not os.path.exists(model_path):
        print(f"错误: 找不到 {lang_name} 模型文件 {model_path}。请先运行训练脚本。")
        return

    print(f"\n{'='*20} {lang_name} 相似词查询结果 {'='*20}")
    
    # 加载模型
    model = Word2Vec.load(model_path)
    
    for word in query_words:
        # 检查词是否在模型的词汇表中
        if word in model.wv:
            # 查找最相似的词
            similar_words = model.wv.most_similar(word, topn=TOP_K)
            print(f"\n与 [{word}] 最相似的 {TOP_K} 个词是:")
            for sim_word, score in similar_words:
                print(f"  - {sim_word:<15} (相似度: {score:.4f})")
        else:
            print(f"\n词 [{word}] 不在词汇表中 (OOV)，已跳过。")

if __name__ == "__main__":
    find_and_print_similar(EN_MODEL_PATH, ENGLISH_QUERY_WORDS, "英文")
    find_and_print_similar(ZH_MODEL_PATH, CHINESE_QUERY_WORDS, "中文")