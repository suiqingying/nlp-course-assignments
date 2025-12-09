
import os
from gensim.models import Word2Vec

# --- 模型路径定义 ---
MODELS_DIR = os.path.join('output', 'models')

# 基础模型 (Skip-gram, neg=5)
MODEL_ZH_BASE = os.path.join(MODELS_DIR, "word2vec_chinese.model")

# 拓展实验模型
MODEL_ZH_TECH = os.path.join(MODELS_DIR, "word2vec_chinese_tech.model") # 领域对比
MODEL_ZH_CBOW = os.path.join(MODELS_DIR, "word2vec_chinese_cbow.model")   # 算法对比
MODEL_ZH_NEG15 = os.path.join(MODELS_DIR, "word2vec_chinese_neg15.model") # 超参数对比

TOP_K = 10

# --- 辅助函数 ---
def find_and_print_similar(model_path, query_words, model_name):
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}。请先确保已训练此模型。\n")
        return

    model = Word2Vec.load(model_path)
    print(f"\n--- 模型: {model_name} ---")
    
    for word in query_words:
        if word in model.wv:
            similar_words = model.wv.most_similar(word, topn=TOP_K)
            print(f"查询词 [{word}] 的 Top-{TOP_K} 相似词:")
            for sim_word, score in similar_words:
                print(f"  - {sim_word:<15} (相似度: {score:.4f})")
        else:
            print(f"查询词 [{word}] 不在词汇表中 (OOV)。")
    print("-"*40)

# --- 实验一：语料领域对比 ---
def experiment_1_domain_comparison():
    print(f"\n{'='*25}")
    print("  拓展实验一: 语料领域对词义表示的影响")
    print(f"{'='*25}")
    
    query_words = ["苹果", "智能", "芯片"]
    
    print("\n[通用新闻语料模型] vs [科技新闻语料模型]")
    find_and_print_similar(MODEL_ZH_BASE, query_words, "通用新闻 (Skip-gram, neg=5)")
    find_and_print_similar(MODEL_ZH_TECH, query_words, "科技新闻 (Skip-gram, neg=5)")

# --- 实验二：训练算法对比 ---
def experiment_2_algorithm_comparison():
    print(f"\n{'='*25}")
    print("  拓展实验二: 训练算法的对比 (Skip-gram vs. CBOW)")
    print(f"{'='*25}")
    
    query_words = ["公司", "技术"]
    
    print("\n[Skip-gram 模型] vs [CBOW 模型]")
    find_and_print_similar(MODEL_ZH_BASE, query_words, "Skip-gram (sg=1, neg=5)")
    find_and_print_similar(MODEL_ZH_CBOW, query_words, "CBOW (sg=0, neg=5)")

# --- 实验三：超参数对比 ---
def experiment_3_hyperparameter_comparison():
    print(f"\n{'='*25}")
    print("  拓展实验三: 训练超参数的对比 (负采样数量)")
    print(f"{'='*25}")
    
    query_words = ["市场"]
    
    print("\n[负采样数=5] vs [负采样数=15]")
    find_and_print_similar(MODEL_ZH_BASE, query_words, "负采样数=5 (Skip-gram)")
    find_and_print_similar(MODEL_ZH_NEG15, query_words, "负采样数=15 (Skip-gram)")

# --- 主程序入口 ---
if __name__ == "__main__":
    print("--- 开始执行所有拓展对比实验 ---")
    experiment_1_domain_comparison()
    experiment_2_algorithm_comparison()
    experiment_3_hyperparameter_comparison()
    print("\n--- 所有拓展实验执行完毕 ---")