import os
import argparse
from gensim.models import Word2Vec 
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm
# Word2Vec 来自外部库 Gensim, 不需要自己实现
# --- 进度条回调类 ---
class TqdmProgressCallback(CallbackAny2Vec):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.pbar = None

    def on_train_begin(self, model):
        print(f"Word2Vec 训练开始，共 {self.total_epochs} 个 epochs...")
        self.pbar = tqdm(total=self.total_epochs, desc="总训练进度")

    def on_epoch_end(self, model):
        self.pbar.update(1)
        
    def on_train_end(self, model):
        self.pbar.close()
        print("Word2Vec 训练完成。")

# --- 训练主函数 ---
def train_model(args):
    if not os.path.exists(args.input):
        print(f"错误: 找不到输入文件 {args.input}，无法开始训练。")
        return

    model_type_str = 'Skip-gram' if args.sg == 1 else 'CBOW'
    print(f"\n--- 开始训练 Word2Vec 模型 ---")
    print(f"  语料: {args.input}")
    print(f"  输出: {args.output}")
    print(f"  参数: {model_type_str}, Dim={args.size}, Win={args.window}, MinCount={args.min_count}, Epochs={args.epochs}, Negative={args.negative}")

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    model = Word2Vec(
        corpus_file=args.input,
        vector_size=args.size,
        window=args.window,
        min_count=args.min_count,
        sg=args.sg,
        negative=args.negative,
        workers=args.workers,
        epochs=args.epochs,
        callbacks=[TqdmProgressCallback(args.epochs)]
    )

    model.save(args.output)
    print(f"模型已成功保存到: {args.output}")

# --- 程序入口 ---
if __name__ == "__main__":
    # --- 路径常量 ---
    PROCESSED_DATA_DIR = os.path.join('data', 'processed')
    MODELS_OUTPUT_DIR = os.path.join('output', 'models')
    
    EN_PROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, "english_data_processed.txt")
    ZH_PROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, "chinese_data_processed.txt")
    
    EN_MODEL_PATH = os.path.join(MODELS_OUTPUT_DIR, "word2vec_english.model")
    ZH_MODEL_PATH = os.path.join(MODELS_OUTPUT_DIR, "word2vec_chinese.model")

    parser = argparse.ArgumentParser(description="通用 Word2Vec 模型训练脚本")
    
    parser.add_argument('--input', type=str, help="指定输入的预处理语料文件路径。")
    parser.add_argument('--output', type=str, help="指定输出模型的完整路径。")
    
    # 从原始脚本中提取的默认参数
    parser.add_argument('--size', type=int, default=150, help="词向量维度")
    parser.add_argument('--window', type=int, default=5, help="上下文窗口大小")
    parser.add_argument('--min_count', type=int, default=5, help="忽略频率低于此值的词")
    parser.add_argument('--epochs', type=int, default=10, help="训练轮次")
    parser.add_argument('--workers', type=int, default=4, help="使用的CPU核心数")
    parser.add_argument('--sg', type=int, default=1, choices=[0, 1], help="模型类型 (0 for CBOW, 1 for Skip-gram)")
    parser.add_argument('--negative', type=int, default=5, help="负采样数量")

    args = parser.parse_args()

    if not args.input and not args.output:
        
        # 英文模型 
        default_en_args = parser.parse_args([]) # 创建一个包含所有默认值的命名空间
        default_en_args.input = EN_PROCESSED_FILE
        default_en_args.output = EN_MODEL_PATH
        train_model(default_en_args)
        
        # 中文模型 
        default_zh_args = parser.parse_args([])
        default_zh_args.input = ZH_PROCESSED_FILE
        default_zh_args.output = ZH_MODEL_PATH
        train_model(default_zh_args)
    
    elif args.input and args.output:
        train_model(args)
    else:
        print("错误: --input 和 --output 参数必须同时提供。") # 可以手动设置 input 和 output