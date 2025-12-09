import argparse
import json
import math
import os
import random
import warnings
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 过滤 PyTorch 的 nested tensor 和 mask 类型警告
warnings.filterwarnings("ignore", message=".*nested tensor.*")
warnings.filterwarnings("ignore", message=".*mismatched key_padding_mask.*")

# BLEU 评估
try:
    from sacrebleu.metrics import BLEU
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False
    print("警告: 未安装 sacrebleu，BLEU 评估将不可用。安装: pip install sacrebleu")


# 位置编码实现（正弦/余弦，固定不训练）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        pe = self.pe  # type: torch.Tensor
        x = x + pe[:, : x.size(1)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_id: int = 3,
    ):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 直接用(batch, seq, dim)
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor):
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ):
        tgt_emb = self.pos_decoder(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        output = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.generator(output)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        memory = self.encode(src, src_key_padding_mask)
        out = self.decode(tgt, memory, tgt_mask, tgt_key_padding_mask, src_key_padding_mask)
        return out


# --- 数据处理 --- #
def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_stripped_lookup(vocab: Dict[str, int]) -> Dict[str, int]:
    """
    将词表中的token去掉@@后作为查找键；若发生碰撞，保留最先出现的id。
    这样在“@@视作分隔标记”的假设下仍可复用原始词表的id空间。
    """
    lookup: Dict[str, int] = {}
    for tok, idx in vocab.items():
        stripped = strip_bpe_token(tok)
        if stripped not in lookup:
            lookup[stripped] = idx
    return lookup


def strip_bpe_token(tok: str) -> str:
    # 去掉BPE续接标记“@@”，供显示使用
    return tok.replace("@@", "")


def detok_bpe_text(text: str, lang: str = "zh") -> str:
    """
    将带@@的BPE文本还原为可读形式。
    lang: "zh" 时额外去掉空格；其他语言保留空格。
    """
    text = text.replace("@@ ", "")
    if lang.lower().startswith("zh"):
        text = text.replace(" ", "")
    return text.strip()


def get_logger(log_file: str):
    def _log(msg: str):
        print(msg)
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
    return _log


def line_to_ids(
    line: str,
    vocab: Dict[str, int],
    max_len: int,
    go_id: int,
    eos_id: int,
    unk_id: int,
) -> List[int]:
    # 按“@@为分隔”理解：直接用 "@@" 切分，然后去空白
    tokens = [tok.strip() for tok in line.split("@@") if tok.strip() != ""]
    ids = [vocab.get(tok, unk_id) for tok in tokens]
    # 预留GO和EOS位置
    ids = ids[: max_len - 2]
    return [go_id] + ids + [eos_id]


def ids_to_line(
    ids: List[int],
    id2tok: Dict[int, str],
    go_id: int,
    eos_id: int,
    pad_id: int,
) -> str:
    out_tokens = []
    for tid in ids:
        if tid in (go_id, pad_id):
            continue
        if tid == eos_id:
            break
        tok = strip_bpe_token(id2tok.get(tid, "<UNK>"))
        out_tokens.append(tok)
    return " ".join(out_tokens)


class MTDataset(Dataset):
    def __init__(
        self,
        src_path: str,
        tgt_path: str,
        src_vocab: Dict[str, int],
        tgt_vocab: Dict[str, int],
        max_len: int,
        go_id: int,
        eos_id: int,
        unk_id: int,
    ):
        super().__init__()
        with open(src_path, "r", encoding="utf-8") as f:
            self.src_lines = f.readlines()
        with open(tgt_path, "r", encoding="utf-8") as f:
            self.tgt_lines = f.readlines()
        assert len(self.src_lines) == len(self.tgt_lines), "源/目标句子数量不一致"
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.go_id = go_id
        self.eos_id = eos_id
        self.unk_id = unk_id

    def __len__(self) -> int:
        return len(self.src_lines)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        src_line = self.src_lines[idx]
        tgt_line = self.tgt_lines[idx]
        src_ids = line_to_ids(src_line, self.src_vocab, self.max_len, self.go_id, self.eos_id, self.unk_id)
        tgt_ids = line_to_ids(tgt_line, self.tgt_vocab, self.max_len, self.go_id, self.eos_id, self.unk_id)
        return src_ids, tgt_ids


def make_causal_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    attention_mask: (batch, seq_len) True表示padding位置
    返回: (seq_len, seq_len) 下三角，未mask位置为0，mask为-Inf
    """
    seq_len = attention_mask.shape[1]
    # 先生成下三角 (seq, seq)
    mask = torch.tril(torch.ones((seq_len, seq_len), device=attention_mask.device))
    # 将1/0转为0/-inf，nn.Transformer期望添加到注意力logits
    mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
    return mask


def collate_fn(batch: List[Tuple[List[int], List[int]]], pad_id: int = 3):
    src_input = [item[0] for item in batch]
    tgt_input = [item[1] for item in batch]
    batch_size = len(src_input)
    src_seq_len = max(len(x) for x in src_input)
    tgt_seq_len = max(len(x) for x in tgt_input)

    src_mask = torch.ones((batch_size, src_seq_len), dtype=torch.bool)
    tgt_mask = torch.ones((batch_size, tgt_seq_len), dtype=torch.bool)
    src = torch.full((batch_size, src_seq_len), pad_id, dtype=torch.long)
    tgt = torch.full((batch_size, tgt_seq_len), pad_id, dtype=torch.long)

    for i in range(batch_size):
        src[i, : len(src_input[i])] = torch.tensor(src_input[i], dtype=torch.long)
        tgt[i, : len(tgt_input[i])] = torch.tensor(tgt_input[i], dtype=torch.long)

    src_mask = src == pad_id
    tgt_mask = tgt == pad_id
    causal_mask = make_causal_mask(tgt_mask)
    return src, src_mask, tgt, tgt_mask, causal_mask


# --- 训练与验证 --- #
def train_one_epoch(model, dataloader, optimizer, criterion, device, check_interval: int, tgt_pad_id: int, log_fn=print):
    model.train()
    total_loss = 0.0
    log_fn(f"开始训练: {len(dataloader)} 个batch")
    for step, batch in enumerate(dataloader):
        src, src_key_padding, tgt, tgt_key_padding, causal_mask = batch
        src = src.to(device)
        src_key_padding = src_key_padding.to(device)
        tgt = tgt.to(device)
        tgt_key_padding = tgt_key_padding.to(device)
        causal_mask = causal_mask.to(device)

        # 输入序列/标签序列（右移一位）
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]
        tgt_input_mask = causal_mask[: tgt_input.size(1), : tgt_input.size(1)]
        tgt_input_key_padding = tgt_key_padding[:, :-1]
        tgt_label_key_padding = tgt_key_padding[:, 1:]

        logits = model(
            src=src,
            tgt=tgt_input,
            src_key_padding_mask=src_key_padding,
            tgt_key_padding_mask=tgt_input_key_padding,
            tgt_mask=tgt_input_mask,
        )
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if check_interval > 0 and (step + 1) % check_interval == 0:
            avg = total_loss / (step + 1)
            log_fn(f"[Train] Step {step+1}/{len(dataloader)}, loss {avg:.4f}")
    return total_loss / max(len(dataloader), 1)


def evaluate(model, dataloader, criterion, device, log_fn=print):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    log_fn(f"开始验证: {len(dataloader)} 个batch")
    with torch.no_grad():
        for batch in dataloader:
            src, src_key_padding, tgt, tgt_key_padding, causal_mask = batch
            src = src.to(device)
            src_key_padding = src_key_padding.to(device)
            tgt = tgt.to(device)
            tgt_key_padding = tgt_key_padding.to(device)
            causal_mask = causal_mask.to(device)

            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]
            tgt_input_mask = causal_mask[: tgt_input.size(1), : tgt_input.size(1)]
            tgt_input_key_padding = tgt_key_padding[:, :-1]

            logits = model(
                src=src,
                tgt=tgt_input,
                src_key_padding_mask=src_key_padding,
                tgt_key_padding_mask=tgt_input_key_padding,
                tgt_mask=tgt_input_mask,
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))
            # 只统计非PAD token
            valid_tokens = (tgt_labels != criterion.ignore_index).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    return total_loss / max(total_tokens, 1)


def evaluate_bleu(
    model, 
    dataloader, 
    args, 
    tgt_id2tok: Dict[int, str],
    go_id: int, 
    eos_id: int, 
    pad_id: int,
    log_fn=print
) -> float:
    """使用贪婪解码计算 BLEU 分数"""
    if not HAS_SACREBLEU:
        log_fn("sacrebleu 未安装，跳过 BLEU 评估")
        return 0.0
    
    model.eval()
    all_preds = []
    all_refs = []
    
    log_fn(f"开始 BLEU 评估: {len(dataloader)} 个batch")
    
    with torch.no_grad():
        for batch in dataloader:
            src, src_key_padding, tgt, tgt_key_padding, _ = batch
            
            # 贪婪解码
            ys = greedy_generate(model, src, src_key_padding, args, go_id, eos_id, pad_id)
            
            # 转换为文本
            for i in range(ys.size(0)):
                pred_line = ids_to_line(ys[i].tolist(), tgt_id2tok, go_id, eos_id, pad_id)
                ref_line = ids_to_line(tgt[i].tolist(), tgt_id2tok, go_id, eos_id, pad_id)
                
                # 去 BPE 还原
                pred_text = detok_bpe_text(pred_line, lang="zh")
                ref_text = detok_bpe_text(ref_line, lang="zh")
                
                all_preds.append(pred_text)
                all_refs.append(ref_text)
    
    # 计算 BLEU
    bleu = BLEU(tokenize="zh")  # 中文分词
    result = bleu.corpus_score(all_preds, [all_refs])
    
    log_fn(f"BLEU 评估完成: {result.score:.2f}")
    return result.score


def greedy_generate(model, src, src_mask, args, go_id: int, eos_id: int, pad_id: int):
    model.eval()
    src = src.to(args.device)
    src_mask = src_mask.to(args.device)
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        batch_size = src.size(0)
        ys = torch.full((batch_size, 1), go_id, dtype=torch.long, device=args.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=args.device)

        for _ in range(args.max_len - 1):
            tgt_mask = make_causal_mask(torch.zeros((batch_size, ys.size(1)), device=args.device, dtype=torch.bool))
            out = model.decode(
                ys,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=torch.zeros_like(ys, dtype=torch.bool),
                memory_key_padding_mask=src_mask,
            )
            next_token = out[:, -1, :].argmax(dim=-1)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_id)
            if finished.all():
                break
        # 将PAD填充到未结束的序列末尾
        ys = torch.where(finished.unsqueeze(1), ys, ys)
        ys = torch.where(ys == 0, torch.full_like(ys, pad_id), ys)
    return ys


# --- 主流程 --- #
def parse_args():
    parser = argparse.ArgumentParser(description="Transformer NMT (按照课件实现)")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录，包含*.en/*.zh及词表json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--check_interval", type=int, default=100, help="打印loss的间隔step数")
    parser.add_argument("--device", type=str, default=None, help="留空自动选GPU优先，否则指定cuda/cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="model_best.pth")
    parser.add_argument("--log_file", type=str, default="train.log", help="日志文件路径，留空则只在屏幕输出")
    parser.add_argument("--eval_only", action="store_true", help="仅评估，不训练")
    parser.add_argument("--test_only", action="store_true", help="仅在测试集评估，不训练")
    parser.add_argument(
        "--test_after_train", action="store_true", help="训练完成后如果存在测试集则自动在测试集评估一次"
    )
    parser.add_argument("--generate", action="store_true", help="训练后示例贪婪解码")
    parser.add_argument("--interactive", action="store_true", help="进入交互式贪婪解码（推理）模式")
    parser.add_argument("--eval_bleu", action="store_true", help="评估时计算 BLEU 分数")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    log = get_logger(args.log_file)
    if args.log_file:
        with open(args.log_file, "w", encoding="utf-8") as f:
            f.write("start logging\n")

    # 设备选择：优先GPU
    if args.device is None or args.device.lower() == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log("指定cuda但未检测到GPU，自动切换到cpu")
        args.device = "cpu"
    if args.device.startswith("cuda"):
        log(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("使用CPU")

    src_vocab_raw = load_vocab(os.path.join(args.data_dir, "train.en.json"))
    tgt_vocab_raw = load_vocab(os.path.join(args.data_dir, "train.zh.json"))
    # 构造“去@@键”的查找表以支持将@@视为分隔标记
    src_vocab = build_stripped_lookup(src_vocab_raw)
    tgt_vocab = build_stripped_lookup(tgt_vocab_raw)
    tgt_id2tok = {v: strip_bpe_token(k) for k, v in tgt_vocab_raw.items()}
    eos_id = src_vocab.get("<EOS>", 0)
    go_id = src_vocab.get("<GO>", 1)
    unk_id = src_vocab.get("<UNK>", 2)
    pad_id = src_vocab.get("<PAD>", 3)

    train_dataset = MTDataset(
        src_path=os.path.join(args.data_dir, "train.en"),
        tgt_path=os.path.join(args.data_dir, "train.zh"),
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_len=args.max_len,
        go_id=go_id,
        eos_id=eos_id,
        unk_id=unk_id,
    )
    val_dataset = MTDataset(
        src_path=os.path.join(args.data_dir, "val.en"),
        tgt_path=os.path.join(args.data_dir, "val.zh"),
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_len=args.max_len,
        go_id=go_id,
        eos_id=eos_id,
        unk_id=unk_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id),
    )
    test_en = os.path.join(args.data_dir, "test.en")
    test_zh = os.path.join(args.data_dir, "test.zh")
    has_test = os.path.exists(test_en) and os.path.exists(test_zh)
    test_loader = None
    if has_test:
        test_dataset = MTDataset(
            src_path=test_en,
            tgt_path=test_zh,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            max_len=args.max_len,
            go_id=go_id,
            eos_id=eos_id,
            unk_id=unk_id,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id),
        )

    model = Seq2SeqTransformer(
        # 词表大小需覆盖最大id，使用原始词表的id空间以避免越界
        src_vocab_size=max(src_vocab_raw.values()) + 1,
        tgt_vocab_size=max(tgt_vocab_raw.values()) + 1,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pad_id=pad_id,
    ).to(args.device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")

    log(
        f"配置: device={args.device}, d_model={args.d_model}, nhead={args.nhead}, "
        f"enc_layers={args.num_encoder_layers}, dec_layers={args.num_decoder_layers}, "
        f"ffn={args.dim_feedforward}, dropout={args.dropout}, batch_size={args.batch_size}, lr={args.lr}, "
        f"epochs={args.epochs}, max_len={args.max_len}"
    )
    log(
        f"数据量: train={len(train_dataset)} 条, val={len(val_dataset)} 条 | "
        f"词表: src={len(src_vocab_raw)}(原始), tgt={len(tgt_vocab_raw)}(原始)"
    )
    if has_test:
        log(f"测试集: test={len(test_dataset)} 条 可用")
    else:
        log("未检测到测试集 test.en/test.zh，测试评估将跳过")

    if args.eval_only and os.path.exists(args.save_path):
        model.load_state_dict(torch.load(args.save_path, map_location=args.device))
        val_loss = evaluate(model, val_loader, criterion, args.device, log_fn=log)
        log(f"Eval-only | val loss: {val_loss:.4f}")
        
        # 计算 BLEU
        if args.eval_bleu:
            bleu_score = evaluate_bleu(
                model, val_loader, args, tgt_id2tok, go_id, eos_id, pad_id, log_fn=log
            )
            log(f"Eval-only | val BLEU: {bleu_score:.2f}")
        
        # eval_only 模式：如果没有 interactive 或 generate，直接返回
        # 如果有 generate，继续执行到 generate 块
        if not args.interactive and not args.generate:
            return
        # 有 generate 时不 return，继续往下执行
    if args.test_only:
        if not os.path.exists(args.save_path):
            log(f"未找到模型权重 {args.save_path}，无法 test_only")
            return
        if not has_test or test_loader is None:
            log("未找到测试集，无法 test_only")
            return
        model.load_state_dict(torch.load(args.save_path, map_location=args.device))
        test_loss = evaluate(model, test_loader, criterion, args.device, log_fn=log)
        log(f"Test-only | test loss: {test_loss:.4f}")
        
        # 计算 BLEU
        if args.eval_bleu:
            bleu_score = evaluate_bleu(
                model, test_loader, args, tgt_id2tok, go_id, eos_id, pad_id, log_fn=log
            )
            log(f"Test-only | test BLEU: {bleu_score:.2f}")
        return

    # 仅交互模式：跳过训练，直接加载模型进入交互
    if args.interactive:
        if os.path.exists(args.save_path):
            model.load_state_dict(torch.load(args.save_path, map_location=args.device))
            log(f"已加载模型 {args.save_path}")
        else:
            log(f"未找到已训练模型 {args.save_path}，将使用当前随机权重。")
        model.eval()
        print("进入交互式推理，输入与训练一致的分词格式，空行或Ctrl+C退出。")
        while True:
            try:
                line = input("源句子> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                break
            if line == "":
                print("已退出。")
                break
            src_ids = line_to_ids(line, src_vocab, args.max_len, go_id, eos_id, unk_id)
            src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0)
            src_pad_mask = src_tensor == pad_id
            ys = greedy_generate(model, src_tensor, src_pad_mask, args, go_id=go_id, eos_id=eos_id, pad_id=pad_id)
            tgt_line = ids_to_line(ys.squeeze(0).tolist(), tgt_id2tok, go_id, eos_id, pad_id)
            print(f"译文(BPE): {tgt_line}")
            print(f"译文(去BPE): {detok_bpe_text(tgt_line, lang='zh')}")
        return

    # 仅生成示例模式（配合 eval_only 使用）
    if args.generate and args.eval_only:
        # 模型已在 eval_only 块中加载
        model.eval()
        with open(os.path.join(args.data_dir, "val.en"), "r", encoding="utf-8") as f:
            src_lines_raw = f.readlines()
        with open(os.path.join(args.data_dir, "val.zh"), "r", encoding="utf-8") as f:
            tgt_lines_raw = f.readlines()
        
        log("\n" + "=" * 60)
        log("翻译示例 (双语对照)")
        log("=" * 60)
        
        num_samples = min(5, len(val_dataset))
        for i in range(num_samples):
            src_ids, tgt_ids = val_dataset[i]
            src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0)
            src_pad_mask = src_tensor == pad_id
            
            ys = greedy_generate(model, src_tensor, src_pad_mask, args, go_id=go_id, eos_id=eos_id, pad_id=pad_id)
            pred_line = ids_to_line(ys.squeeze(0).tolist(), tgt_id2tok, go_id, eos_id, pad_id)
            pred_text = detok_bpe_text(pred_line, lang="zh")
            
            src_text = src_lines_raw[i].strip().replace("@@", "").replace(" ", " ")
            ref_text = tgt_lines_raw[i].strip().replace("@@", "").replace(" ", "")
            
            log(f"\n[样例 {i+1}]")
            log(f"  源文(EN): {src_text}")
            log(f"  参考(ZH): {ref_text}")
            log(f"  预测(ZH): {pred_text}")
        
        log("\n" + "=" * 60)
        return

    for epoch in range(1, args.epochs + 1):
        log(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, args.device, args.check_interval, pad_id, log_fn=log
        )
        val_loss = evaluate(model, val_loader, criterion, args.device, log_fn=log)
        log(f"[Epoch {epoch}] Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.save_path)
            log(f"保存最佳模型到 {args.save_path} (val loss={best_val:.4f})")

    if args.test_after_train and has_test and os.path.exists(args.save_path):
        model.load_state_dict(torch.load(args.save_path, map_location=args.device))
        test_loss = evaluate(model, test_loader, criterion, args.device, log_fn=log)
        log(f"[Test after train] test loss: {test_loss:.4f}")
        
        if args.eval_bleu:
            bleu_score = evaluate_bleu(
                model, test_loader, args, tgt_id2tok, go_id, eos_id, pad_id, log_fn=log
            )
            log(f"[Test after train] test BLEU: {bleu_score:.2f}")

    if args.generate:
        # 用验证集前几条做示例解码，展示双语对照
        model.load_state_dict(torch.load(args.save_path, map_location=args.device))
        model.eval()
        
        # 读取原始文本用于展示
        with open(os.path.join(args.data_dir, "val.en"), "r", encoding="utf-8") as f:
            src_lines_raw = f.readlines()
        with open(os.path.join(args.data_dir, "val.zh"), "r", encoding="utf-8") as f:
            tgt_lines_raw = f.readlines()
        
        log("\n" + "=" * 60)
        log("翻译示例 (双语对照)")
        log("=" * 60)
        
        num_samples = min(5, len(val_dataset))
        for i in range(num_samples):
            src_ids, tgt_ids = val_dataset[i]
            src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0)
            src_pad_mask = src_tensor == pad_id
            
            ys = greedy_generate(model, src_tensor, src_pad_mask, args, go_id=go_id, eos_id=eos_id, pad_id=pad_id)
            pred_line = ids_to_line(ys.squeeze(0).tolist(), tgt_id2tok, go_id, eos_id, pad_id)
            pred_text = detok_bpe_text(pred_line, lang="zh")
            
            # 原始文本还原
            src_text = src_lines_raw[i].strip().replace("@@", "").replace(" ", " ")
            ref_text = tgt_lines_raw[i].strip().replace("@@", "").replace(" ", "")
            
            log(f"\n[样例 {i+1}]")
            log(f"  源文(EN): {src_text}")
            log(f"  参考(ZH): {ref_text}")
            log(f"  预测(ZH): {pred_text}")
        
        log("\n" + "=" * 60)



if __name__ == "__main__":
    main()

