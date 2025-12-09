"""
mBART-50 微调/推理脚本 - 英语→中文翻译

使用示例:
=========

1. 训练（微调）:
   python mbart_finetune.py --epochs 5 --batch_size 8 --bf16

2. 评估已微调的模型:
   python mbart_finetune.py --predict_only --checkpoint mbart_ckpt

3. 交互式翻译（使用微调后的模型）:
   python mbart_finetune.py --interactive --checkpoint mbart_ckpt

4. 直接使用预训练模型（不微调）:
   python mbart_finetune.py --no_finetune --interactive
   python mbart_finetune.py --no_finetune --predict_only

5. 对比测试（预训练 vs 微调）:
   python mbart_finetune.py --no_finetune --predict_only        # 预训练模型
   python mbart_finetune.py --predict_only --checkpoint mbart_ckpt  # 微调后模型
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset, DatasetDict  # type: ignore[import]
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
import evaluate  # huggingface evaluate library


def debpe(text: str, lang: str = "en") -> str:
    """将 @@ 分隔的 BPE 文本还原为原始文本"""
    # @@ 是分隔符，去掉后 token 直接拼接
    text = text.replace("@@", "")
    # 对于中文，去掉空格；对于英文，保留空格
    if lang.lower() in ("zh", "zh_cn", "chinese"):
        text = text.replace(" ", "")
    return text.strip()


def read_parallel(src_path: str, tgt_path: str, debpe_src: bool = True, debpe_tgt: bool = True) -> List[Dict[str, str]]:
    """读取平行语料，可选择是否还原 BPE"""
    with open(src_path, "r", encoding="utf-8") as f_src, open(tgt_path, "r", encoding="utf-8") as f_tgt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()
    assert len(src_lines) == len(tgt_lines), "源/目标句子数量不一致"
    
    result = []
    for s, t in zip(src_lines, tgt_lines):
        src_text = debpe(s.strip(), lang="en") if debpe_src else s.strip()
        tgt_text = debpe(t.strip(), lang="zh") if debpe_tgt else t.strip()
        result.append({"src": src_text, "tgt": tgt_text})
    return result


@dataclass
class TokenizerWrapper:
    tokenizer: MBart50TokenizerFast
    src_lang: str
    tgt_lang: str
    max_len: int

    def __call__(self, batch):
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang

        # Use text_target to avoid deprecated as_target_tokenizer and ensure tgt lang tokens are set
        model_inputs = self.tokenizer(
            batch["src"],
            text_target=batch["tgt"],
            max_length=self.max_len,
            truncation=True,
        )
        return model_inputs


def load_datasets(data_dir: str, max_len: int, src_lang: str, tgt_lang: str):
    """加载数据集，自动还原 BPE 格式"""
    train = read_parallel(
        os.path.join(data_dir, "train.en"), 
        os.path.join(data_dir, "train.zh"),
        debpe_src=True, debpe_tgt=True
    )
    val = read_parallel(
        os.path.join(data_dir, "val.en"), 
        os.path.join(data_dir, "val.zh"),
        debpe_src=True, debpe_tgt=True
    )
    data = DatasetDict(
        {
            "train": Dataset.from_list(train),
            "validation": Dataset.from_list(val),
        }
    )
    print(f"训练集: {len(train)} 条, 验证集: {len(val)} 条")
    print(f"样例 src: {train[0]['src'][:80]}...")
    print(f"样例 tgt: {train[0]['tgt'][:80]}...")
    return data


def parse_args():
    ap = argparse.ArgumentParser(description="mBART-50 微调中英翻译（en->zh）")
    # 模型选择：mbart-large-50 适合多语言，也可用 Helsinki-NLP/opus-mt-en-zh 等更轻量模型
    ap.add_argument("--model_name", default="facebook/mbart-large-50-many-to-many-mmt",
                    help="预训练模型，推荐: facebook/mbart-large-50-many-to-many-mmt")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="mbart_ckpt")
    ap.add_argument("--batch_size", type=int, default=8, help="4090可用8-16")
    ap.add_argument("--grad_accum", type=int, default=2, help="梯度累积步数")
    ap.add_argument("--max_len", type=int, default=128, help="最大序列长度")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--fp16", action="store_true", default=True, help="启用混合精度(4090默认开启)")
    ap.add_argument("--bf16", action="store_true", help="使用bf16(4090支持)")
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup比例")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--src_lang", default="en_XX")
    ap.add_argument("--tgt_lang", default="zh_CN")
    ap.add_argument("--predict_only", action="store_true", help="仅加载已有权重做验证集推理")
    ap.add_argument("--checkpoint", default=None, help="加载指定 checkpoint 进行评估/继续训练")
    ap.add_argument("--interactive", action="store_true", help="交互式翻译模式（加载已训权重）")
    ap.add_argument("--early_stopping", type=int, default=3, help="早停patience，0表示不启用")
    ap.add_argument("--no_finetune", action="store_true", help="直接使用预训练模型，不加载微调权重")
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    tokenizer = MBart50TokenizerFast.from_pretrained(args.model_name)
    model = MBartForConditionalGeneration.from_pretrained(args.model_name)
    model = model.to(device)  # type: ignore[arg-type]

    data = load_datasets(args.data_dir, args.max_len, args.src_lang, args.tgt_lang)
    tok_wrapper = TokenizerWrapper(tokenizer, args.src_lang, args.tgt_lang, args.max_len)
    tokenized = data.map(tok_wrapper, batched=True, remove_columns=["src", "tgt"])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    # BLEU 评估
    bleu_metric = evaluate.load("sacrebleu")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # 将 -100 替换为 pad_token_id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # sacrebleu 需要 references 是 list of list
        decoded_labels = [[label] for label in decoded_labels]
        
        result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    # 根据硬件选择精度
    use_fp16 = args.fp16 and not args.bf16
    use_bf16 = args.bf16

    # 设置目标语言的 forced_bos_token_id，确保生成中文
    forced_bos_token_id = tokenizer.lang_code_to_id[args.tgt_lang]

    training_args = Seq2SeqTrainingArguments(  # type: ignore[call-arg]
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="steps",  # type: ignore[arg-type]
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        generation_max_length=args.max_len,
        generation_num_beams=args.beam,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        save_total_limit=3,
    )
    
    # 设置模型的生成配置，指定目标语言
    model.config.forced_bos_token_id = forced_bos_token_id

    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping))

    trainer = Seq2SeqTrainer(  # type: ignore[call-arg]
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,  # type: ignore[arg-type]
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # 仅预测模式
    if args.predict_only:
        if args.no_finetune:
            print("predict_only (使用原始预训练模型，未微调)")
            # model 已经是预训练模型，不需要重新加载
        else:
            ckpt = args.checkpoint or args.output_dir
            print(f"predict_only, loading {ckpt}")
            trainer.model = MBartForConditionalGeneration.from_pretrained(ckpt).to(device)  # type: ignore[arg-type]
        
        preds = trainer.predict(tokenized["validation"], max_length=args.max_len, num_beams=args.beam)
        metrics = preds.metrics or {}
        print("=" * 50)
        mode_str = "预训练模型(未微调)" if args.no_finetune else "微调后模型"
        print(f"验证集评估结果 [{mode_str}]:")
        print(f"  Loss: {metrics.get('test_loss', 'n/a')}")
        print(f"  BLEU: {metrics.get('test_bleu', 'n/a')}")
        print("=" * 50)
        
        # 显示几个翻译样例（含参考译文对比）
        if preds.predictions is not None:
            print("\n翻译样例:")
            decoded_preds = tokenizer.batch_decode(preds.predictions[:5], skip_special_tokens=True)
            val_data = data["validation"]
            for i, pred in enumerate(decoded_preds):
                src = val_data[i]["src"][:50] + "..." if len(val_data[i]["src"]) > 50 else val_data[i]["src"]
                ref = val_data[i]["tgt"][:50] + "..." if len(val_data[i]["tgt"]) > 50 else val_data[i]["tgt"]
                print(f"  [{i+1}] 源文: {src}")
                print(f"      参考: {ref}")
                print(f"      预测: {pred}")
                print()
        return

    # 交互模式：加载权重后，读取用户输入，直接翻译
    if args.interactive:
        if args.no_finetune:
            print("interactive (使用原始预训练模型，未微调)")
            # model 已经是预训练模型
        else:
            ckpt = args.checkpoint or args.output_dir
            print(f"interactive, loading {ckpt}")
            model = MBartForConditionalGeneration.from_pretrained(ckpt).to(device)  # type: ignore[arg-type]
        
        tokenizer.src_lang = args.src_lang
        mode_str = "预训练模型" if args.no_finetune else "微调模型"
        print(f"[{mode_str}] 输入英文句子（空行退出）:")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出。")
                break
            if not line:
                print("退出。")
                break
            enc = tokenizer(line, return_tensors="pt", max_length=args.max_len, truncation=True).to(device)
            generated_tokens = model.generate(
                **enc,
                forced_bos_token_id=tokenizer.lang_code_to_id[args.tgt_lang],
                max_length=args.max_len,
                num_beams=args.beam,
            )
            out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            print(out[0])
        return

    # 如果指定了 no_finetune 但没有指定 predict_only 或 interactive，提示用户
    if args.no_finetune:
        print("错误: --no_finetune 需要配合 --predict_only 或 --interactive 使用")
        print("示例: python mbart_finetune.py --no_finetune --interactive")
        return

    trainer.train(resume_from_checkpoint=args.checkpoint)
    trainer.save_model(args.output_dir)
    trainer.save_state()
    print("finished training, evaluating best model...")
    eval_metrics = trainer.evaluate(max_length=args.max_len, num_beams=args.beam)
    print(eval_metrics)


if __name__ == "__main__":
    main()

