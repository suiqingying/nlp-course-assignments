import torch
import json
import argparse
import os
from collections import namedtuple
import jieba
from common import TextCNN

# Convert config dict to a simple object with attribute access
def _to_config(obj):
    if isinstance(obj, dict):
        return namedtuple('config', obj.keys())(**obj)
    return obj


# Load the model checkpoint
def _try_load_checkpoint(path, device):
    """Attempt to load a torch checkpoint; return dict or raise."""
    # Try new zip format first
    ckpt = torch.load(path, map_location=device)
    return ckpt


def load_model(checkpoint_path, vocab_path, device):
    if not torch.cuda.is_available():
        device = torch.device('cpu')  # fallback to CPU if CUDA not available

    candidate_paths = [checkpoint_path]
    # If default path is invalid, also try an alternative (best.pt) commonly used in this project
    alt_path = '../save_model/best.pt'
    if alt_path not in candidate_paths:
        candidate_paths.append(alt_path)

    ckpt = None
    errors = []
    for path in candidate_paths:
        if not os.path.exists(path):
            errors.append(f"{path}: file not found")
            continue
        try:
            ckpt = _try_load_checkpoint(path, device)
            checkpoint_path = path  # use the successful one
            break
        except Exception as e_zip:
            # Try legacy serialization flag
            try:
                ckpt = torch.load(path, map_location=device, _use_new_zipfile_serialization=False)
                checkpoint_path = path
                break
            except Exception as e_legacy:
                errors.append(f"{path}: {e_zip} / legacy: {e_legacy}")
                continue

    if ckpt is None:
        raise RuntimeError("Failed to load any checkpoint. Errors: " + " | ".join(errors))

    if not isinstance(ckpt, dict) or 'config' not in ckpt or 'model_state_dict' not in ckpt:
        raise ValueError(f"Checkpoint {checkpoint_path} is not a valid TextCNN checkpoint (missing 'config' or 'model_state_dict').")

    config = _to_config(ckpt['config'])
    model = TextCNN(config)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    print(f"Loaded checkpoint from: {checkpoint_path}")
    return model, vocab

# Preprocess the input sentence
def preprocess_sentence(sentence, vocab, min_len=1):
    """
    将输入文本转换为 id，并在必要时填充到最小长度，避免卷积核大于序列长度。
    """
    unk_token = 'UNK' if 'UNK' in vocab else '<UNK>'
    pad_idx = vocab.get('PAD', 8019)  # 与训练时的 padding_idx 对齐

    tokens = [vocab.get(char, vocab.get(unk_token, 0)) for char in sentence]
    if len(tokens) < min_len:
        tokens.extend([pad_idx] * (min_len - len(tokens)))

    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension

# Perform inference
def infer(model, sentence, vocab, device):
    # 确保长度至少为最大卷积核大小，避免 RuntimeError
    min_len = max(getattr(model, 'filter_sizes', [1]))
    input_tensor = preprocess_sentence(sentence, vocab, min_len=min_len).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
    return prediction

if __name__ == '__main__':
    checkpoint_path = '../save_model/bert_best.pt'  # Path to the saved model
    vocab_path = '../dataset/vocab.json'  # Path to the vocabulary file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading model...")
    model, vocab = load_model(checkpoint_path, vocab_path, device)
    print("Model loaded successfully.")

    print("\n--- Interactive Inference ---")
    while True:
        sentence = input("Enter a sentence (or 'quit' to exit): ")
        if sentence.lower() == 'quit':
            print("Exiting interactive inference.")
            break

        prediction = infer(model, sentence, vocab, device)
        label = "Positive" if prediction == 1 else "Negative"
        print(f"Prediction: {label}\n")

