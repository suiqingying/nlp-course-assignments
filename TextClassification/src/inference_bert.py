import os
import torch
import argparse
from transformers import AutoTokenizer
from bert_model import RobertaClassifier

def load_bert_model(model_path, pretrained_model_name, device):
    """
    Loads the trained RobertaClassifier model.
    """
    model = RobertaClassifier(pretrained_model_name=pretrained_model_name, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def infer_bert(model, tokenizer, sentence, device, max_len=128):
    """
    Performs inference on a single sentence using the RoBERTa model.
    """
    inputs = tokenizer(
        sentence,
        padding='max_length',  # Pad to max_len
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)

    with torch.no_grad():
        prediction = model.predict(input_ids, attention_mask, token_type_ids)
    
    return prediction.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference with a trained RoBERTa model.")
    parser.add_argument('--model_path', type=str, default='./save_model/bert_best.pt', help='Path to the saved model state dict.')
    parser.add_argument('--pretrained_model_name', type=str, default='hfl/chinese-roberta-wwm-ext', help='Name of the pretrained model.')
    parser.add_argument('--max_len', type=int, default=128, help="Max sequence length for tokenizer.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading model and tokenizer...")
    try:
        # Use local cache directory (parent of this script); do not attempt network download
        root_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        os.environ['HF_HOME'] = root_cache
        os.environ['TRANSFORMERS_CACHE'] = root_cache
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, cache_dir=root_cache, local_files_only=True)
        model = load_bert_model(args.model_path, args.pretrained_model_name, device)
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer from local cache '{root_cache}': {e}")
        print("Please ensure the pretrained model and tokenizer are present in the cache directory (no network downloads allowed).")
        exit()

    print("\n--- Interactive Inference (RoBERTa) ---")
    print("Labels: 0 -> Negative, 1 -> Positive")
    while True:
        sentence = input("Enter a sentence (or 'quit' to exit): ")
        if sentence.lower() == 'quit':
            print("Exiting interactive inference.")
            break

        prediction = infer_bert(model, tokenizer, sentence, device, args.max_len)
        label = "Positive" if prediction == 1 else "Negative"
        
        print(f"> Prediction: {label} ({prediction})")
        print("-" * 20)
