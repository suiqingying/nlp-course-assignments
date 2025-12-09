#!/usr/bin/env python3
"""
Simple helper to download a Hugging Face model + tokenizer into the script directory cache.
Usage:
  python download_model.py --pretrained hfl/chinese-roberta-wwm-ext [--force]

Behavior:
- By default, tries to load locally first (no network). If not present, downloads into script directory.
- If `--force` is passed, always downloads (overwrites cache behavior handled by transformers).
- Does NOT require CUDA; runs in CPU-only environment.
"""
import os
import argparse
import logging
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, required=True, help='Pretrained model name or path')
    parser.add_argument('--cache-dir', type=str, default=os.path.abspath(os.path.dirname(__file__)), help='Directory to use for HF cache')
    parser.add_argument('--force', action='store_true', help='Force download even if local files exist')
    args = parser.parse_args()

    os.environ['HF_HOME'] = args.cache_dir
    os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
    logging.info(f"Using cache dir: {args.cache_dir}")

    name = args.pretrained

    # Check local presence by trying local-only load
    have_local = False
    if not args.force:
        try:
            logging.info(f"Checking for local tokenizer for '{name}' (local_files_only=True)")
            AutoTokenizer.from_pretrained(name, local_files_only=True, cache_dir=args.cache_dir)
            logging.info("Tokenizer found locally")

            logging.info(f"Checking for local model for '{name}' (local_files_only=True)")
            AutoModel.from_pretrained(name, local_files_only=True, cache_dir=args.cache_dir)
            logging.info("Model found locally")

            have_local = True
        except Exception as e:
            logging.info(f"Local check failed or not complete: {e}")
            have_local = False

    if have_local and not args.force:
        logging.info(f"Model and tokenizer already present in cache ({args.cache_dir}). Nothing to download.")
        return

    # Otherwise download (or force)
    logging.info(f"Downloading tokenizer and model for '{name}' into cache dir...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=args.cache_dir, local_files_only=False)
        model = AutoModel.from_pretrained(name, cache_dir=args.cache_dir, local_files_only=False)
        logging.info("Download complete.")
        logging.info(f"Tokenizer saved in cache dir; model saved in cache dir: {args.cache_dir}")
    except Exception as e:
        logging.error(f"Failed to download model or tokenizer: {e}")
        raise

if __name__ == '__main__':
    main()
