"""Download WikiText-103 val split, BPE-tokenize with GPT-2 vocab, write val.bin.

The standard cross-distribution LM benchmark for our eval table.
"""

import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wikitext-103")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    print("loading wikitext-103-raw-v1 validation split ...")
    # bare "wikitext" is no longer a valid HF dataset id; use the namespaced fork
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")

    all_tokens = []
    for example in tqdm(ds, desc="tokenizing"):
        text = example["text"]
        if text.strip():
            all_tokens.extend(enc.encode_ordinary(text))

    print(f"wikitext-103 val: {len(all_tokens):,} BPE tokens")
    out_path = os.path.join(DATA_DIR, "val.bin")
    np.array(all_tokens, dtype=np.uint16).tofile(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
