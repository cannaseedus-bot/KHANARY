#!/usr/bin/env python3
"""
jsonl_to_tokens.py — Tokenize JSONL training data for gpt2_trainer.exe

Reads {input, output} JSONL records, tokenizes with GPT-2 BPE (tiktoken),
packs into fixed-length sequences, writes binary:
  [uint32 n_seq][uint32 seq_len][int32 tokens...  (n_seq * seq_len)]

Usage:
  python jsonl_to_tokens.py --input combined_train2.jsonl --out tokens_train2.bin
  python jsonl_to_tokens.py --input xshard_train.jsonl   --out tokens_xshard.bin --seq-len 256
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

try:
    import tiktoken
except ImportError:
    print("Installing tiktoken...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken", "-q"])
    import tiktoken

BIN_DIR  = Path(r"C:\Users\canna\.gpu_trainer\bin")
TRAIN_DIR = Path(r"C:\Users\canna\.gpu_trainer\trainer")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   type=Path, required=True,
                        help="Input JSONL file or directory of JSONL shards with {input, output} records")
    parser.add_argument("--out",     type=Path, default=None,
                        help="Output .bin file (default: bin/<stem>.bin)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Fixed sequence length (default: 128, use 256 for medium)")
    parser.add_argument("--max-seq", type=int, default=None,
                        help="Cap number of sequences (for quick tests)")
    parser.add_argument("--max-chunk", type=int, default=None,
                        help="When --input is a shard directory, include chunk_N.jsonl through this number")
    parser.add_argument("--profile", choices=["all", "ast-json"], default="all",
                        help="all=keep usable conversations; ast-json=keep coding/AST/JSON-bearing records")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="Path to a HuggingFace tokenizer.json (uses Qwen/LLaMA vocab instead of GPT-2 BPE)")
    args = parser.parse_args()

    # Resolve paths
    # Project-relative paths must stay in the active checkout.  The legacy
    # .gpu_trainer tree remains a fallback for old invocations only.
    if args.input.is_absolute():
        src = args.input
    else:
        project_src = Path.cwd() / args.input
        src = project_src if project_src.exists() else TRAIN_DIR / args.input
    if not src.exists():
        print(f"ERROR: {src} not found")
        sys.exit(1)

    if src.is_dir():
        sources = sorted(src.glob("chunk_*.jsonl"))
        if not sources:
            sources = sorted(src.glob("*.jsonl"))
        if args.max_chunk is not None:
            sources = [
                item for item in sources
                if (match := re.fullmatch(r"chunk_(\d+)\.jsonl", item.name, re.IGNORECASE))
                and int(match.group(1)) <= args.max_chunk
            ]
        if not sources:
            print(f"ERROR: no JSONL files found in {src}")
            sys.exit(1)
    else:
        sources = [src]

    out = args.out
    if out is None:
        out = BIN_DIR / (src.stem + ".bin")
    elif not out.is_absolute():
        out = Path.cwd() / out

    S = args.seq_len

    if args.tokenizer:
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tokenizers", "-q"])
            from tokenizers import Tokenizer as HFTokenizer
        _hf = HFTokenizer.from_file(str(args.tokenizer))
        _eos_candidates = ["<|endoftext|>", "</s>", "<eos>", "<|im_end|>"]
        _vocab = _hf.get_vocab()
        eot = next((v for k, v in _vocab.items() if k in _eos_candidates), 0)
        def _encode(text):
            return _hf.encode(text).ids
        print(f"Tokenizer: {args.tokenizer.name}  vocab={len(_vocab)}  eot={eot}")
    else:
        # GPT-2 tokenizer (same vocab as gpt2_medium)
        _gpt2 = tiktoken.get_encoding("gpt2")
        eot = _gpt2.eot_token   # 50256 = <|endoftext|>
        def _encode(text):
            return _gpt2.encode(text, allowed_special={"<|endoftext|>"})

    print(f"Tokenizing {len(sources)} source(s) from {src} -> {out}  seq_len={S}")

    # Tokenize all records into one flat token stream
    token_stream = []
    n_records = 0
    n_selected = 0

    def normalize_json_fences(text):
        def replace(match):
            language = (match.group(1) or "").lower()
            body = match.group(2).strip()
            if language not in ("json", "jsonc"):
                return match.group(0)
            try:
                value = json.loads(body)
                return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False) + "\n```"
            except Exception:
                return match.group(0)
        return re.sub(r"```(jsonc?|JSONC?)\s*\n(.*?)```", replace, text, flags=re.DOTALL)

    def record_text(record):
        messages = record.get("messages")
        if isinstance(messages, list):
            parts = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", "")).strip().lower()
                content = message.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(str(item) for item in content)
                content = str(content).strip()
                if content:
                    parts.append(f"{role or 'message'}: {content}")
            return "\n".join(parts)
        inp = str(record.get("input", record.get("prompt", "")) or "").strip()
        out_text = str(record.get("output", record.get("response", record.get("completion", ""))) or "").strip()
        if inp and out_text:
            return f"user: {inp}\nassistant: {out_text}"
        return out_text or inp or str(record.get("text", "") or "").strip()

    def is_ast_json_record(text):
        lower = text.lower()
        return bool(
            "```" in text
            or "abstract syntax tree" in lower
            or "ast" in lower
            or re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
            or '"type"' in lower
            or '"body"' in lower
        )

    for source in sources:
        print(f"  source: {source.name}")
        with open(source, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    text = record_text(r)
                    if not text or (args.profile == "ast-json" and not is_ast_json_record(text)):
                        continue
                    text = normalize_json_fences(text)
                    toks = _encode(text)
                    toks.append(eot)
                    token_stream.extend(toks)
                    n_records += 1
                    n_selected += 1
                    if n_records % 50000 == 0:
                        print(f"  {n_records:,} records, {len(token_stream):,} tokens ...")
                except Exception:
                    pass

    print(f"  Total: {n_records:,} selected records, {len(token_stream):,} tokens")

    # Pack into sequences of length S (drop last incomplete sequence)
    n_seq = len(token_stream) // S
    if args.max_seq:
        n_seq = min(n_seq, args.max_seq)

    print(f"  Packing into {n_seq:,} sequences × {S} tokens")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(struct.pack("<II", n_seq, S))
        for i in range(n_seq):
            chunk = token_stream[i*S : (i+1)*S]
            f.write(struct.pack(f"<{S}i", *chunk))

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  Written: {out}  ({size_mb:.1f} MB)")
    print()
    print("Run trainer:")
    print(f'  gpt2_trainer.exe \\')
    print(f'    --model "C:\\Users\\canna\\.gpu_trainer\\trainer\\gpt2_medium_dx11\\model.safetensors" \\')
    print(f'    --data  "{out}" \\')
    print(f'    --out   "C:\\Users\\canna\\.gpu_trainer\\trainer\\gpt2_medium_dx11\\model_ft.safetensors" \\')
    print(f'    --steps 5000 --batch 4 --block {S} --lr 3e-5')


if __name__ == "__main__":
    main()
