#!/usr/bin/env python3
# tokenize_transitions.py -- GPT-2 BPE tokenize data into the flat int32
# token binary the D3D11 trainer eats (gpt2_trainer.exe --data tokens.bin: reads block-length
# int32 sequences).  Each example is separated by <|endoftext|> (50256).
#
# KUHUL special tokens (50260-50281) are inserted by detecting the literal tag strings
# before tiktoken encoding -- tiktoken never sees them, so disallowed_special=() is still used
# for the remainder of each segment.
#
# Modes:
#   default  — reads --field (default "text") from each JSONL line
#   --chat   — reads chat pairs (question/response, messages, prompt/completion, instruction/output,
#              CSV question/answer) and wraps with <USER>...</USER><INSTRUCT>...</INSTRUCT>
#
# Input accepts comma-separated file paths in both modes.
# CSV files with --chat are read via csv.DictReader; JSONL uses json.loads per line.
#
# Usage:
#   python tools/tokenize_transitions.py <in.jsonl> <out.bin> [--limit N]
#   python tools/tokenize_transitions.py <in.jsonl[,in2.jsonl,...]> <out.bin> --chat [--limit N]

import sys, re, json, argparse, csv, numpy as np, tiktoken

# ── Fold-phase tokens (Pop/Sek/Xul/Chen navigation) ─────────────────────────
KUHUL_TOKENS = {
    "<AGENT>":      50260,   # Sek  — execute / implement
    "</AGENT>":     50261,
    "<THINK>":      50262,   # Pop  — observe / read types
    "</THINK>":     50263,
    "<TOOL_CALL>":  50264,   # Chen — verify / error-fix
    "</TOOL_CALL>": 50265,
    "<INSTRUCT>":   50266,   # Xul  — emit / export surface
    "</INSTRUCT>":  50267,
    "<USER>":       50268,   # user turn boundary
    "</USER>":      50269,
    # ── TypeScript grammar sub-tokens (lanes) ──────────────────────────────
    # Bone 0  — Declaration lane (interface/type/class header)
    "<TS_INTERFACE>":  50270,
    "</TS_INTERFACE>": 50271,
    "<TS_TYPE>":       50272,
    "</TS_TYPE>":      50273,
    # Bone 1+2 — Implementation lane (function/class body)
    "<TS_FUNCTION>":   50274,
    "</TS_FUNCTION>":  50275,
    "<TS_CLASS>":      50276,
    "</TS_CLASS>":     50277,
    # Bone 3  — Verification lane (tsc error / fix pair)
    "<TS_ERROR>":      50278,
    "</TS_ERROR>":     50279,
    "<TS_FIX>":        50280,
    "</TS_FIX>":       50281,
    # ── Fold role tokens — chat turn boundaries keyed by K'UHUL phase ─────────
    # These are ROLE tokens (like <USER>/<ASSISTANT>), not lane tokens.
    # The opening fold sets the cognitive mode for that turn; <XUL> is always
    # the model's emit/response token.
    #
    #   <POP>…</POP>   observe / describe / read context
    #   <WO>…</WO>     plan / schedule / organize
    #   <YAX>…</YAX>   explore / branch / generate alternatives
    #   <SEK>…</SEK>   code / implement / execute
    #   <CHEN>…</CHEN> verify / validate / debug
    #   <XUL>…</XUL>   emit / output  (model response — analogous to ASSISTANT)
    "<POP>":    50282,
    "</POP>":   50283,
    "<WO>":     50284,
    "</WO>":    50285,
    "<YAX>":    50286,
    "</YAX>":   50287,
    "<SEK>":    50288,
    "</SEK>":   50289,
    "<CHEN>":   50290,
    "</CHEN>":  50291,
    "<XUL>":    50292,
    "</XUL>":   50293,
}

# Sort longer tags first so </TOOL_CALL> matches before a hypothetical </TOOL>
_KUHUL_TAGS_SORTED = sorted(KUHUL_TOKENS.keys(), key=len, reverse=True)
KUHUL_SPLIT_RE = re.compile("(" + "|".join(re.escape(t) for t in _KUHUL_TAGS_SORTED) + ")")

EXTENDED_VOCAB = max(KUHUL_TOKENS.values()) + 1  # 50294

EOT_ID = 50256  # <|endoftext|>


def encode_with_kuhul(enc, text: str) -> list:
    """Encode text, inserting KUHUL token IDs for special tags."""
    ids = []
    for part in KUHUL_SPLIT_RE.split(text):
        if not part:
            continue
        if part in KUHUL_TOKENS:
            ids.append(KUHUL_TOKENS[part])
        else:
            ids.extend(enc.encode(part, disallowed_special=()))
    return ids


# ── Chat pair extraction (same formats as filter_rlhf.py) ────────────────────

def _walk_mapping(mapping: dict) -> list:
    def node_key(item):
        try:
            return int(item[0])
        except Exception:
            return 0
    ordered = sorted(mapping.items(), key=node_key)
    turns = []
    for _, node in ordered:
        if not isinstance(node, dict):
            continue
        frags = (node.get('message') or {}).get('fragments', [])
        for frag in frags:
            ftype = (frag.get('type') or '').upper()
            content = (frag.get('content') or '').strip()
            if ftype in ('REQUEST', 'RESPONSE') and content:
                turns.append((ftype, content))
    pairs = []
    i = 0
    while i < len(turns) - 1:
        if turns[i][0] == 'REQUEST' and turns[i+1][0] == 'RESPONSE':
            pairs.append((turns[i][1], turns[i+1][1]))
            i += 2
        else:
            i += 1
    return pairs


def extract_pairs_from_obj(obj: dict) -> list:
    """Return list of (prompt, completion) from one parsed JSON object."""
    # DeepSeek mapping tree
    record = obj.get('record', {})
    if isinstance(record, dict) and 'mapping' in record:
        pairs = _walk_mapping(record['mapping'])
        if pairs:
            return pairs

    # chatml messages array — yield all user/assistant turns
    if 'messages' in obj:
        msgs = obj['messages']
        pairs = []
        i = 0
        while i < len(msgs) - 1:
            if msgs[i].get('role') == 'user' and msgs[i+1].get('role') == 'assistant':
                u = (msgs[i].get('content') or '').strip()
                a = (msgs[i+1].get('content') or '').strip()
                if u and a:
                    pairs.append((u, a))
                i += 2
            else:
                i += 1
        if pairs:
            return pairs

    # RLHF chosen
    if 'chosen' in obj:
        chosen = obj['chosen']
        if isinstance(chosen, list):
            u = next((m.get('content', '') for m in chosen if m.get('role') == 'user'), '')
            a = next((m.get('content', '') for m in chosen if m.get('role') == 'assistant'), '')
            if u and a:
                return [(u.strip(), a.strip())]

    # Flat field variants
    q = (obj.get('prompt') or obj.get('question') or obj.get('input') or
         obj.get('instruction') or '').strip()
    a = (obj.get('completion') or obj.get('response') or obj.get('answer') or
         obj.get('output') or '').strip()
    if q and a:
        return [(q, a)]

    return []


# ── Language detection for --group-by-lang ───────────────────────────────────
# Priority order when writing the packed .bin: foundational types first,
# specialised/domain types last. Within each group, coherent blocks reduce
# gradient noise and act as a natural curriculum.
LANG_ORDER = [
    'python', 'javascript', 'typescript', 'java', 'cpp',
    'rust', 'go', 'csharp', 'sql', 'shell', 'html', 'css', 'other',
]

_LANG_ALIASES = {
    'python': 'python', 'py': 'python',
    'javascript': 'javascript', 'js': 'javascript',
    'typescript': 'typescript', 'ts': 'typescript',
    'java': 'java',
    'cpp': 'cpp', 'c++': 'cpp', 'c': 'cpp',
    'rust': 'rust', 'rs': 'rust',
    'go': 'go', 'golang': 'go',
    'csharp': 'csharp', 'cs': 'csharp', 'c#': 'csharp', 'dotnet': 'csharp',
    'sql': 'sql',
    'bash': 'shell', 'sh': 'shell', 'shell': 'shell', 'powershell': 'shell', 'ps': 'shell',
    'html': 'html', 'htm': 'html',
    'css': 'css', 'scss': 'css', 'less': 'css',
}

_CODE_BLOCK_RE = re.compile(r'```(\w+)', re.IGNORECASE)

def detect_lang(prompt: str, completion: str) -> str:
    """Detect primary programming language from a prompt/completion pair."""
    text = prompt + '\n' + completion
    # 1. Explicit fenced code block tags are most reliable
    for m in _CODE_BLOCK_RE.findall(text):
        lang = _LANG_ALIASES.get(m.lower())
        if lang:
            return lang
    # 2. Keyword heuristics as fallback
    tl = text.lower()
    if 'def ' in tl and ('python' in tl or 'import ' in tl or ':\n' in tl):
        return 'python'
    if re.search(r'\bfunction\b|\bconst\b|\blet\b|\bvar\b', tl) and (
            'javascript' in tl or '.js' in tl or 'node' in tl or '=>' in tl):
        return 'javascript'
    if 'interface ' in tl and ('typescript' in tl or '.ts' in tl or ': string' in tl):
        return 'typescript'
    if re.search(r'\bpublic\s+class\b|\bpublic\s+static\s+void\b', tl):
        return 'java'
    if re.search(r'\bint\s+main\b|\b#include\b|\bstd::\b', tl):
        return 'cpp'
    if 'fn ' in tl and ('rust' in tl or 'println!' in tl or '-> result' in tl):
        return 'rust'
    if re.search(r'\bfunc\b', tl) and ('go' in tl or 'golang' in tl or 'fmt.print' in tl):
        return 'go'
    if re.search(r'\bselect\b|\bfrom\b|\bwhere\b', tl) and 'sql' in tl:
        return 'sql'
    if re.search(r'#!/bin|bash|chmod|grep|awk|sed', tl):
        return 'shell'
    return 'other'


def iter_chat_file(path: str):
    """Yield (prompt, completion) pairs from a JSONL or CSV file."""
    enc_lower = path.lower()
    if enc_lower.endswith('.csv'):
        with open(path, encoding='utf-8-sig', newline='', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = (row.get('question') or row.get('prompt') or row.get('input') or '').strip()
                a = (row.get('answer') or row.get('completion') or row.get('response') or
                     row.get('output') or '').strip()
                if q and a:
                    yield (q, a)
    else:
        with open(path, encoding='utf-8-sig', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                yield from extract_pairs_from_obj(obj)


def encode_chat_pair(enc, prompt: str, completion: str) -> list:
    """Encode a chat pair as <USER>prompt</USER><INSTRUCT>completion</INSTRUCT> EOT."""
    ids = []
    ids.append(KUHUL_TOKENS["<USER>"])
    ids.extend(encode_with_kuhul(enc, prompt))
    ids.append(KUHUL_TOKENS["</USER>"])
    ids.append(KUHUL_TOKENS["<INSTRUCT>"])
    ids.extend(encode_with_kuhul(enc, completion))
    ids.append(KUHUL_TOKENS["</INSTRUCT>"])
    ids.append(EOT_ID)
    return ids


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", help="Input file(s) — comma-separated for multiple")
    ap.add_argument("outfile")
    ap.add_argument("--limit", type=int, default=0, help="Only tokenize first N examples (probe)")
    ap.add_argument("--field", default="text", help="Field name in default (non-chat) mode")
    ap.add_argument("--no-kuhul", action="store_true", help="Disable KUHUL tag injection")
    ap.add_argument("--chat", action="store_true",
                    help="Chat mode: extract prompt/completion pairs and wrap with KUHUL turn tokens")
    ap.add_argument("--group-by-lang", action="store_true",
                    help="In --chat mode: detect programming language of each pair and sort all pairs "
                         "by language before packing. Creates coherent 128-token blocks (all Python "
                         "together, then JS, then C++...) which stabilises gradients and acts as a "
                         "natural curriculum. Order: " + ", ".join(LANG_ORDER))
    ap.add_argument("--pack", type=int, default=0, metavar="N",
                    help="Pack flat token stream into fixed-length blocks of N tokens and write "
                         "the (n_seq uint32, seq_len uint32, data...) header format that "
                         "gpt2_trainer.exe expects. Blocks shorter than N are zero-padded. "
                         "Incompatible with --field mode; works in both default and --chat modes.")
    a = ap.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    ids = []
    n = 0
    kuhul_count = 0

    paths = [p.strip() for p in a.infile.split(',') if p.strip()]

    if a.chat:
        if a.group_by_lang:
            # Collect all pairs grouped by language, then encode in curriculum order
            from collections import defaultdict
            lang_buckets = defaultdict(list)
            raw_n = 0
            for path in paths:
                for prompt, completion in iter_chat_file(path):
                    lang = detect_lang(prompt, completion)
                    lang_buckets[lang].append((prompt, completion))
                    raw_n += 1
                    if raw_n % 5000 == 0:
                        print(f"  {raw_n:,} pairs collected...", flush=True)
                    if a.limit and raw_n >= a.limit:
                        break
                if a.limit and raw_n >= a.limit:
                    break
            # Report bucket sizes
            for lang in LANG_ORDER:
                cnt = len(lang_buckets.get(lang, []))
                if cnt:
                    print(f"  {lang:<12} {cnt:>6,} pairs", flush=True)
            # Encode in curriculum order
            for lang in LANG_ORDER:
                for prompt, completion in lang_buckets.get(lang, []):
                    toks = encode_chat_pair(enc, prompt, completion)
                    ids.extend(toks)
                    kuhul_count += sum(1 for tok in toks if tok >= 50260)
                    n += 1
        else:
            for path in paths:
                for prompt, completion in iter_chat_file(path):
                    toks = encode_chat_pair(enc, prompt, completion)
                    ids.extend(toks)
                    kuhul_count += sum(1 for tok in toks if tok >= 50260)
                    n += 1
                    if n % 1000 == 0:
                        print(f"  {n:,} pairs tokenized...", flush=True)
                    if a.limit and n >= a.limit:
                        break
                if a.limit and n >= a.limit:
                    break
    else:
        for path in paths:
            with open(path, encoding='utf-8-sig', errors='replace') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    t = rec.get(a.field)
                    if not t:
                        continue
                    if a.no_kuhul:
                        toks = enc.encode(t, disallowed_special=())
                    else:
                        toks = encode_with_kuhul(enc, t)
                        kuhul_count += sum(1 for tok in toks if tok >= 50260)
                    ids.extend(toks)
                    ids.append(EOT_ID)
                    n += 1
                    if a.limit and n >= a.limit:
                        break
            if a.limit and n >= a.limit:
                break

    if not ids:
        print("[warn] 0 tokens written — check input paths and field names", file=sys.stderr)
        sys.exit(1)

    arr = np.asarray(ids, dtype=np.int32)
    max_valid = EXTENDED_VOCAB - 1  # 50281
    assert arr.min() >= 0 and arr.max() <= max_valid, \
        f"token OOB: min={arr.min()} max={arr.max()} (expected 0..{max_valid})"

    if a.pack > 0:
        # Write trainer-headed format: uint32 n_seq, uint32 seq_len, then data
        seq_len = a.pack
        n_seq = (len(arr) + seq_len - 1) // seq_len
        packed = np.zeros(n_seq * seq_len, dtype=np.int32)
        packed[:len(arr)] = arr  # zero-pad last block if needed
        import struct
        with open(a.outfile, 'wb') as f:
            f.write(struct.pack('<II', n_seq, seq_len))
            packed.tofile(f)
        mb = packed.nbytes / 1e6 + 8 / 1e6
        print(f"[ok] {a.outfile}: {n_seq:,} sequences x {seq_len} tokens ({mb:.1f} MB) [pack={seq_len}]")
    else:
        arr.tofile(a.outfile)
        mb = arr.nbytes / 1e6
        mode_str = "chat" if a.chat else f"field={a.field}"
        print(f"[ok] {a.outfile}: {len(arr):,} tokens from {n:,} examples ({mb:.1f} MB int32) [{mode_str}]")

    if not a.no_kuhul:
        print(f"     KUHUL tokens injected: {kuhul_count:,} ({100*kuhul_count/len(arr):.2f}% of stream)")
    for blk in (64, 128, 256):
        print(f"     block={blk} -> {len(arr)//blk:,} sequences")


if __name__ == "__main__":
    main()
