#!/usr/bin/env python3
"""
prep_chat_corpus.py -- normalize diverse chat JSONL schemas to {"text": "..."} JSONL
                       ready for tokenize_transitions.py

Supported schemas (auto-detected per file from first non-empty line):
  A  {"text": "..."}
  B  {"messages": [{"role":"...", "content":"..."}]}   (ultrachat, opus46)
  C  {"prompt": "...", "response": "..."}              (yi34B, qwen14B, sft_mixed, ...)
  D  {"system_prompt":"...", "question":"...", "response":"..."}  (coder chunks)
  E  {"input": "...", "output": "..."}                 (combined_test/val -- already Human:/Asst:)
  F  {"prompt":"...", "instruction":"...", "output":"..."} (prompt_code_layer -- already prefixed)
  G  {"prompt":"...", "chosen":"..."}                  (rlhf_pairs -- use chosen)

Usage:
  python tools/prep_chat_corpus.py <in1.jsonl> [in2 ...] -o <out.jsonl>
  python tools/prep_chat_corpus.py --stats <in1.jsonl> ...     # quality report only, no output
"""

import sys, json, re, argparse, unicodedata
from pathlib import Path
from collections import Counter

# ─── Schema detection ──────────────────────────────────────────────────────────

def detect_schema(rec: dict) -> str:
    k = set(rec.keys())
    if "text" in k:                                 return "A"
    if "messages" in k:                             return "B"
    if "system_prompt" in k and "question" in k:    return "D"
    if "input" in k and "output" in k:              return "E"
    if "instruction" in k and "output" in k:        return "F"
    if "prompt" in k and "chosen" in k:             return "G"
    if "prompt" in k and "response" in k:           return "C"
    if "query" in k and "response" in k:            return "H"
    if "input_text" in k and "output_text" in k:    return "I"
    return "?"


# ─── Formatting ────────────────────────────────────────────────────────────────

def fmt_messages(msgs: list) -> str:
    parts = []
    for m in msgs:
        role    = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    return "\n\n".join(parts)


def to_text(rec: dict, schema: str) -> "str | None":
    if schema == "A":
        return (rec.get("text") or "").strip() or None

    if schema == "B":
        t = fmt_messages(rec.get("messages") or [])
        return t if t else None

    if schema == "C":
        p = (rec.get("prompt")   or "").strip()
        r = (rec.get("response") or "").strip()
        if not p or not r:
            return None
        return f"User: {p}\n\nAssistant: {r}"

    if schema == "D":
        sys_ = (rec.get("system_prompt") or "").strip()
        q    = (rec.get("question")      or "").strip()
        r    = (rec.get("response")      or "").strip()
        if not q or not r:
            return None
        if sys_:
            return f"System: {sys_}\n\nUser: {q}\n\nAssistant: {r}"
        return f"User: {q}\n\nAssistant: {r}"

    if schema == "E":
        inp = (rec.get("input")  or "").strip()
        out = (rec.get("output") or "").strip()
        if not inp or not out:
            return None
        return f"{inp}\n\n{out}"

    if schema == "F":
        # instruction == prompt, both already "Human: ..." prefixed; output is "Assistant: ..."
        instr = (rec.get("instruction") or rec.get("prompt") or "").strip()
        out   = (rec.get("output")      or "").strip()
        if not instr or not out:
            return None
        return f"{instr}\n\n{out}"

    if schema == "G":
        p = (rec.get("prompt") or "").strip()
        c = (rec.get("chosen") or "").strip()
        if not p or not c:
            return None
        return f"User: {p}\n\nAssistant: {c}"

    if schema == "H":
        # khanary_clean_train style: response is already conversation-formatted ("Human:..." / "A:...")
        r = (rec.get("response") or "").strip()
        return r if r else None

    if schema == "I":
        # contextual_layer style with input_text/output_text
        inp = (rec.get("input_text")  or "").strip()
        out = (rec.get("output_text") or "").strip()
        if not inp or not out:
            return None
        return f"User: {inp}\n\nAssistant: {out}"

    return None


# ─── Quality filters ───────────────────────────────────────────────────────────

# Long base64 blobs (data URIs, embedded images, serialized weights)
_RE_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")
# data: URI (images, audio, video)
_RE_DATA_URI    = re.compile(r"data:[a-z]+/[a-z0-9.+-]+;base64,", re.I)
# Excessive URL density (line that is nearly all URL)
_RE_URL_LINE    = re.compile(r"https?://\S{80,}")
# Repeated punctuation bursts (░▒▓ or ======= artifacts)
_RE_FILL_CHARS  = re.compile(r"[░▒▓=─━┄]{20,}")

# Non-KUHUL opcodes: foreign model template tokens + tool-call formats that don't
# align with the K'UHUL glyph/opcode system. Training on these would teach the
# model a different tool-use grammar and corrupt the KUHUL opcode vocabulary.
_RE_NON_KUHUL = re.compile(
    r"""
    \[tool_call\]           |   # generic tool call marker
    \[tool_result\]         |   # generic tool result
    \[function_call\]       |   # openai function call
    \[function_result\]     |
    <\|im_start\|>          |   # ChatML / qwen template
    <\|im_end\|>            |
    <\|system\|>            |   # phi-style system token
    <\|user\|>              |
    <\|assistant\|>         |
    \[INST\]                |   # Llama-2 instruction template
    \[/INST\]               |
    <<SYS>>                 |   # Llama-2 system block
    <</SYS>>                |
    <s>                     |   # sentencepiece BOS (raw in text)
    </s>                    |   # sentencepiece EOS (raw in text)
    <\|endoftext\|>         |   # GPT-2/GPT-3 EOT embedded mid-text
    <\|pad\|>               |
    <\|sep\|>               |
    <think>                 |   # scratchpad/CoT tokens (DeepSeek-R1 style)
    </think>                |
    <reasoning>             |
    </reasoning>            |
    <tool_call>             |   # XML-style tool call (Qwen2.5 / Mistral)
    </tool_call>            |
    <tool_response>         |
    </tool_response>        |
    <function_calls>        |
    </function_calls>       |
    <invoke>                |   # Claude XML tool format
    </invoke>               |
    <tool_use>              |
    </tool_use>
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _ngram_repetition_ratio(text: str, n: int = 5, threshold: int = 4) -> float:
    """Returns fraction of words that belong to a repeated n-gram (appears >= threshold times)."""
    words = text.split()
    if len(words) < n * threshold:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated_starts = {i for i, ng in enumerate(ngrams) if counts[ng] >= threshold}
    words_in_repeats = set()
    for start in repeated_starts:
        words_in_repeats.update(range(start, start + n))
    return len(words_in_repeats) / len(words)


def quality_ok(text: str, min_len: int = 50) -> tuple[bool, str]:
    """Returns (ok, reason_if_rejected)."""

    if len(text) < min_len:
        return False, "too_short"

    # Binary / replacement character ratio
    total = len(text)
    bad_chars = sum(
        1 for c in text
        if c == "�"
        or (unicodedata.category(c) == "Cc" and c not in "\n\r\t")
    )
    if bad_chars / total > 0.02:
        return False, f"bad_encoding ({bad_chars}/{total})"

    # Data URI (embedded images/audio)
    if _RE_DATA_URI.search(text):
        return False, "data_uri"

    # Base64 blob
    if _RE_BASE64_BLOB.search(text):
        return False, "base64_blob"

    # Fill / artifact characters
    if _RE_FILL_CHARS.search(text):
        return False, "fill_chars"

    # Non-KUHUL opcodes (foreign model template tokens contaminate opcode vocabulary)
    m = _RE_NON_KUHUL.search(text)
    if m:
        return False, f"non_kuhul_opcode ({m.group().strip()})"

    # N-gram repetition (catches looped/hallucinated output from source models)
    rep = _ngram_repetition_ratio(text, n=5, threshold=4)
    if rep > 0.35:
        return False, f"ngram_repeat ({rep:.2f})"

    # Very high non-ASCII fraction (e.g. pure CJK dump with no Latin at all)
    # — We keep CJK if it's < 80% of the text (mixed is fine, monolingual is fine;
    #   the concern is garbled multi-byte scrapes)
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / total > 0.80:
        return False, f"non_ascii_dominant ({non_ascii}/{total})"

    return True, ""


# ─── File processor ────────────────────────────────────────────────────────────

def process_file(path: str, out, min_len: int, stats_only: bool) -> dict:
    schema = None
    counts = Counter()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                counts["parse_err"] += 1
                continue
            if schema is None:
                schema = detect_schema(rec)
            if schema == "?":
                counts["unknown_schema"] += 1
                continue
            counts["read"] += 1
            text = to_text(rec, schema)
            if text is None:
                counts["empty"] += 1
                continue
            ok, reason = quality_ok(text, min_len)
            if not ok:
                counts[f"reject:{reason}"] += 1
                continue
            counts["written"] += 1
            if not stats_only:
                out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return {"schema": schema or "?", **dict(counts)}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Input JSONL files")
    ap.add_argument("-o", "--output", default=None, help="Output JSONL (omit for --stats mode)")
    ap.add_argument("--stats", action="store_true", help="Quality report only, no output written")
    ap.add_argument("--min-len", type=int, default=50,
                    help="Min text character length to keep (default 50)")
    a = ap.parse_args()

    if not a.stats and not a.output:
        ap.error("Provide -o OUTPUT or --stats")

    stats_only = a.stats

    total_read = total_written = 0
    all_rejects: Counter = Counter()

    ctx = open(a.output, "w", encoding="utf-8") if not stats_only else open_devnull()
    with ctx as out:
        for path in a.inputs:
            r = process_file(path, out, a.min_len, stats_only)
            schema  = r.pop("schema")
            read    = r.get("read", 0)
            written = r.get("written", 0)
            total_read    += read
            total_written += written
            name = Path(path).name
            pct  = 100 * written / max(read, 1)
            reject_parts = [f"{k.split(':',1)[-1]}={v}" for k, v in r.items()
                            if k.startswith("reject:") and v > 0]
            reject_str = ", ".join(reject_parts) if reject_parts else "none"
            print(f"  [{schema}] {name}: {read:,} read -> {written:,} kept ({pct:.0f}%) | rejected: {reject_str}")
            for k, v in r.items():
                if k.startswith("reject:"):
                    all_rejects[k] += v

    print(f"\nTotal: {total_read:,} read -> {total_written:,} written")
    if all_rejects:
        print("Rejection breakdown:")
        for k, v in all_rejects.most_common():
            print(f"  {k[7:]}: {v:,}")
    if not stats_only:
        out_mb = Path(a.output).stat().st_size / 1e6
        print(f"Output: {a.output}  ({out_mb:.1f} MB)")


class open_devnull:
    """Context manager that provides a no-op write() for --stats mode."""
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def write(self, _): pass


if __name__ == "__main__":
    main()
