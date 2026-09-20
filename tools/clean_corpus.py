#!/usr/bin/env python3
# clean_corpus.py -- normalize mixed instruction/chat JSONL into ONE clean training file,
# stripping images / base64 / binary / control chars (per the "no images or bad chars" rule).
#
# Handles four shapes seen in E:\data:
#   flat query/response      (help_merged, instruct_merged)
#   flat instruction/output  (qa-corpus)
#   ChatGPT export 'mapping'  (0031, 0013)  -> linearize user/assistant turns
#   {content: "<json string of a LIST of export objects>"} (0032) -> parse + recurse
# Non-string content parts (image_asset_pointer etc.) are dropped; data:URIs and long
# base64 runs are scrubbed from text; a pair is kept only if BOTH sides survive clean.
#
# Usage: python tools/clean_corpus.py <out.jsonl> <in1.jsonl> [in2.jsonl ...]

import sys, os, json, re, unicodedata

DATA_URI   = re.compile(r'data:[a-zA-Z0-9.+-]+/[a-zA-Z0-9.+-]+;[^\s"\')]{16,}')  # data:image/...;base64,....
B64_RUN    = re.compile(r'(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{200,}={0,2}(?![A-Za-z0-9+/])')
IMG_MD     = re.compile(r'!\[[^\]]*\]\((?:data:|[^)]*\.(?:png|jpe?g|gif|webp|svg)[^)]*)\)', re.I)
CTRL       = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')  # keep \t \n \r
# --- opcode / bytecode / machine-code contamination (per "no opcodes") ---
HEX_RUN    = re.compile(r'(?:(?:0[xX])?[0-9A-Fa-f]{2}[\s,]+){7,}(?:0[xX])?[0-9A-Fa-f]{2}')  # hex byte dumps
ESC_RUN    = re.compile(r'(?:\\x[0-9A-Fa-f]{2}){5,}')                                       # \xNN escape runs
GCODE_LN   = re.compile(r'(?mi)^\s*[NGM]\d{1,4}(?:\s+[A-Z][-+]?\d*\.?\d+)+\s*$')            # CNC G-code lines
ASM_LN     = re.compile(r'(?mi)^\s*(?:0[xX])?[0-9A-Fa-f]{4,}:?\s+(?:[0-9A-Fa-f]{2}\s+){2,}') # disasm addr+bytes
GLYPH_OP   = re.compile(r'(?:0x0[0-9A-Fa-f]\b[\s,]*){4,}')                                  # K'UHUL glyph opcodes 0x00-0x08

def scrub(t):
    if not isinstance(t, str): return ""
    t = unicodedata.normalize("NFC", t)
    t = IMG_MD.sub("", t)
    t = DATA_URI.sub("", t)
    t = B64_RUN.sub("", t)
    t = ASM_LN.sub("", t)
    t = GCODE_LN.sub("", t)
    t = HEX_RUN.sub("", t)
    t = ESC_RUN.sub("", t)
    t = GLYPH_OP.sub("", t)
    t = CTRL.sub("", t)
    return t.strip()

def opcode_heavy(t):
    """True if raw text is dominated by opcode/hex/bytecode/G-code -- drop, don't frankenscrub."""
    if not t: return False
    matched = sum(len(m.group()) for rx in (HEX_RUN, ESC_RUN, GCODE_LN, ASM_LN, GLYPH_OP, B64_RUN, DATA_URI)
                  for m in rx.finditer(t))
    return matched > 0.25 * len(t)

def ok_pair(q, r):
    if opcode_heavy(q) or opcode_heavy(r): return None   # drop opcode/bytecode-dominated records
    q, r = scrub(q), scrub(r)
    if len(q) < 2 or len(r) < 2: return None
    if len(q) > 24000 or len(r) > 24000:  # runaway record -> skip, don't truncate mid-token
        return None
    return {"query": q, "response": r}

def parts_text(msg):
    """Extract only STRING parts of a ChatGPT-export message; drop image/asset parts."""
    try:
        c = msg["content"]
        if c.get("content_type") not in (None, "text", "multimodal_text"): return None
        out = [p for p in c.get("parts", []) if isinstance(p, str)]
        return "\n".join(out).strip() or None
    except Exception:
        return None

def from_mapping(mapping):
    """Linearize a ChatGPT 'mapping' dict into ordered (role, text) turns, then pair them."""
    turns = []
    for node in mapping.values():
        m = node.get("message")
        if not m: continue
        role = (m.get("author") or {}).get("role")
        if role not in ("user", "assistant"): continue
        txt = parts_text(m)
        if txt: turns.append((m.get("create_time") or 0, role, txt))
    turns.sort(key=lambda x: x[0])
    pairs, pend = [], None
    for _, role, txt in turns:
        if role == "user": pend = txt
        elif role == "assistant" and pend is not None:
            p = ok_pair(pend, txt)
            if p: pairs.append(p)
            pend = None
    return pairs

def records_from(obj):
    """Yield {query,response} dicts from one parsed JSONL object of any known shape."""
    if not isinstance(obj, dict): return
    if "query" in obj and "response" in obj:
        p = ok_pair(obj["query"], obj["response"]);  yield from ([p] if p else []); return
    if "instruction" in obj and "output" in obj:
        p = ok_pair(obj["instruction"], obj["output"]); yield from ([p] if p else []); return
    if "mapping" in obj and isinstance(obj["mapping"], dict):
        yield from from_mapping(obj["mapping"]); return
    if "content" in obj and isinstance(obj["content"], str):  # 0032: content is JSON-string list
        try: inner = json.loads(obj["content"])
        except Exception: return
        for it in (inner if isinstance(inner, list) else [inner]):
            if isinstance(it, dict) and "mapping" in it:
                yield from from_mapping(it["mapping"])
        return

def main():
    out_path, ins = sys.argv[1], sys.argv[2:]
    kept = imgs_scrubbed = 0
    seen = set()
    with open(out_path, "w", encoding="utf-8") as out:
        for fp in ins:
            name = os.path.basename(fp)
            fin = fout = fimg = 0
            with open(fp, "rb") as f:
                for raw in f:
                    fin += 1
                    try: obj = json.loads(raw.decode("utf-8"))
                    except Exception: continue
                    if DATA_URI.search(raw.decode("utf-8","ignore")) or B64_RUN.search(raw.decode("utf-8","ignore")):
                        fimg += 1
                    for rec in records_from(obj):
                        key = hash(rec["query"][:200] + "\x00" + rec["response"][:200])
                        if key in seen: continue     # dedup
                        seen.add(key)
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fout += 1
            kept += fout; imgs_scrubbed += fimg
            print(f"  {name}: in={fin} -> pairs={fout}  (records with media scrubbed: {fimg})")
    print(f"[ok] {out_path}: {kept} clean pairs, {os.path.getsize(out_path)/1e6:.1f} MB")

if __name__ == "__main__":
    main()
