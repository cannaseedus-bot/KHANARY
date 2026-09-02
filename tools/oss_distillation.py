"""oss_distillation.py — GPT-OSS teacher -> from_zero LoRA distillation

Uses the configured local HTTP API as the GPT-OSS teacher, or Ollama cloud
teacher via --ollama-url / --ollama-model.  Trains low-rank adapter weights (LoRA)
on a from_zero student to match teacher outputs.
No PEFT / HuggingFace transformers required — pure PyTorch + safetensors.

Strategy: response distillation
  1. Send prompts to GPT-OSS teacher -> get completion text
  2. Tokenize the (prompt + completion) as a sequence
  3. Run from_zero student forward pass
  4. Cross-entropy loss on the completion tokens (prompt tokens masked)
  5. Backprop through LoRA adapter params only
  6. Save LoRA.safetensors

LoRA: adds W_delta = A @ B to each frozen projection matrix.
  - A: [r, in_dim]  initialized N(0, 0.02/r)
  - B: [out_dim, r] initialized zeros
  - W_eff = W + B @ A  (scale omitted; alpha=r gives unit scaling)
  - Only A, B are in optimizer; W stays frozen

Usage:
    python tools/oss_distillation.py \
        --student    models/from_zero/from_zero_v0.1_folded.safetensors \
        --out        models/from_zero/from_zero_v0.1_lora.safetensors \
        --rank       8 \
        --steps      200 \
        --lr         1e-4 \
        --engine     http://127.0.0.1:17480 \
        --atomic-dom models/from_zero/atomic.manifest.json \
        --ollama-url  http://127.0.0.1:11434 \
        --ollama-model gpt-oss:120b-cloud

If --engine is unreachable and no Ollama teacher is configured, falls back to
self-distillation (adapter shape validation only — not meaningful training).
"""

import datetime
import argparse
import json
import math
import os
import urllib.request
import urllib.error
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

# In-memory cache for Ollama teacher completions (keyed by url+model+prompt+tokens)
_OLLAMA_CACHE: dict = {}

# ---------------------------------------------------------------------------
# GPT-2 config (small / medium — auto-detected from wte shape)
# ---------------------------------------------------------------------------
GPT2_SMALL  = dict(n_layer=6,  n_head=6,  n_embd=768,  vocab_size=50270, block_size=1024)
GPT2_MEDIUM = dict(n_layer=12, n_head=12, n_embd=768,  vocab_size=50270, block_size=1024)

# ---------------------------------------------------------------------------
# Weight shape map — flat storage → proper 2D shapes
# ---------------------------------------------------------------------------
def _gpt2_shapes(cfg):
    E, H, L = cfg['n_embd'], cfg['n_head'], cfg['n_layer']
    V, S    = cfg['vocab_size'], cfg['block_size']
    head    = E // H
    shapes  = {}
    shapes['transformer.wte.weight']  = [V, E]
    shapes['transformer.wpe.weight']  = [S, E]
    shapes['transformer.ln_f.weight'] = [E]
    shapes['transformer.ln_f.bias']   = [E]
    for i in range(L):
        p = f'transformer.h.{i}'
        shapes[f'{p}.ln_1.weight']           = [E]
        shapes[f'{p}.ln_1.bias']             = [E]
        shapes[f'{p}.ln_2.weight']           = [E]
        shapes[f'{p}.ln_2.bias']             = [E]
        shapes[f'{p}.attn.c_attn.weight']    = [E, 3*E]
        shapes[f'{p}.attn.c_attn.bias']      = [3*E]
        shapes[f'{p}.attn.c_proj.weight']    = [E, E]
        shapes[f'{p}.attn.c_proj.bias']      = [E]
        shapes[f'{p}.mlp.c_fc.weight']       = [E, 4*E]
        shapes[f'{p}.mlp.c_fc.bias']         = [4*E]
        shapes[f'{p}.mlp.c_proj.weight']     = [4*E, E]
        shapes[f'{p}.mlp.c_proj.bias']       = [E]
    return shapes

# ---------------------------------------------------------------------------
# Load safetensors into reshaped tensors
# ---------------------------------------------------------------------------
def load_model(path: str, cfg: dict) -> dict:
    shapes = _gpt2_shapes(cfg)
    state  = {}
    with safe_open(path, framework='pt') as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k in shapes:
                expected_numel = 1
                for d in shapes[k]:
                    expected_numel *= d
                if t.numel() == expected_numel and list(t.shape) != shapes[k]:
                    t = t.reshape(shapes[k])
            state[k] = t
    print(f'[student] loaded {len(state)} tensors from {path}')
    return state

# ---------------------------------------------------------------------------
# Minimal GPT-2 forward pass (returns logits, no KV-cache)
# ---------------------------------------------------------------------------
def gpt2_forward(state: dict, lora_deltas: dict, input_ids: torch.Tensor, cfg: dict):
    B, T     = input_ids.shape
    E        = cfg['n_embd']
    n_head   = cfg['n_head']
    n_layer  = cfg['n_layer']
    head_dim = E // n_head

    def get(key):
        base = state[key].float()
        if key in lora_deltas:
            A, B_ = lora_deltas[key]
            base = base + (B_.float() @ A.float())
        return base

    # Embeddings
    wte = get('transformer.wte.weight')
    wpe = get('transformer.wpe.weight')
    pos = torch.arange(T, device=input_ids.device)
    x = wte[input_ids] + wpe[pos].unsqueeze(0)  # [B, T, E]

    for i in range(n_layer):
        p = f'transformer.h.{i}'

        # LayerNorm 1
        ln1_w = get(f'{p}.ln_1.weight')
        ln1_b = get(f'{p}.ln_1.bias')
        x_ln  = F.layer_norm(x, [E], ln1_w, ln1_b)

        # Self-attention
        W_qkv = get(f'{p}.attn.c_attn.weight')   # [E, 3E]
        b_qkv = get(f'{p}.attn.c_attn.bias')     # [3E]
        qkv   = x_ln @ W_qkv + b_qkv             # [B, T, 3E]
        q, k, v = qkv.chunk(3, dim=-1)            # [B, T, E] each

        # Reshape to heads
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(head_dim)
        att   = (q @ k.transpose(-2, -1)) * scale
        # Causal mask
        mask  = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att   = att.masked_fill(mask == 0, float('-inf'))
        att   = F.softmax(att, dim=-1)

        attn_out = att @ v                                          # [B, nh, T, hd]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, E)

        W_proj = get(f'{p}.attn.c_proj.weight')
        b_proj = get(f'{p}.attn.c_proj.bias')
        attn_out = attn_out @ W_proj + b_proj
        x = x + attn_out

        # LayerNorm 2
        ln2_w = get(f'{p}.ln_2.weight')
        ln2_b = get(f'{p}.ln_2.bias')
        x_ln2 = F.layer_norm(x, [E], ln2_w, ln2_b)

        # MLP
        W_fc  = get(f'{p}.mlp.c_fc.weight')
        b_fc  = get(f'{p}.mlp.c_fc.bias')
        W_cp  = get(f'{p}.mlp.c_proj.weight')
        b_cp  = get(f'{p}.mlp.c_proj.bias')
        h_mlp = x_ln2 @ W_fc + b_fc
        h_mlp = h_mlp * torch.sigmoid(1.702 * h_mlp)   # GELU approx
        h_mlp = h_mlp @ W_cp + b_cp
        x     = x + h_mlp

    # Final LN + LM head
    lnf_w  = get('transformer.ln_f.weight')
    lnf_b  = get('transformer.ln_f.bias')
    x      = F.layer_norm(x, [E], lnf_w, lnf_b)
    lm_w   = state.get('transformer.lm_head.weight', state['transformer.wte.weight'])
    logits = x @ lm_w.float().T                  # [B, T, V]
    return logits

# ---------------------------------------------------------------------------
# LoRA adapter factory
# ---------------------------------------------------------------------------
def make_lora_adapters(state: dict, rank: int, lora_keys: list) -> dict:
    adapters = {}
    for key in lora_keys:
        if key not in state:
            continue
        W = state[key]
        if W.dim() < 2:
            continue
        out_dim, in_dim = W.shape[0], W.shape[1]
        A = nn.Parameter(torch.randn(rank, in_dim) * (0.02 / rank))
        B = nn.Parameter(torch.zeros(out_dim, rank))
        adapters[key] = (A, B)
    return adapters

def resume_adapters(adapters: dict, resume_path: str) -> dict:
    """Overwrite A/B in adapters with weights from an existing LoRA safetensors file."""
    if not os.path.isfile(resume_path):
        return adapters
    print(f'[resume] loading LoRA weights from {resume_path}')
    with safe_open(resume_path, framework='pt') as f:
        for full_key in f.keys():
            # keys look like lora_A__transformer__h__0__attn__c_attn__weight
            parts = full_key.split('__', 2)
            if len(parts) < 3 or parts[0] not in ('lora_A', 'lora_B'):
                continue
            which = parts[0]          # lora_A or lora_B
            base_key = parts[1] + '__' + parts[2] if len(parts) == 3 else parts[1]
            base_key = base_key.replace('__', '.')
            if base_key not in adapters:
                continue
            A, B = adapters[base_key]
            t = f.get_tensor(full_key)
            if which == 'lora_A' and t.shape == A.shape:
                A.data.copy_(t)
            elif which == 'lora_B' and t.shape == B.shape:
                B.data.copy_(t)
    # Re-check shapes
    ok = 0
    for key, (A, B_) in adapters.items():
        if B_.abs().sum().item() == 0:
            pass
        else:
            ok += 1
    print(f'[resume] adapters updated ({len(adapters)} total)')
    return adapters

# ---------------------------------------------------------------------------
# GPT-2 tokenizer (byte-pair encoding via tiktoken if available, else simple)
# ---------------------------------------------------------------------------
def get_tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
        return lambda text: enc.encode(text)
    except ImportError:
        pass
    # Fallback: very naive character-level (won't give GPT-2 tokens but allows testing)
    def char_encode(text):
        return [ord(c) % 50270 for c in text]
    print('[tokenizer] tiktoken not found, using char fallback — install with: pip install tiktoken')
    return char_encode

# ---------------------------------------------------------------------------
# Teacher: call kuhul_engine (GPT-OSS) for completion
# ---------------------------------------------------------------------------
def teacher_complete(engine_base: str, prompt: str, max_tokens: int = 256,
                     ollama_url: str = None, ollama_model: str = None, tools=None, tool_choice=None) -> str:
    if ollama_url and ollama_model:
        return teacher_complete_ollama(ollama_url, ollama_model, prompt, max_tokens)
    # Fallback to original kuhul_engine path
    payload = {
        'model': 'gpt-oss-20b-MXFP4',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': 0.7,
        'stream': False
    }
    if tools:
        payload['tools'] = tools
        payload['tool_choice'] = tool_choice or 'auto'
    req = urllib.request.Request(
        f'{engine_base}/v1/chat/completions',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            msg = data['choices'][0]['message']
            if msg.get('content'):
                return msg['content']
            if msg.get('tool_calls'):
                # render the tool call as <tool_call>{json}</tool_call> for the student
                f = msg['tool_calls'][0].get('function', {})
                return f'<tool_call>{json.dumps({"name": f.get("name"), "arguments": f.get("arguments")})}</tool_call>'
            return None
    except Exception as e:
        return None

def teacher_complete_xshard(xshard_url: str, prompt: str, max_tokens: int = 256) -> str:
    """Call nnc_k_server (local xshard DDS streaming) as a teacher.

    Endpoint: {xshard_url}/v1/chat/completions  (OpenAI-compatible)
    No model name hardcoded — the server knows its own model.
    """
    cache_key = ('xshard', xshard_url, prompt, max_tokens)
    if cache_key in _OLLAMA_CACHE:
        return _OLLAMA_CACHE[cache_key]

    payload = json.dumps({
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': 0.7,
        'stream': False,
    }).encode()
    req = urllib.request.Request(
        f'{xshard_url}/v1/chat/completions',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
                text = data['choices'][0]['message']['content']
                if text and text.strip():
                    _OLLAMA_CACHE[cache_key] = text
                return text
        except Exception as e:
            last_err = e
    print(f'[xshard] teacher_complete_xshard failed after 3 attempts: {last_err}')
    return None


def micronaut_xquery(kuhul_url: str, prompt: str) -> dict:
    """POST /micronauts/select → {name, system_context, atomic_blocks}.

    Returns defaults on any error so the caller can always proceed.
    """
    payload = json.dumps({'prompt': prompt}).encode()
    req = urllib.request.Request(
        f'{kuhul_url}/micronauts/select',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return {
                'name': data.get('selected', {}).get('name', 'khanary'),
                'system_context': data.get('system_context'),
                'atomic_blocks': data.get('atomic_blocks', ['BODY']),
            }
    except Exception:
        return {'name': 'khanary', 'system_context': None, 'atomic_blocks': ['BODY']}


def write_micronaut_qa(micronauts_dir: str, micronaut_name: str, prompt: str, completion: str, step: int):
    """Append Q&A pair to micronauts/semantic/{name}_distil.json for factory pickup."""
    sem_dir = os.path.join(micronauts_dir, 'semantic')
    os.makedirs(sem_dir, exist_ok=True)
    qa_file = os.path.join(sem_dir, f'{micronaut_name}_distil.json')
    entry = {
        'prompt': prompt,
        'completion': completion,
        'step': step,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    if os.path.isfile(qa_file):
        try:
            with open(qa_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            data = {'name': micronaut_name, 'qa_pairs': []}
    else:
        data = {'name': micronaut_name, 'qa_pairs': []}
    data['qa_pairs'].append(entry)
    with open(qa_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def teacher_complete_ollama(ollama_url: str, ollama_model: str, prompt: str, max_tokens: int = 256,
                            system_context: str = None) -> str:
    cache_key = (ollama_url, ollama_model, prompt, max_tokens, system_context)
    if cache_key in _OLLAMA_CACHE:
        return _OLLAMA_CACHE[cache_key]

    system_msg = system_context if system_context else 'Answer briefly and directly.'
    payload = json.dumps({
        'model': ollama_model,
        'messages': [
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': prompt}
        ],
        'stream': False,
        'options': {
            'num_predict': max_tokens,
            'temperature': 0.7,
            'reasoning': False,
        }
    }).encode()

    last_err = None
    for attempt in range(3):
        req = urllib.request.Request(
            f'{ollama_url}/api/chat',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
                content = data.get('message', {}).get('content', '') or data.get('response', '')
                _OLLAMA_CACHE[cache_key] = content
                return content
        except Exception as e:
            last_err = e
            print(f'[teacher] ollama error (attempt {attempt + 1}/3): {e}')
            if attempt < 2:
                import time
                time.sleep(2 ** attempt)  # 1s, 2s
    return None

# ---------------------------------------------------------------------------
# Atomic DOM block tagging for each prompt (mirrors kuhul-server /micronauts/select)
# ---------------------------------------------------------------------------
import re as _re_distill

MICRONAUT_KEYWORDS = {
    'coder':            ['code', 'program', 'bug', 'refactor', 'compile', 'function', 'error', 'fix', 'implement', 'script', 'cpp', 'c++', 'python', 'javascript', 'js'],
    'memory':           ['memory', 'remember', 'recall', 'forget', 'save this', 'stored', 'retain'],
    'ui':               ['ui', 'html', 'css', 'interface', 'frontend', 'canvas', 'display', 'render', 'webview', 'react'],
    'stack_doc':        ['stack', 'kuhul', 'khanary', 'system', 'architecture', 'how does', 'what is', 'native', 'driver', 'dll', 'micronauts', 'registry'],
    'primeos_guide':    ['primeos', 'desktop', 'wpf', 'app shell', 'window', 'model selector', 'webview2', 'winui'],
    'scx_guide':        ['scx', 'scxq2', 'bytecode', 'compile', 'decompile', 'runtime', 'opcode', '@op', 'instruction', 'xcfe'],
    'asx_guide':        ['asx', 'gravity', 'entropy', 'pressure', 'physics', 'routing', 'attention', 'state', 'shared memory'],
    'distillation_guide': ['distill', 'lora', 'teacher', 'train', 'fine-tune', 'from_zero', 'safetensors', 'oss_distillation'],
    'programs':         ['@op', '@control', '@state', '@program', 'programs_dispatcher', 'load_program', 'resolve_stdlib', 'json program', 'control graph', 'native.read', 'native.write', 'native.eval', 'native.hash', 'native.call', 'native.phase', 'stdlib', 'op dispatch', 'fold_enter', 'fold_exit', 'gpu_dispatch', 'verb', 'primitive contract'],
    'tool_call':        ['tool', 'function call', 'call', 'invoke', 'mcp', 'json-rpc', 'kuhul_task_boss'],
    'chat':             ['chat', 'talk', 'hello', 'hi', 'hey', 'conversation'],
}

ATOMIC_BLOCKS_FOR_MICRONAUT = {
    'coder': ['BODY'],
    'memory': ['MENU', 'BODY'],
    'ui': ['FOOTER', 'BODY'],
    'stack_doc': ['MENU', 'BODY'],
    'primeos_guide': ['MENU', 'BODY'],
    'scx_guide': ['MENU', 'BODY'],
    'asx_guide': ['MENU', 'BODY'],
    'distillation_guide': ['MENU', 'BODY'],
    'programs': ['BODY'],
    'tool_call': ['BODY'],
    'chat': ['BODY'],
    'khanary': ['HEADER', 'BODY'],
    'pop': ['HEADER'],
    'wo': ['MENU'],
    'yax': ['MENU', 'BODY'],
    'sek': ['BODY'],
    'chen': ['BODY'],
    'xul': ['FOOTER'],
}

def select_micronaut_for_prompt(prompt: str) -> dict:
    text = (prompt or '').lower()
    best = 'khanary'
    best_score = 0.0
    for name, kws in MICRONAUT_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in text)
        if score > 0 and score > best_score:
            best_score = score
            best = name
    fold_match = _re_distill.search(r'\b(pop|wo|yax|sek|chen|xul)\b', text)
    if fold_match:
        best = fold_match.group(1)
    return {
        'micronaut': best,
        'atomic_blocks': ATOMIC_BLOCKS_FOR_MICRONAUT.get(best, ['BODY'])
    }

# ---------------------------------------------------------------------------
# Default prompts (kuhul domain)
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    "Explain the K'UHUL fold system and how it organizes semantic execution phases.",
    "Describe the role of the BOSS orchestrator in the WebX compute engine.",
    "How does the micronaut factory create sampling-profile micronauts?",
    "What is the difference between a fold micronaut and a chat micronaut?",
    "Explain the π-nary arc weighting in the KuhulPhysics gradient controller.",
    "How does the DirectML GEMM bridge accelerate matrix operations on Intel HD 4600?",
    "What is SCXQ2 IR and how does it represent the K'UHUL execution graph?",
    "Describe the KUHUL APPS app generation studio and its kuhul_engine backend.",
    "How do semantic micronauts influence llama-server sampling parameters?",
    "Explain the SLERP merge strategy for combining two training phase checkpoints.",
    "What are atomic blocks in the K'UHUL semantic field system?",
    "How does the grammar validator use kuhul.ebnf for constrained generation?",
    "Describe the LoRA distillation pipeline from GPT-OSS to from_zero.",
    "What is the role of the MicrosoftSDK.ps1 in the NNC-K stack?",
    "How does the Phase Pop-Wo-Sek-Chen-Xul loop work in K'UHUL execution?",
]

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    # Auto-detect config from actual weight keys + wte shape
    with safe_open(args.student, framework='pt') as f:
        keys = list(f.keys())
        wte  = f.get_tensor('transformer.wte.weight')
    n_embd  = wte.shape[1] if wte.dim() == 2 else 768
    vocab_size = wte.shape[0] if wte.dim() == 2 else 50270
    layer_idxs = set()
    for k in keys:
        parts = k.split('.')
        if len(parts) > 2 and parts[0] == 'transformer' and parts[1] == 'h' and parts[2].isdigit():
            layer_idxs.add(int(parts[2]))
    n_layer = max(layer_idxs) + 1 if layer_idxs else 12
    n_head  = n_embd // 64
    cfg = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, vocab_size=vocab_size, block_size=1024)
    print(f'[config] n_layer={n_layer} n_embd={n_embd} n_head={n_head} vocab={vocab_size}')

    # Load student
    state = load_model(args.student, cfg)

    # LoRA targets: c_attn and c_proj projection weights for every layer
    lora_keys = []
    for i in range(cfg['n_layer']):
        p = f'transformer.h.{i}'
        lora_keys += [
            f'{p}.attn.c_attn.weight',
            f'{p}.attn.c_proj.weight',
            f'{p}.mlp.c_fc.weight',
            f'{p}.mlp.c_proj.weight',
        ]

    adapters = make_lora_adapters(state, args.rank, lora_keys)
    if getattr(args, 'resume', False):
        adapters = resume_adapters(adapters, args.out)
    params   = [p for ab in adapters.values() for p in ab]
    optim    = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    tokenize = get_tokenizer()

    # Load pre-existing prompt+completion pairs from JSONL/CSV (no live teacher needed).
    # Supports: {"prompt":…,"completion":…}, {"messages":[…]}, {"input":…,"output":…},
    #           {"instruction":…,"output":…}, CSV with question/answer columns.
    paired_data: list[tuple[str, str]] = []
    if getattr(args, 'jsonl_data', None):
        import csv as _csv
        for src_path in [p.strip() for p in args.jsonl_data.split(',') if p.strip()]:
            if not os.path.isfile(src_path):
                print(f'[data] WARN: not found {src_path}')
                continue
            ext = os.path.splitext(src_path)[1].lower()
            loaded = 0
            if ext == '.csv':
                with open(src_path, encoding='utf-8-sig', errors='replace') as fh:
                    reader = _csv.DictReader(fh)
                    for row in reader:
                        q = row.get('question') or row.get('prompt') or row.get('input') or ''
                        a = row.get('answer')   or row.get('completion') or row.get('output') or ''
                        if q and a:
                            paired_data.append((q.strip(), a.strip()))
                            loaded += 1
            else:  # .jsonl or .json
                with open(src_path, encoding='utf-8-sig', errors='replace') as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        # Normalise various formats
                        if 'messages' in obj:
                            msgs = obj['messages']
                            user_msg  = next((m['content'] for m in msgs if m.get('role') == 'user'), '')
                            asst_msg  = next((m['content'] for m in msgs if m.get('role') == 'assistant'), '')
                            if user_msg and asst_msg:
                                paired_data.append((user_msg.strip(), asst_msg.strip()))
                                loaded += 1
                        else:
                            q = obj.get('prompt') or obj.get('input') or obj.get('instruction') or obj.get('question') or ''
                            a = obj.get('completion') or obj.get('output') or obj.get('response') or ''
                            if q and a:
                                paired_data.append((q.strip(), a.strip()))
                                loaded += 1
            print(f'[data] {src_path}: {loaded} pairs loaded')
        print(f'[data] total paired samples: {len(paired_data)}')

    # Load prompts
    prompts = DEFAULT_PROMPTS[:]
    prompt_tools = {}
    if args.prompts and os.path.isfile(args.prompts):
        with open(args.prompts) as fh:
            raw = [l.strip() for l in fh if l.strip()]
        prompts = []
        for line in raw:
            if '|' in line:
                tool, p = line.split('|', 1)
                prompt_tools[len(prompts)] = {'type': 'function', 'function': {'name': tool.strip()}}
                prompts.append(p.strip())
            else:
                prompts.append(line)
        print(f'[prompts] loaded {len(prompts)} from {args.prompts} ({len(prompt_tools)} forced tool_choice)')
    else:
        print(f'[prompts] using {len(prompts)} built-in kuhul prompts')

    teacher_tools = None
    if getattr(args, 'teacher_tools', None) and os.path.isfile(args.teacher_tools):
        with open(args.teacher_tools) as fh:
            teacher_tools = json.load(fh)
        print(f'[tools] loaded {len(teacher_tools)} teacher tools from {args.teacher_tools}')

    # Prepend Atomic DOM prompts if manifest provided
    if getattr(args, 'atomic_dom', None) and os.path.isfile(args.atomic_dom):
        dom_prompts = load_atomic_dom_prompts(args.atomic_dom)
        if dom_prompts:
            prompts = dom_prompts + prompts
            print(f'[prompts] +{len(dom_prompts)} from atomic dom -> {len(prompts)} total')

    # JSONL log setup
    logs_dir = getattr(args, 'log_dir', None) or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_ts = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    log_path = os.path.join(logs_dir, f'distillation_{log_ts}.jsonl')
    trace_path = os.path.abspath(
        getattr(args, 'trace_jsonl', None)
        or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                        'dist', 'MX-2', 'io', 'distillation.jsonl')
    )
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    def log_entry(entry: dict):
        entry.setdefault('timestamp', datetime.datetime.now().isoformat())
        with open(log_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def trace_entry(entry: dict):
        """Append a replayable teacher->student record for MX-2 promotion."""
        record = {
            'schema': 'mx2.distillation-trace.v1',
            'teacher': 'gpt-oss-20b-MXFP4',
            'student': os.path.basename(args.student),
            'source': 'tools/oss_distillation.py',
            **entry,
        }
        record.setdefault('timestamp', datetime.datetime.now().isoformat())
        with open(trace_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')

    log_entry({
        'event': 'start',
        'student': args.student,
        'out': args.out,
        'rank': args.rank,
        'steps': args.steps,
        'lr': args.lr,
        'engine': args.engine,
        'ollama_url': args.ollama_url,
        'ollama_model': args.ollama_model,
        'teacher_tokens': args.teacher_tokens,
        'prompts_file': args.prompts,
        'atomic_dom': getattr(args, 'atomic_dom', None),
        'xshard_url': getattr(args, 'xshard_url', None),
        'kuhul_server': getattr(args, 'kuhul_server', None),
        'prompt_count': len(prompts),
        'config': cfg,
        'trace_jsonl': trace_path,
    })

    teacher_mode = 'self'
    xshard_ok = False
    engine_ok = False
    ollama_ok = False

    if getattr(args, 'no_teacher', False):
        print('[teacher] --no-teacher: all teacher paths disabled, paired data only')
        # Override steps to actual pair count so we don't cycle into self-distillation
        if paired_data and args.steps > len(paired_data):
            print(f'[teacher] clamping steps {args.steps} -> {len(paired_data)} (pair count)')
            args.steps = len(paired_data)

    # Priority: xshard (local DDS) > ollama (cloud) > engine > self
    if getattr(args, 'no_teacher', False):
        pass  # skip all teacher probes
    elif getattr(args, 'xshard_url', None):
        test = teacher_complete_xshard(args.xshard_url, 'ping', max_tokens=4)
        xshard_ok = test is not None
    if not getattr(args, 'no_teacher', False):
        if not xshard_ok and args.ollama_url and args.ollama_model:
            test = teacher_complete_ollama(args.ollama_url, args.ollama_model, 'ping', max_tokens=4)
            ollama_ok = test is not None
        if not xshard_ok and not ollama_ok and args.engine:
            test = teacher_complete(args.engine, 'ping', max_tokens=4)
            engine_ok = test is not None

    if xshard_ok:
        teacher_mode = 'xshard_dds'
    elif ollama_ok:
        teacher_mode = 'ollama'
    elif engine_ok:
        teacher_mode = 'kuhul_engine'

    print(f'[teacher] xshard={getattr(args, "xshard_url", None)} -> {"OK" if xshard_ok else "skip/unreachable"}'
          f' | ollama={args.ollama_url}/{args.ollama_model} -> {"OK" if ollama_ok else "skip/unreachable"}'
          f' | engine={args.engine} -> {"OK" if engine_ok else "skip/unreachable"}'
          f' | mode={teacher_mode}')

    best_loss = float('inf')
    step      = 0

    while step < args.steps:
        # Paired data (pre-generated completions) takes priority over live teacher.
        if paired_data:
            prompt, completion = paired_data[step % len(paired_data)]
            teacher_mode = 'paired_data'
            routing = select_micronaut_for_prompt(prompt)
            routing = {'micronaut': routing['micronaut'], 'atomic_blocks': routing['atomic_blocks']}
            micronaut_system = None
        else:
            prompt = prompts[step % len(prompts)]

            # Micronaut XQuery — get grounded system context from kuhul-server
            xq = micronaut_xquery(args.kuhul_server, prompt)
            micronaut_system = xq.get('system_context')
            routing = {
                'micronaut': xq['name'],
                'atomic_blocks': xq['atomic_blocks'],
            }

            # Get teacher completion (priority: xshard_dds > ollama > engine > self)
            if xshard_ok:
                completion = teacher_complete_xshard(args.xshard_url, prompt, max_tokens=args.teacher_tokens)
                if completion is None:
                    xshard_ok = False
                    completion = prompt
            elif ollama_ok:
                completion = teacher_complete_ollama(
                    args.ollama_url, args.ollama_model, prompt,
                    max_tokens=args.teacher_tokens, system_context=micronaut_system,
                )
                if completion is None:
                    ollama_ok = False
                    completion = prompt
            elif engine_ok:
                completion = teacher_complete(args.engine, prompt, max_tokens=args.teacher_tokens,
                                              tools=teacher_tools, tool_choice=prompt_tools.get(step % len(prompts)))
                if completion is None:
                    engine_ok = False
                    completion = prompt
            else:
                completion = prompt

            if not xshard_ok and not ollama_ok and not engine_ok:
                teacher_mode = 'self'

            # Skip empty teacher responses so we don't train on silence
            if teacher_mode != 'self' and (completion is None or str(completion).strip() == ''):
                log_entry({
                    'event': 'step_skip',
                    'step': step + 1,
                    'teacher_mode': teacher_mode,
                    'prompt': prompt,
                    'reason': 'empty_teacher_response',
                    'micronaut': routing['micronaut'],
                    'atomic_blocks': routing['atomic_blocks'],
                })
                step += 1
                continue

        # Log prompt + teacher response
        sample_record = {
            'event': 'step_sample',
            'step': step + 1,
            'teacher_mode': teacher_mode,
            'prompt': prompt,
            'completion': completion,
            'micronaut': routing['micronaut'],
            'atomic_blocks': routing['atomic_blocks'],
            'system_context_used': micronaut_system is not None,
        }
        log_entry(sample_record)
        trace_entry({
            'event': 'teacher_sample',
            'step': step + 1,
            'status': 'recorded',
            'prompt': prompt,
            'response': completion,
            'completion': completion,
            'teacher_mode': teacher_mode,
            'helper': routing['micronaut'],
            'atomic_blocks': routing['atomic_blocks'],
            'system_context_used': micronaut_system is not None,
            'replay_source': log_path,
            'promotion': 'candidate_only',
        })

        # Write Q&A back to semantic micronaut store for factory pickup
        mn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'micronauts')
        write_micronaut_qa(mn_dir, routing['micronaut'], prompt, completion, step + 1)

        # Tokenize
        full_text  = prompt + '\n' + completion
        token_ids  = tokenize(full_text)[:cfg['block_size']]
        prompt_len = len(tokenize(prompt))

        if len(token_ids) < 4:
            step += 1
            continue

        input_ids  = torch.tensor([token_ids[:-1]], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:],    dtype=torch.long)

        # Forward pass (student)
        logits = gpt2_forward(state, adapters, input_ids, cfg)   # [1, T-1, V]
        logits = logits[0]                                         # [T-1, V]

        # Mask prompt tokens (only compute loss on completion)
        mask_start = max(0, prompt_len - 1)
        if mask_start >= len(target_ids):
            mask_start = 0

        loss_logits = logits[mask_start:]
        loss_targets = target_ids[mask_start:]

        loss = F.cross_entropy(
            loss_logits.float(),
            loss_targets.to(logits.device),
            ignore_index=-1
        )

        optim.zero_grad()
        loss.backward()

        # Gradient clip
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()

        step += 1
        current_loss = loss.item()
        if step % 10 == 0:
            print(f'  step {step:4d}/{args.steps}  loss={current_loss:.4f}  '
                  f'teacher={teacher_mode}')
        if current_loss < best_loss:
            best_loss = current_loss
            log_entry({
                'event': 'best_loss',
                'step': step,
                'best_loss': best_loss,
                'teacher_mode': teacher_mode,
            })

    print(f'[distill] done. best_loss={best_loss:.4f}')
    log_entry({
        'event': 'done',
        'steps_completed': step,
        'best_loss': best_loss,
        'final_teacher_mode': teacher_mode,
        'log_path': log_path,
        'trace_jsonl': trace_path,
    })
    trace_entry({
        'event': 'distillation_complete',
        'status': 'candidate_complete',
        'steps_completed': step,
        'best_loss': best_loss,
        'output': args.out,
        'replay_source': log_path,
        'promotion': 'candidate_only',
    })

    # Save LoRA weights
    lora_state = {}
    for key, (A, B_) in adapters.items():
        safe_key = key.replace('.', '__')
        lora_state[f'lora_A__{safe_key}'] = A.detach().cpu().float()
        lora_state[f'lora_B__{safe_key}'] = B_.detach().cpu().float()

    out_path = args.out
    save_file(lora_state, out_path)
    size_mb = os.path.getsize(out_path) // (1024 * 1024)
    print(f'[distill] LoRA saved: {out_path}  ({size_mb} MB, {len(lora_state)} tensors)')

    # Verify
    with safe_open(out_path, framework='pt') as f:
        keys = list(f.keys())
    print(f'[verify] {len(keys)} LoRA tensors, e.g.: {keys[:2]}')


def load_atomic_dom_prompts(manifest_path: str) -> list:
    """Extract distillation prompts from an Atomic DOM manifest.

    Reads npc.system_prompt, npc.role, and npc.rules — plain English content only.
    No block→prompt fabrication; only what the manifest declares.
    """
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)

    npc = manifest.get('app', {}).get('npc', {})
    name = npc.get('name', 'the model')
    role = npc.get('role', '').strip()
    system_prompt = npc.get('system_prompt', '').strip()
    rules = [r.strip() for r in npc.get('rules', []) if r.strip()]

    prompts = []

    if role:
        prompts.append(f'What is the role of {name}?')

    if system_prompt:
        prompts.append(f'Introduce yourself as {name}.')
        prompts.append(f'Summarize what {name} does in two sentences.')

    for rule in rules:
        prompts.append(f'Apply this execution principle to a user request: {rule}')

    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--student',       default='models/from_zero/from_zero_v0.1_folded.safetensors')
    ap.add_argument('--out',           default='models/from_zero/from_zero_v0.1_lora.safetensors')
    ap.add_argument('--rank',    type=int,   default=8,     help='LoRA rank (default 8)')
    ap.add_argument('--steps',   type=int,   default=200,   help='Training steps')
    ap.add_argument('--lr',      type=float, default=1e-4,  help='Learning rate')
    ap.add_argument('--engine',  default='http://127.0.0.1:8787', help='local teacher API base URL')
    ap.add_argument('--prompts', default=None, help='Path to prompts file (one per line)')
    ap.add_argument('--atomic-dom', default=None, dest='atomic_dom',
                    help='Atomic DOM manifest.json — extracts npc role/rules as extra prompts')
    ap.add_argument('--xshard-url', default=None, dest='xshard_url',
                    help='Local xshard DDS server URL (e.g. http://127.0.0.1:1235) — highest-priority teacher')
    ap.add_argument('--teacher-tokens', type=int, default=200, help='Max tokens from teacher per step')
    ap.add_argument('--teacher-tools', default=None, help='Path to a JSON file of OpenAI tools schema (teacher emits tool_calls)')
    ap.add_argument('--ollama-url',   default='http://127.0.0.1:11434', help='Ollama API base URL (alternative teacher)')
    ap.add_argument('--ollama-model', default='gpt-oss:120b-cloud',      help='Ollama model name for teacher completions')
    ap.add_argument('--resume', action='store_true', help='Resume from existing --out LoRA checkpoint (load A/B weights)')
    ap.add_argument('--log-dir', default=None, help='Directory for distillation JSONL logs (default: ../logs)')
    ap.add_argument('--trace-jsonl', default=None, dest='trace_jsonl',
                    help='Canonical MX-2 distillation JSONL output (default: dist/MX-2/io/distillation.jsonl)')
    ap.add_argument('--kuhul-server', default='http://127.0.0.1:8764', dest='kuhul_server',
                    help='kuhul-server URL for micronaut XQuery (default: http://127.0.0.1:8764)')
    ap.add_argument('--no-teacher', action='store_true', dest='no_teacher',
                    help='Disable all teacher calls (paired data only). Stops at end of data '
                         'if steps exceeds pair count rather than falling through to any teacher.')
    ap.add_argument('--jsonl-data', default=None, dest='jsonl_data',
                    help='Comma-separated JSONL/CSV paths with pre-generated prompt+completion pairs. '
                         'Skips live teacher; supports {prompt/completion}, {messages}, {input/output}, '
                         '{instruction/output} and CSV question/answer columns.')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
