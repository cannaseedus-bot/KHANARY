"""oss_distillation.py — GPT-OSS teacher -> from_zero LoRA distillation

Uses kuhul_engine HTTP API (port 17474) as the GPT-OSS teacher.
Trains low-rank adapter weights (LoRA) on from_zero_v0.6 to match teacher outputs.
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
        --student  models/from_zero/from_zero_v0.6_merged.safetensors \
        --out      models/from_zero/from_zero_v0.6_lora.safetensors \
        --rank     8 \
        --steps    500 \
        --lr       1e-4 \
        --engine   http://127.0.0.1:17474 \
        --prompts  tools/distill_prompts.txt

If --engine is unreachable, the script falls back to self-distillation
(student teaches itself — useful for adapter shape validation without the engine running).
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
                     ollama_url: str = None, ollama_model: str = None) -> str:
    if ollama_url and ollama_model:
        return teacher_complete_ollama(ollama_url, ollama_model, prompt, max_tokens)
    # Fallback to original kuhul_engine path
    payload = json.dumps({
        'model': 'gpt-oss-20b-MXFP4.gguf',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': 0.7,
        'stream': False
    }).encode()
    req = urllib.request.Request(
        f'{engine_base}/v1/chat/completions',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data['choices'][0]['message']['content']
    except Exception as e:
        return None

def teacher_complete_ollama(ollama_url: str, ollama_model: str, prompt: str, max_tokens: int = 256) -> str:
    cache_key = (ollama_url, ollama_model, prompt, max_tokens)
    if cache_key in _OLLAMA_CACHE:
        return _OLLAMA_CACHE[cache_key]

    payload = json.dumps({
        'model': ollama_model,
        'messages': [
            {'role': 'system', 'content': 'Answer briefly and directly.'},
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

    # Load prompts
    prompts = DEFAULT_PROMPTS[:]
    if args.prompts and os.path.isfile(args.prompts):
        with open(args.prompts) as fh:
            prompts = [l.strip() for l in fh if l.strip()]
        print(f'[prompts] loaded {len(prompts)} from {args.prompts}')
    else:
        print(f'[prompts] using {len(prompts)} built-in kuhul prompts')

    # JSONL log setup
    logs_dir = getattr(args, 'log_dir', None) or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_ts = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    log_path = os.path.join(logs_dir, f'distillation_{log_ts}.jsonl')
    def log_entry(entry: dict):
        entry.setdefault('timestamp', datetime.datetime.now().isoformat())
        with open(log_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + '\n')

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
        'prompt_count': len(prompts),
        'config': cfg,
    })

    teacher_mode = 'self'
    engine_ok = False
    ollama_ok = False
    if args.engine:
        test = teacher_complete(args.engine, 'ping', max_tokens=4)
        engine_ok = test is not None
    if not engine_ok and args.ollama_url and args.ollama_model:
        test = teacher_complete_ollama(args.ollama_url, args.ollama_model, 'ping', max_tokens=4)
        ollama_ok = test is not None
    if engine_ok:
        teacher_mode = 'kuhul_engine'
    elif ollama_ok:
        teacher_mode = 'ollama'
    print(f'[teacher] engine={args.engine} -> {"OK" if engine_ok else "unreachable"} | ollama={args.ollama_url}/{args.ollama_model} -> {"OK" if ollama_ok else "unreachable"} | mode={teacher_mode}')

    best_loss = float('inf')
    step      = 0

    while step < args.steps:
        prompt = prompts[step % len(prompts)]

        # Get teacher completion
        if engine_ok:
            completion = teacher_complete(args.engine, prompt, max_tokens=args.teacher_tokens)
            if completion is None:
                engine_ok = False
                completion = prompt   # self-distill fallback
        elif ollama_ok:
            completion = teacher_complete_ollama(args.ollama_url, args.ollama_model, prompt, max_tokens=args.teacher_tokens)
            if completion is None:
                ollama_ok = False
                completion = prompt
        else:
            completion = prompt

        # Update mode label if we fell back during the run
        if not engine_ok and not ollama_ok:
            teacher_mode = 'self'

        routing = select_micronaut_for_prompt(prompt)

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
        log_entry({
            'event': 'step_sample',
            'step': step + 1,
            'teacher_mode': teacher_mode,
            'prompt': prompt,
            'completion': completion,
            'micronaut': routing['micronaut'],
            'atomic_blocks': routing['atomic_blocks'],
        })

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--student',       default='models/from_zero/from_zero_v0.6_merged.safetensors')
    ap.add_argument('--out',           default='models/from_zero/from_zero_v0.6_lora.safetensors')
    ap.add_argument('--rank',    type=int,   default=8,     help='LoRA rank (default 8)')
    ap.add_argument('--steps',   type=int,   default=500,   help='Training steps')
    ap.add_argument('--lr',      type=float, default=1e-4,  help='Learning rate')
    ap.add_argument('--engine',  default='http://127.0.0.1:17474', help='kuhul_engine base URL')
    ap.add_argument('--prompts', default=None, help='Path to prompts file (one per line)')
    ap.add_argument('--teacher-tokens', type=int, default=200, help='Max tokens from teacher per step')
    ap.add_argument('--ollama-url',   default='http://127.0.0.1:11434', help='Ollama API base URL (alternative teacher)')
    ap.add_argument('--ollama-model', default='gpt-oss:120b-cloud',      help='Ollama model name for teacher completions')
    ap.add_argument('--resume', action='store_true', help='Resume from existing --out LoRA checkpoint (load A/B weights)')
    ap.add_argument('--log-dir', default=None, help='Directory for distillation JSONL logs (default: ../logs)')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
