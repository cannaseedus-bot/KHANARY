#!/usr/bin/env python3
# eval_field_consistency.py -- A/B metric for the field-guided finetune: on HELD-OUT transitions,
# does the model assign higher probability to the Delta-tokens Trinity's field endorses?
#
# For each held-out example we run the model, take log P(target) at every position, and split
# positions by whether the target token is field-endorsed (given that example's Preserve-state).
#   endorsed_lp   = mean log-prob the model gives to field-endorsed target tokens
#   overall_lp    = mean log-prob over all target tokens
#   ALIGNMENT     = endorsed_lp - overall_lp   (how much the model favors field concepts)
# A field-guided model (B) should have a HIGHER alignment than the unguided one (A).
#
# Usage: python tools/eval_field_consistency.py <model_dir> <heldout.jsonl> <field.json> [--limit N]

import sys, os, json, argparse, math
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_hf import detect_and_load, field_endorsed_ids
from trinity_field import TrinityField
from transformers import GPT2TokenizerFast

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("heldout"); ap.add_argument("field")
    ap.add_argument("--limit", type=int, default=400); ap.add_argument("--seq", type=int, default=128)
    a = ap.parse_args()
    torch.set_num_threads(4)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    fld = TrinityField().load(a.field)
    model, _ = detect_and_load(os.path.join(a.model_dir, "model.safetensors"))
    model.eval()

    end_lp_sum = end_n = all_lp_sum = all_n = 0
    with open(a.heldout, encoding="utf-8") as f, torch.no_grad():
        for line in f:
            try: r = json.loads(line)
            except Exception: continue
            t = r.get("text")
            if not t: continue
            ids = tok.encode(t)[: a.seq]
            if len(ids) < 8: continue
            endorsed = field_endorsed_ids(fld, r.get("preserve", []), tok)
            x = torch.tensor([ids])
            logp = F.log_softmax(model(input_ids=x).logits[0], dim=-1)  # [T, V]
            for pos in range(len(ids) - 1):
                tgt = ids[pos + 1]
                lp = logp[pos, tgt].item()
                all_lp_sum += lp; all_n += 1
                if tgt in endorsed:
                    end_lp_sum += lp; end_n += 1
            a.limit -= 1
            if a.limit <= 0: break

    end_lp = end_lp_sum / max(end_n, 1)
    all_lp = all_lp_sum / max(all_n, 1)
    print(f"[{os.path.basename(a.model_dir)}]  endorsed_lp={end_lp:.4f}  overall_lp={all_lp:.4f}  "
          f"ALIGNMENT={end_lp - all_lp:+.4f}  (endorsed positions={end_n}/{all_n})")
    # emit machine-readable line for A/B comparison
    print(f"RESULT {a.model_dir} {end_lp:.5f} {all_lp:.5f} {end_lp-all_lp:.5f}")

if __name__ == "__main__":
    main()
