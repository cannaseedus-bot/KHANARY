#!/usr/bin/env python3
"""
kuhul_dataset_validator.py -- Validate and compile π-KUHUL structured training records.

Input format:  {"messages": [{"role": "...", "content": "...", "weight_tensor": [...]}]}
Output format: {"text": "<INSTRUCT>\n...\n</INSTRUCT>\n..."} JSONL

Roles and their hardware tag boundaries (from tokenizer_config.json):
  instruction  <INSTRUCT>  </INSTRUCT>   ID 50266/50267
  user         <USER>      </USER>       ID 50268/50269
  thought      <THINK>     </THINK>      ID 50262/50263  (requires weight_tensor[5])
  tool_call    <TOOL_CALL> </TOOL_CALL>  ID 50264/50265
  agent        <AGENT>     </AGENT>      ID 50260/50261

Usage:
  python tools/kuhul_dataset_validator.py dataset.jsonl -o compiled_train.jsonl
  python tools/kuhul_dataset_validator.py dataset.jsonl --stats
"""

import json, sys, argparse
from typing import List, Dict, Any

TENSOR_DIM = 5

TAGS = {
    "instruction": ("<INSTRUCT>",   "</INSTRUCT>"),
    "user":        ("<USER>",       "</USER>"),
    "thought":     ("<THINK>",      "</THINK>"),
    "tool_call":   ("<TOOL_CALL>",  "</TOOL_CALL>"),
    "agent":       ("<AGENT>",      "</AGENT>"),
}


class KuhulDatasetValidator:
    def __init__(self, tensor_dim: int = TENSOR_DIM):
        self.tensor_dim = tensor_dim
        self.tags = TAGS

    def validate_structure(self, filepath: str) -> List[Dict[str, Any]]:
        valid_records = []
        total = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                total = line_idx

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as je:
                    print(f"[Line {line_idx}] JSON parse error: {je}")
                    continue

                messages = data.get("messages")
                if not messages:
                    print(f"[Line {line_idx}] Error: missing or empty 'messages' array")
                    continue

                is_valid = True
                for msg in messages:
                    role    = msg.get("role", "")
                    content = msg.get("content", "")

                    if role not in self.tags:
                        print(f"[Line {line_idx}] Error: unknown role '{role}'")
                        is_valid = False
                        break

                    if role == "thought":
                        wt = msg.get("weight_tensor")
                        if not isinstance(wt, list) or len(wt) != self.tensor_dim:
                            print(f"[Line {line_idx}] Error: 'thought' requires weight_tensor[{self.tensor_dim}]")
                            is_valid = False
                            break
                        if not all(isinstance(x, (int, float)) and 0.0 <= float(x) <= 1.0 for x in wt):
                            print(f"[Line {line_idx}] Error: weight_tensor values must be floats in [0, 1]")
                            is_valid = False
                            break

                    open_tag, close_tag = self.tags[role]
                    if open_tag in content or close_tag in content:
                        print(f"[Line {line_idx}] Warning: role '{role}' content contains unescaped tag")

                if is_valid:
                    valid_records.append(data)

        print(f"\n[Validation] {len(valid_records)} / {total} records passed")
        return valid_records

    def compile_to_training_tokens(self, valid_records: List[Dict[str, Any]],
                                   output_path: str) -> int:
        """
        Transforms verified role structures into text JSONL delimited by KUHUL hardware tags.
        Each thought block appends its weight tensor as a WEIGHT: line so the model trains
        to predict the geometric bias coefficients g0-g4 from context.
        """
        written = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in valid_records:
                parts = []
                for msg in rec["messages"]:
                    role    = msg["role"]
                    content = msg["content"].strip()
                    open_tag, close_tag = self.tags[role]

                    if role == "thought":
                        wt = msg.get("weight_tensor", [])
                        wt_str = "[" + ", ".join(f"{v:.3f}" for v in wt) + "]"
                        parts.append(f"{open_tag}\nWEIGHT: {wt_str}\n{content}\n{close_tag}")
                    else:
                        parts.append(f"{open_tag}\n{content}\n{close_tag}")

                text = "\n".join(parts)
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                written += 1

        print(f"[Compile] wrote {written:,} records -> {output_path}")
        return written


def main():
    ap = argparse.ArgumentParser(
        description="Validate and compile π-KUHUL structured training records"
    )
    ap.add_argument("input", help="Input JSONL with {messages: [...]} records")
    ap.add_argument("-o", "--output", default=None,
                    help="Output path for compiled {text: ...} JSONL (omit for --stats only)")
    ap.add_argument("--stats", action="store_true",
                    help="Validation report only — do not write output")
    ap.add_argument("--tensor-dim", type=int, default=TENSOR_DIM,
                    help=f"Expected weight_tensor dimension (default {TENSOR_DIM})")
    a = ap.parse_args()

    if not a.stats and not a.output:
        ap.error("Provide -o OUTPUT or --stats")

    v = KuhulDatasetValidator(tensor_dim=a.tensor_dim)
    records = v.validate_structure(a.input)

    if not a.stats and a.output:
        v.compile_to_training_tokens(records, a.output)


if __name__ == "__main__":
    main()
