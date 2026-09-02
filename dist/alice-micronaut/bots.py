"""
ALICE-1 AliceMicronaut — bots.py
Gram-fold pure AIML symbolic agent. No neural generation.
Corpus: micronaut-v4/ELIZA/alice/ (mp0-mp6, ~7.3M tokens AIML).
Fold: ⟁GRAM_FOLD⟁
"""

import sys
import re
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

_MICRONAUT_ID   = "ALICE-1"
_MICRONAUT_NAME = "AliceMicronaut"
_MICRONAUT_FOLD = "⟁GRAM_FOLD⟁"

_CORPUS_DIR = Path(__file__).parents[2] / "micronaut-v4" / "ELIZA" / "alice"

# ---------------------------------------------------------------------------
# Minimal AIML engine
# ---------------------------------------------------------------------------

class AimlEngine:
    def __init__(self, max_srai: int = 10):
        self._rules: list[tuple[str, str, str]] = []  # (topic_pat, that_pat, input_pat) → template
        self._templates: dict[tuple[str,str,str], str] = {}
        self._loaded_files: list[str] = []
        self.max_srai = max_srai

    def load_file(self, path: Path):
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            for cat in root.iter("category"):
                pat_el    = cat.find("pattern")
                that_el   = cat.find("that")
                topic_el  = cat.find("topic")
                tmpl_el   = cat.find("template")
                if pat_el is None or tmpl_el is None:
                    continue
                pat   = (pat_el.text or "").strip().upper()
                that  = (that_el.text  or "*").strip().upper() if that_el  is not None else "*"
                topic = (topic_el.text or "*").strip().upper() if topic_el is not None else "*"
                key   = (topic, that, pat)
                self._rules.append(key)
                self._templates[key] = "".join(tmpl_el.itertext()).strip()
            self._loaded_files.append(str(path.name))
        except Exception:
            pass

    def load_corpus(self, corpus_dir: Path, files: list[str] | None = None):
        if files is None:
            files = ["bot.aiml", "default.aiml", "that.aiml", "personality.aiml"]
        for name in files:
            f = corpus_dir / name
            if f.exists():
                self.load_file(f)

    def _match(self, pattern: str, text: str) -> bool:
        regex = "^" + re.escape(pattern).replace(r"\*", ".*").replace(r"\#", r"\S+") + "$"
        return bool(re.match(regex, text, re.IGNORECASE))

    def respond(self, input_text: str, topic: str = "*", that: str = "*",
                depth: int = 0) -> str:
        if depth >= self.max_srai:
            return ""
        norm = input_text.strip().upper()
        for key in self._rules:
            t_pat, th_pat, in_pat = key
            if (self._match(in_pat, norm)
                    and self._match(t_pat, topic.upper())
                    and self._match(th_pat, that.upper())):
                tmpl = self._templates[key]
                # Resolve SRAI
                tmpl = re.sub(
                    r"<srai>(.*?)</srai>",
                    lambda m: self.respond(m.group(1), topic, that, depth + 1),
                    tmpl, flags=re.IGNORECASE | re.DOTALL
                )
                return tmpl
        return "I don't know."


_ENGINE: AimlEngine | None = None


def _get_engine() -> AimlEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = AimlEngine()
        _ENGINE.load_corpus(_CORPUS_DIR)
    return _ENGINE

# ---------------------------------------------------------------------------

def health_check() -> dict:
    eng = _get_engine()
    return {
        "status": "ok",
        "micronaut": _MICRONAUT_ID,
        "files_loaded": len(eng._loaded_files),
        "rules_loaded": len(eng._rules),
        "corpus_dir": str(_CORPUS_DIR),
    }


def chat(input_text: str, topic: str = "*", that: str = "*") -> str:
    t0  = time.perf_counter()
    eng = _get_engine()
    response = eng.respond(input_text, topic, that)
    latency  = (time.perf_counter() - t0) * 1000
    return json.dumps({
        "micronaut": _MICRONAUT_ID,
        "fold": _MICRONAUT_FOLD,
        "input": input_text,
        "response": response,
        "topic": topic,
        "that": response,   # next turn's <that> = this response
        "latency_ms": round(latency, 2),
    })


def dispatch(task: str, payload: dict | None = None) -> str:
    payload = payload or {}
    t = task.lower()
    if t in ("chat", "generate", "respond", "run"):
        return chat(payload.get("input", payload.get("prompt", "")),
                    payload.get("topic", "*"),
                    payload.get("that", "*"))
    if t == "health":
        return json.dumps(health_check())
    if t == "reset":
        global _ENGINE
        _ENGINE = None
        return json.dumps({"status": "reset", "micronaut": _MICRONAUT_ID})
    return json.dumps({"error": f"Unknown task '{task}'",
                       "supported": ["chat", "health", "reset"],
                       "micronaut": _MICRONAUT_ID})


if __name__ == "__main__":
    print(f"[{_MICRONAUT_ID}] {_MICRONAUT_NAME} smoke test")
    h = health_check()
    print(f"[1] health: rules={h.get('rules_loaded',0)} files={h.get('files_loaded',0)}")
    r = json.loads(chat("Hello"))
    print(f"[2] chat('Hello') → {r.get('response', '')[:100]}")
    r2 = json.loads(chat("What is your name?"))
    print(f"[3] chat('What is your name?') → {r2.get('response', '')[:100]}")
    print(f"[{_MICRONAUT_ID}] done")
