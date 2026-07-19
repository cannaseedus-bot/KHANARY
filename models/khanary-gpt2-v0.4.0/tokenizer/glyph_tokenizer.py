#!/usr/bin/env python3
"""
glyph_tokenizer.py — GlyphTokenizer for pi-kuhul / mini-GPT shared vocabulary.

Implements the 1024-token glyph opcode vocabulary defined in glyph_opcode_system.xjson.

Vocabulary layout:
  0x000-0x00F  Control tokens     (PAD=0, BOS=1, EOS=2, UNK=3, ...)
  0x010-0x01F  Scalar compute     (LOAD=16, SCALE=17, ACCUM=18, RELU=19, ...)
  0x020-0x02F  Vector compute     (DOT=32, NORM=34, SOFTMAX=47, ...)
  0x030-0x04F  Tools              (T_READ=48, T_WRITE=49, T_BASH=53, ...)
  0x050-0x05F  Agents / skills    (T_AGENT_WAIT=80, SKILL_CALL=83, ...)
  0x060-0x06F  Micronauts / XCFE  (ATLAS_INIT=96, GRAM_EXEC=97, ...)
  0x070-0x07F  Commands / intent  (CMD_BUILD=112, I_QUESTION=120, ...)
  0x080-0x3FF  Language slots     (word_hash mod 896 + 128)

INT4 hot-path: nibble N → token_id = N + 16 (compute tier).

Special sequences:
  Tool calls:   [TOOL_CALL(11), T_XXX(id), ...args..., TOOL_RESULT(12)]
  Code blocks:  [CODE_START(7), ...tokens..., CODE_END(8)]
  Think blocks: [THINK_START(13), ...tokens..., THINK_END(14)]
  Turn sep:     [SEP(4)]
"""

import json, zlib
from pathlib import Path

# ─── Token ID constants ────────────────────────────────────────────────────────

PAD          = 0
BOS          = 1
EOS          = 2
UNK          = 3
SEP          = 4
MASK         = 5
NL           = 6
CODE_START   = 7
CODE_END     = 8
QUOTE_START  = 9
QUOTE_END    = 10
TOOL_CALL    = 11
TOOL_RESULT  = 12
THINK_START  = 13
THINK_END    = 14
NOP_TOKEN    = 15

# Compute tier (token_id = nibble + 16)
LOAD         = 16
SCALE        = 17
ACCUM        = 18
RELU         = 19
ADD          = 20
MUL          = 21
DIV          = 22
EMIT         = 23
READ_S       = 24
WRITE_S      = 25
BRANCH       = 26
HALT         = 27
RETURN       = 28
CALL         = 29
PUSH         = 30
POP          = 31

# Vector tier
DOT          = 32
LEN          = 33
NORM         = 34
VSUB         = 35
LERP         = 36
CROSS        = 37
PROJ         = 38
EXP          = 39
VSUM         = 40
VDIV         = 41
RSQRT        = 42
CLAMP        = 43
MAX4         = 44
MIN4         = 45
FMADD        = 46
SOFTMAX      = 47

# Tool tier
T_READ       = 48
T_WRITE      = 49
T_EDIT       = 50
T_GLOB       = 51
T_GREP       = 52
T_BASH       = 53
T_BASH_BG    = 54
T_WEB_FETCH  = 55
T_WEB_SEARCH = 56
T_ASK        = 57
T_NB_EDIT    = 58
T_MCP_CALL   = 59
T_MCP_RES    = 60
T_REMOTE     = 61
T_TOOL_SRCH  = 62
T_SKILL      = 63

T_TASK_CREATE = 64
T_TASK_UPDATE = 65
T_TASK_GET    = 66
T_TASK_LIST   = 67
T_TASK_STOP   = 68
T_TASK_OUT    = 69
T_PLAN_ENTER  = 70
T_PLAN_EXIT   = 71
T_WT_ENTER    = 72
T_WT_EXIT     = 73
T_CONFIG      = 74
T_CRON_ADD    = 75
T_CRON_DEL    = 76
T_CRON_LIST   = 77
T_AGENT_SPAWN = 78
T_AGENT_MSG   = 79

T_AGENT_WAIT  = 80
T_AGENT_KILL  = 81
T_AGENT_RES   = 82
SKILL_CALL    = 83
SKILL_LOAD    = 84
SKILL_LIST    = 85
SKILL_RES     = 86
SCHED_AGENT   = 87
REMOTE_TRIG   = 88
PERM_ADD      = 89
PERM_DEL      = 90
ENV_SET       = 91
HOOK_ADD      = 92
HOOK_DEL      = 93
MEM_READ      = 94
MEM_WRITE     = 95

ATLAS_INIT    = 96
GRAM_EXEC     = 97
GLYPH_DISP    = 98
K_AYAB        = 99
SHARD_LOAD    = 100
SHARD_HASH    = 101
SHARD_FETCH   = 102
SHARD_POLL    = 103
SCO_CALL      = 104
SCO_RESOLVE   = 105
SCO_AWAIT     = 106
SCO_EMIT      = 107
MN_SPAWN      = 108
MN_LOAD       = 109
MN_EXEC       = 110
MN_STOP       = 111

CMD_BUILD     = 112
CMD_TEST      = 113
CMD_DEPLOY    = 114
CMD_COMMIT    = 115
CMD_PUSH      = 116
CMD_PULL      = 117
CMD_BRANCH    = 118
CMD_MERGE     = 119
I_QUESTION    = 120
I_ANSWER      = 121
I_CODE        = 122
I_EXPLAIN     = 123
I_FIX         = 124
I_CREATE      = 125
I_UPDATE      = 126
I_ANALYZE     = 127

VOCAB_SIZE    = 1024
LANG_BASE     = 128
LANG_SLOTS    = 894   # 1024 - 128 - 2 (ids 1022-1023 reserved for agent_pool tier)

# ─── Extended agent_pool tier (v5) — carved from top of atlas ─────────────────
# agents tier (80-95) + micronauts tier (96-111) were full, so these dedicated
# agent-pool members occupy the top two atlas slots (out of the language range).
CSSC_AGENT    = 1022  # Agent: CSS-C compression-calculus
COMPUTE_SHEET = 1023  # Micronaut: compute-sheet (SVG/CSS DFA compute primitive)

# INT4 nibble offset: nibble N = token_id N + 16
INT4_NIBBLE_OFFSET = 16

# ─── Tool name → token ID mapping ─────────────────────────────────────────────

TOOL_TO_TOKEN = {
    'Read':              T_READ,
    'Write':             T_WRITE,
    'Edit':              T_EDIT,
    'Glob':              T_GLOB,
    'Grep':              T_GREP,
    'Bash':              T_BASH,
    'WebFetch':          T_WEB_FETCH,
    'WebSearch':         T_WEB_SEARCH,
    'AskUserQuestion':   T_ASK,
    'NotebookEdit':      T_NB_EDIT,
    'TaskCreate':        T_TASK_CREATE,
    'TaskUpdate':        T_TASK_UPDATE,
    'TaskGet':           T_TASK_GET,
    'TaskList':          T_TASK_LIST,
    'TaskStop':          T_TASK_STOP,
    'TaskOutput':        T_TASK_OUT,
    'EnterPlanMode':     T_PLAN_ENTER,
    'ExitPlanMode':      T_PLAN_EXIT,
    'EnterWorktree':     T_WT_ENTER,
    'ExitWorktree':      T_WT_EXIT,
    'CronCreate':        T_CRON_ADD,
    'CronDelete':        T_CRON_DEL,
    'CronList':          T_CRON_LIST,
    'RemoteTrigger':     T_REMOTE,
    'ToolSearch':        T_TOOL_SRCH,
    'Skill':             T_SKILL,
    'Agent':             T_AGENT_SPAWN,
}

TOKEN_TO_TOOL = {v: k for k, v in TOOL_TO_TOKEN.items()}

# ─── Tokenizer ────────────────────────────────────────────────────────────────

class GlyphTokenizer:
    """
    Encodes text (and optionally structured tool/agent sequences) to glyph token IDs.

    The 1024-token vocabulary is shared between pi-kuhul and mini-GPT (retokenized),
    enabling weight merging of transformer layers across both models.

    Encoding:
      - Words → hash into language slots (128-1023)
      - Newlines → NL token (6)
      - Tool invocations → TOOL_CALL + T_XXX + args + TOOL_RESULT
      - Code blocks → CODE_START + tokens + CODE_END
      - Turn boundaries → SEP token (4)
    """

    PAD  = PAD
    BOS  = BOS
    EOS  = EOS
    UNK  = UNK
    VOCAB_SIZE = VOCAB_SIZE

    def _word_to_id(self, word: str) -> int:
        """Hash a word into a language slot (128-1023). Uses CRC32 — deterministic across sessions."""
        return (zlib.crc32(word.encode()) & 0x7FFFFFFF) % LANG_SLOTS + LANG_BASE

    def encode(self, text: str, max_len: int = 512, add_special: bool = True) -> list:
        """
        Encode a text string to a list of glyph token IDs.

        Recognizes:
          - Newlines: mapped to NL(6)
          - Words: hashed into language slots
          - No tool structure is injected here; use encode_turn() for that.
        """
        tokens = [BOS] if add_special else []

        for line in text.split('\n'):
            if line.strip():
                for word in line.split():
                    tokens.append(self._word_to_id(word))
            tokens.append(NL)

        # Remove trailing NL
        while tokens and tokens[-1] == NL:
            tokens.pop()

        if add_special:
            tokens.append(EOS)

        tokens = tokens[:max_len]
        tokens += [PAD] * (max_len - len(tokens))
        return tokens

    def encode_raw(self, text: str, add_special: bool = True) -> list:
        """Encode text without padding or truncation."""
        tokens = [BOS] if add_special else []

        for line in text.split('\n'):
            if line.strip():
                for word in line.split():
                    tokens.append(self._word_to_id(word))
            tokens.append(NL)

        while tokens and tokens[-1] == NL:
            tokens.pop()

        if add_special:
            tokens.append(EOS)

        return tokens

    def encode_turn(self, role: str, text: str, max_len: int = 512) -> list:
        """
        Encode a single conversation turn with role-aware intent token prefix.

        role: 'human' | 'assistant' | 'tool' | 'system'
        """
        tokens = []

        if role == 'human':
            tokens.append(I_QUESTION)
        elif role == 'assistant':
            tokens.append(I_ANSWER)
        elif role == 'system':
            tokens.append(I_EXPLAIN)

        tokens.extend(self.encode(text, max_len=max_len - 2, add_special=False))
        tokens.append(SEP)
        return tokens

    def encode_tool_call(self, tool_name: str, content: str) -> list:
        """
        Encode a tool invocation: TOOL_CALL + T_XXX + content words + TOOL_RESULT.

        Example:
          encode_tool_call('Read', 'C:/foo.txt') →
          [TOOL_CALL(11), T_READ(48), <word tokens>, TOOL_RESULT(12)]
        """
        tokens = [TOOL_CALL]
        tool_id = TOOL_TO_TOKEN.get(tool_name, UNK)
        tokens.append(tool_id)
        for word in content.split():
            tokens.append(self._word_to_id(word))
        tokens.append(TOOL_RESULT)
        return tokens

    def encode_dialogue(self, records: list, max_len: int = 512) -> list:
        """
        Encode a multi-turn dialogue record into a flat glyph sequence.

        records: list of dicts with keys 'role' and 'text' (or 'content')

        Returns a padded token list of length max_len.
        """
        tokens = [BOS]
        for rec in records:
            role = rec.get('role', 'assistant')
            text = rec.get('text', rec.get('content', rec.get('output', '')))
            turn_tokens = self.encode_turn(role, text, max_len=max_len)
            tokens.extend(turn_tokens)
            if len(tokens) >= max_len:
                break
        tokens.append(EOS)
        tokens = tokens[:max_len]
        tokens += [PAD] * (max_len - len(tokens))
        return tokens

    def decode(self, ids: list, skip_special: bool = False) -> str:
        """
        Decode token IDs back to a human-readable string.

        Named tokens (0-127) decode to their symbol name.
        Language tokens (128-1023) decode as <slot_NNN>.
        """
        parts = []
        specials = {PAD, BOS, EOS, NL, SEP} if skip_special else {}
        for tid in ids:
            if tid in specials:
                if tid == NL:
                    parts.append('\n')
                continue
            if tid == PAD:
                parts.append('[PAD]')
            elif tid == BOS:
                parts.append('[BOS]')
            elif tid == EOS:
                parts.append('[EOS]')
            elif tid == NL:
                parts.append('\n')
            elif tid == SEP:
                parts.append('[SEP]')
            elif 0 <= tid < LANG_BASE:
                parts.append(f'[{_ID_TO_NAME.get(tid, f"T{tid}")}]')
            elif LANG_BASE <= tid < VOCAB_SIZE:
                parts.append(f'<s{tid}>')
            else:
                parts.append(f'<unk{tid}>')
        return ' '.join(parts)

    def nibble_to_token(self, nibble: int) -> int:
        """Convert INT4 gram nibble (0-15) to compute-tier token ID."""
        return nibble + INT4_NIBBLE_OFFSET

    def token_to_nibble(self, token_id: int) -> int:
        """Convert compute-tier token ID to INT4 gram nibble. Returns -1 if not in compute tier."""
        n = token_id - INT4_NIBBLE_OFFSET
        return n if 0 <= n < 16 else -1


# ─── Reverse lookup table ─────────────────────────────────────────────────────

_ID_TO_NAME = {}

def _build_id_map():
    _known = [
        (0,'PAD'),(1,'BOS'),(2,'EOS'),(3,'UNK'),(4,'SEP'),(5,'MASK'),
        (6,'NL'),(7,'CODE+'),(8,'CODE-'),(9,'QUOTE+'),(10,'QUOTE-'),
        (11,'TOOL+'),(12,'TOOL-'),(13,'THINK+'),(14,'THINK-'),(15,'NOP'),
        (16,'LOAD'),(17,'SCALE'),(18,'ACCUM'),(19,'RELU'),(20,'ADD'),
        (21,'MUL'),(22,'DIV'),(23,'EMIT'),(24,'READ_S'),(25,'WRITE_S'),
        (26,'BRANCH'),(27,'HALT'),(28,'RETURN'),(29,'CALL'),(30,'PUSH'),(31,'POP'),
        (32,'DOT'),(33,'LEN'),(34,'NORM'),(35,'VSUB'),(36,'LERP'),(37,'CROSS'),
        (38,'PROJ'),(39,'EXP'),(40,'VSUM'),(41,'VDIV'),(42,'RSQRT'),(43,'CLAMP'),
        (44,'MAX4'),(45,'MIN4'),(46,'FMADD'),(47,'SOFTMAX'),
        (48,'T_READ'),(49,'T_WRITE'),(50,'T_EDIT'),(51,'T_GLOB'),(52,'T_GREP'),
        (53,'T_BASH'),(54,'T_BASH_BG'),(55,'T_WEB_FETCH'),(56,'T_WEB_SEARCH'),
        (57,'T_ASK'),(58,'T_NB_EDIT'),(59,'T_MCP_CALL'),(60,'T_MCP_RES'),
        (61,'T_REMOTE'),(62,'T_TOOL_SRCH'),(63,'T_SKILL'),
        (64,'T_TASK_CREATE'),(65,'T_TASK_UPDATE'),(66,'T_TASK_GET'),
        (67,'T_TASK_LIST'),(68,'T_TASK_STOP'),(69,'T_TASK_OUT'),
        (70,'T_PLAN_ENTER'),(71,'T_PLAN_EXIT'),(72,'T_WT_ENTER'),(73,'T_WT_EXIT'),
        (74,'T_CONFIG'),(75,'T_CRON_ADD'),(76,'T_CRON_DEL'),(77,'T_CRON_LIST'),
        (78,'T_AGENT_SPAWN'),(79,'T_AGENT_MSG'),
        (80,'T_AGENT_WAIT'),(81,'T_AGENT_KILL'),(82,'T_AGENT_RES'),
        (83,'SKILL_CALL'),(84,'SKILL_LOAD'),(85,'SKILL_LIST'),(86,'SKILL_RES'),
        (87,'SCHED_AGENT'),(88,'REMOTE_TRIG'),(89,'PERM_ADD'),(90,'PERM_DEL'),
        (91,'ENV_SET'),(92,'HOOK_ADD'),(93,'HOOK_DEL'),(94,'MEM_READ'),(95,'MEM_WRITE'),
        (96,'ATLAS_INIT'),(97,'GRAM_EXEC'),(98,'GLYPH_DISP'),(99,'K_AYAB'),
        (100,'SHARD_LOAD'),(101,'SHARD_HASH'),(102,'SHARD_FETCH'),(103,'SHARD_POLL'),
        (104,'SCO_CALL'),(105,'SCO_RESOLVE'),(106,'SCO_AWAIT'),(107,'SCO_EMIT'),
        (108,'MN_SPAWN'),(109,'MN_LOAD'),(110,'MN_EXEC'),(111,'MN_STOP'),
        (112,'CMD_BUILD'),(113,'CMD_TEST'),(114,'CMD_DEPLOY'),(115,'CMD_COMMIT'),
        (116,'CMD_PUSH'),(117,'CMD_PULL'),(118,'CMD_BRANCH'),(119,'CMD_MERGE'),
        (120,'I_QUESTION'),(121,'I_ANSWER'),(122,'I_CODE'),(123,'I_EXPLAIN'),
        (124,'I_FIX'),(125,'I_CREATE'),(126,'I_UPDATE'),(127,'I_ANALYZE'),
    ]
    for tid, name in _known:
        _ID_TO_NAME[tid] = name

_build_id_map()


# ─── Quick test ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tok = GlyphTokenizer()

    text = "Hello world, this is a test of the glyph tokenizer."
    ids = tok.encode(text, max_len=32)
    print(f"encode:  {ids[:16]}...")
    print(f"decode:  {tok.decode(ids[:16])}")

    tool_ids = tok.encode_tool_call('Read', 'C:/Users/canna/.gpu_trainer/README.md')
    print(f"tool:    {tool_ids}")
    print(f"decoded: {tok.decode(tool_ids)}")

    nibble = 8  # EMIT
    tid = tok.nibble_to_token(nibble)
    print(f"nibble {nibble} -> token {tid} ({_ID_TO_NAME.get(tid, '?')})")
    print(f"SOFTMAX token: {SOFTMAX}, MN_EXEC token: {MN_EXEC}")
