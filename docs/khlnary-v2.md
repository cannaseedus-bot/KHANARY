# KHΛNARY v0.2 Specification
**Status:** Draft  
**Audience:** Runtime implementers, compiler authors, substrate architects

---

## 1. Overview

**KHΛNARY** is a multi‑alphabet execution substrate that encodes **KUHUL semantic glyphs** into fixed‑width words suitable for deterministic replay over binary, ternary, and quaternary substrates.

v0.2 extends v0.1 with:

- **.bin/.stb tensor integration**
- **explicit control‑flow + function glyphs**
- **binary profile `KHΛ‑2‑DENSE‑32` as the canonical baseline**

KHΛNARY **does not define meaning**; it defines **representation**.  
Meaning is defined by KUHUL; KHΛNARY is the encoding law.

---

## 2. Profiles and word layout

### 2.1 Profile ID

- **Profile name:** `KHΛ-2-DENSE-32`
- **Substrate:** binary (`Σ₂ = {0,1}`)
- **Word size:** 32 bits
- **Endianness:** big‑endian for bit numbering; storage endianness is implementation‑defined but MUST be consistent within a build.

### 2.2 KNU‑32 layout

A **KHΛNARY Unit (KNU)** is a single 32‑bit word:

| Bits   | Width | Field         | Description                                  |
|--------|--------|---------------|----------------------------------------------|
| 31–28  | 4      | `VER`         | KHΛNARY spec version (0–15)                  |
| 27–20  | 8      | `GLYPH_ID`    | KUHUL glyph identifier                       |
| 19–16  | 4      | `ARITY`       | Operand count (0–15)                         |
| 15–12  | 4      | `FLAGS`       | Profile‑specific flags                       |
| 11–4   | 8      | `PAYLOAD`     | Small immediate / descriptor byte            |
| 3–1    | 3      | `AUTH_CLASS`  | Authority class (π/tenant/role)             |
| 0      | 1      | `PARITY`      | Even parity over bits 31–1                   |

**Invariants:**

- `VER` for this spec: **`0x2`** (KHΛNARY v0.2).
- `PARITY` is chosen so that the total number of `1` bits in `[31:0]` is **even**.
- For any valid KNU, decoding MUST yield either:
  - a valid KUHUL glyph + metadata, or  
  - a typed KHΛNARY error (never silent corruption).

---

## 3. Glyph set (v0.2 core)

### 3.1 Common flags

- **`FLAGS` bit usage (for `KHΛ-2-DENSE-32`):**

| Bit | Mask | Meaning                         |
|-----|------|---------------------------------|
| 0   | 0x1  | `IMM` – `PAYLOAD` is immediate |
| 1   | 0x2  | `BIN_REF` – references .bin/.stb |
| 2   | 0x4  | `SHAPE_DESC` – payload indexes shape table |
| 3   | 0x8  | Reserved                        |

---

### 3.2 Arithmetic and stack glyphs

| Name          | `GLYPH_ID` | `ARITY` | Flags usage         | Semantics (KUHUL)                          |
|---------------|-----------:|--------:|---------------------|--------------------------------------------|
| `G_NOP`       | `0x00`     | 0       | 0                   | No operation                               |
| `G_CONST_I8`  | `0x01`     | 0       | `IMM`               | Push 8‑bit signed int from `PAYLOAD`       |
| `G_ADD_I32`   | `0x02`     | 2       | 0                   | Add top two stack i32                      |
| `G_RET`       | `0x03`     | 1       | 0                   | Return top of stack from current frame     |

---

### 3.3 Control‑flow glyphs

| Name           | `GLYPH_ID` | `ARITY` | Flags usage | Semantics |
|----------------|-----------:|--------:|-------------|-----------|
| `G_IFZ_JUMP8`  | `0x10`     | 1       | `IMM`       | Pop top; if zero, jump by signed 8‑bit offset in `PAYLOAD` (KNU‑relative) |
| `G_JUMP8`      | `0x11`     | 0       | `IMM`       | Unconditional jump by signed 8‑bit offset in `PAYLOAD` (KNU‑relative) |
| `G_WHILE_HEAD` | `0x12`     | 0       | 0           | Loop head marker (for tooling/debug; runtime may treat as NOP) |
| `G_WHILE_TAIL` | `0x13`     | 0       | 0           | Loop tail marker (for tooling/debug; runtime may treat as NOP) |

**Offset semantics:**

- Offsets are **signed 8‑bit** (`−128…+127`).
- Offset is applied relative to the **index of the current KNU** (implementation MUST define whether it is from current or next; recommended: `PC ← PC + offset` where `PC` points at current KNU).

---

### 3.4 Function glyphs

| Name         | `GLYPH_ID` | `ARITY` | Flags usage | Semantics |
|--------------|-----------:|--------:|-------------|-----------|
| `G_FUNC_DEF` | `0x20`     | 1       | `IMM`       | Begin function with ID = `PAYLOAD` (0–255); pushes new frame context |
| `G_FUNC_END` | `0x21`     | 0       | 0           | End current function; implementation‑defined whether implicit `RET` |
| `G_CALL`     | `0x22`     | 1       | `IMM`       | Call function with ID = `PAYLOAD`; arguments taken from stack |

**Function ID space:**

- `PAYLOAD` is a **function ID** in `[0,255]`.
- Mapping from function ID → entry PC is defined by the **module metadata** (not KHΛNARY itself).

---

## 4. .bin / .stb integration glyphs

These glyphs bridge KHΛNARY to the **binary tensor substrate** (`.stb` files).

### 4.1 External tables

Implementations MUST provide (per module):

- **Bin file table**: index → `{path, size, alignment, dtype}`
- **Shape table**: index → `{rank, dims[], layout}`

Indices are 8‑bit (`0–255`) and referenced via `PAYLOAD` or extended descriptors.

---

### 4.2 `G_LOAD_BIN_TENSOR` (`0x30`)

**Purpose:** Bind a tensor handle to a region in a `.stb` file.

- **`GLYPH_ID`**: `0x30`
- **`ARITY`**: 0 (pushes a tensor handle onto stack or into a register, depending on KUHUL convention)
- **`FLAGS`**:
  - `BIN_REF` MUST be set
  - `SHAPE_DESC` MAY be set
- **`PAYLOAD` layout (KHΛ‑2‑DENSE‑32):**

  Single‑byte descriptor:

  - bits 7–4: `bin_file_id` (0–15)  
  - bits 3–0: `shape_id` (0–15)

  For larger spaces, implementations MAY extend via **side tables** keyed by KNU index; v0.2 keeps it small and deterministic.

**Semantics (high‑level):**

- Resolve `bin_file_id` in the bin file table.
- Resolve `shape_id` in the shape table.
- Compute base pointer = `file_base + offset` (offset is defined in the bin metadata for that tensor).
- Push a **tensor handle** (implementation‑defined) into the KUHUL execution context.

---

### 4.3 `G_MMAP_BIN_REGION` (`0x31`)

**Purpose:** Ensure a `.stb` region is memory‑mapped and ready for device access.

- **`GLYPH_ID`**: `0x31`
- **`ARITY`**: 0 or 1 (depending on whether it consumes a file/tensor handle)
- **`FLAGS`**:
  - `BIN_REF` MUST be set
- **`PAYLOAD`**: `bin_file_id` (0–255) or small handle index.

**Semantics:**

- For CPU targets: `mmap` or equivalent.
- For GPU targets: pre‑stage into device memory or use GPU‑direct if available.
- May be a no‑op if already mapped.

---

### 4.4 `G_PREFETCH_BIN` (`0x32`)

**Purpose:** Hint to prefetch a `.stb` region into cache or device memory.

- **`GLYPH_ID`**: `0x32`
- **`ARITY`**: 0
- **`FLAGS`**:
  - `BIN_REF` MUST be set
- **`PAYLOAD`**: `bin_file_id` (0–255) or region index.

**Semantics:**

- Non‑blocking hint; MUST NOT change program semantics.
- Implementations MAY ignore on constrained targets.

---

## 5. Authority model

### 5.1 `AUTH_CLASS` field

3‑bit authority class:

| Value | Meaning (example)          |
|-------|----------------------------|
| `0`   | reserved / invalid         |
| `1`   | user code                  |
| `2`   | system/runtime             |
| `3`   | privileged I/O             |
| `4–7` | implementation‑specific    |

**Invariant:**

- A runtime MUST NOT execute a glyph whose **effective authority** exceeds the current execution context.
- `.bin` glyphs (`0x30–0x32`) SHOULD require at least `AUTH_CLASS ≥ 1`, and memory‑mapping operations MAY require higher classes depending on deployment.

---

## 6. Parity and validation

### 6.1 Parity computation

Given a 32‑bit word `W`:

- Let `W' = W & ~0x1` (clear bit 0).
- Let `ones = popcount(W' >> 1)` (bits 31–1).
- `PARITY` bit (bit 0) MUST be set to `ones & 1`.

Validation:

- On decode, recompute parity; if mismatch, raise a **KHΛNARY parity error** and treat the KNU as invalid.

---

## 7. Module and replay invariants

- A **KHΛNARY module** is an ordered sequence of KNUs plus:
  - function table (ID → PC)
  - bin file table
  - shape table
  - optional metadata (debug, symbols, etc.)

**Replay law:**

- A module is replay‑valid iff:
  - `VER == 0x2` or explicitly up‑migrated
  - all KNUs pass parity and authority checks
  - all referenced `bin_file_id` and `shape_id` entries exist
  - control‑flow offsets remain within module bounds

---

## 8. Versioning

- **Spec ID:** `KHΛNARY-2`
- **Binary profile:** `KHΛ-2-DENSE-32`
- **Breaking changes** (require new `VER`):
  - KNU layout changes
  - glyph ID reassignments
  - semantics changes for existing glyphs
- **Non‑breaking changes:**
  - adding new glyph IDs
  - adding new profiles (e.g., `KHΛ-3-BRANCH`, `KHΛ-4-VECTOR`)
  - extending external tables (bin/shape metadata)

---

If you want, next step I can draft the **`.stb` binary format spec** to sit alongside this in the same repo (`docs/khlnary-v2.md` + `docs/stb-format.md`) and wire the glyphs to concrete header fields.
