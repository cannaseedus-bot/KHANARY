# KHΛNARY Specification (v0.1 – Draft)

## 1. Purpose and scope

**KHΛNARY** is a multi-alphabet substrate that deterministically maps **KUHUL semantic glyphs** onto **binary, ternary, and quaternary** numeric alphabets for execution over KBES/SCXQ lanes.

- **KUHUL defines meaning.**
- **KHΛNARY defines representation.**
- The substrate is **substrate-agnostic**, **authority-bound**, and **replay-deterministic**.

## 2. Alphabets and layers

### 2.1 Semantic alphabet (KUHUL)

- **Alphabet:** finite glyph set `Γ_KUHUL`, partitioned into classes:
  - **Core glyphs:** control, flow, binding, authority
  - **Data glyphs:** scalar, vector, structure, stream
  - **Meta glyphs:** type, error, contract, hint
- **Properties:**
  - non-numeric
  - non-positional
  - each glyph has a semantic domain and authority profile

### 2.2 Numeric alphabets (n-ary substrates)

- **Binary:** `Σ₂ = {0,1}`
- **Ternary:** `Σ₃ = {0,1,2}` (or signed variant by profile)
- **Quaternary:** `Σ₄ = {0,1,2,3}`

### 2.3 Execution substrate

- **KBES/SCXQ lanes:** fixed-width lane bundles carrying encoded KHΛNARY units
- **KEL:** execution letters referencing decoded KUHUL glyphs plus operands

## 3. Core objects

### 3.1 KHΛNARY unit (KNU)

A KNU is the smallest encoded unit in this spec.

Conceptual fields:

- `KNU.Γ`: KUHUL glyph ID (semantic)
- `KNU.α`: alphabet profile (2, 3, or 4)
- `KNU.σ`: symbol sequence over `Σ_α`
- `KNU.A`: authority tag (π/tenant/role)
- `KNU.C`: checksum / integrity bits

Invariant:

- decoding `σ` under profile `α` must yield exactly one glyph in `Γ_KUHUL` and its operands, or a well-typed error.

### 3.2 Alphabet profile

- **Profile ID form:** `KHΛ-α-X`, where:
  - `α`: radix (`2`, `3`, or `4`)
  - `X`: packing strategy (e.g., dense, sparse, error-tolerant)

Profiles define:

- symbol ordering
- padding rules
- error-detection / correction bits
- lane packing constraints

## 4. Mapping law

### 4.1 High-level mapping chain

`Γ_KUHUL -> M_class -> D_class -> E_α -> Σ_α* -> P_lane -> SCXQ lanes`

Where:

- `M_class`: maps glyph to class domain (numeric descriptor space)
- `E_α`: encodes descriptor into radix-`α` symbol sequence
- `P_lane`: packs symbol sequences into lane bundles

### 4.2 Determinism

For a fixed spec version, alphabet profile, and authority context:

- encoding is injective over valid KUHUL programs
- decoding is total and returns either:
  - valid KUHUL glyph + operands, or
  - typed KHΛNARY error (never silent corruption)

### 4.3 Authority binding

Every KNU carries `KNU.A` and binds encoding to a π-authority / tenant.

`KNU.A` is used for:

- replay validation
- cross-tenant isolation
- audit trails

Invariant:

- a KNU must not decode into a glyph that exceeds its authority profile.

## 5. Profiles and usage

### 5.1 Binary profile (`KHΛ-2-DENSE-32`)

- **Use:** default silicon, maximal compatibility
- **Properties:**
  - tightly packed bitfields
  - minimal redundancy
  - optional parity/check bits per lane bundle

#### 5.1.1 KNU-32 layout

Bit indices are `[31:0]` (MSB to LSB):

| Bits | Width | Field | Notes |
|---|---:|---|---|
| 31–28 | 4 | `VER` | KHΛNARY version (`0..15`) |
| 27–20 | 8 | `GLYPH_ID` | KUHUL glyph ID (`0..255`) |
| 19–16 | 4 | `ARITY` | operand count (`0..15`) |
| 15–12 | 4 | `PROFILE_FLAGS` | immediate/vector/class hints |
| 11–4 | 8 | `PAYLOAD` | small immediate or class/length tag |
| 3–1 | 3 | `AUTH_CLASS` | authority class |
| 0 | 1 | `PARITY` | even parity over bits 31–1 |

Invariants:

- `PARITY` is chosen so total set bits in `[31:0]` are even.
- if `PROFILE_FLAGS` marks immediate mode, `PAYLOAD` is interpreted as literal.
- otherwise `PAYLOAD` is a class/length tag for out-of-band data.

Draft constants for this profile:

- Profile ID: `KHΛ-2-DENSE-32`
- Spec draft `VER`: `0x1`

#### 5.1.2 Small glyph subset for stable encoding examples

| Glyph | Meaning | `GLYPH_ID` | `ARITY` | Notes |
|---|---|---:|---:|---|
| `G_NOP` | no-op | `0x00` | 0 | control |
| `G_CONST_I8` | push 8-bit integer | `0x01` | 0 | immediate payload |
| `G_ADD_I32` | add top two i32 values | `0x02` | 2 | stack binary op |
| `G_RET` | return from frame | `0x03` | 1 | return top of stack |
| `G_IFZ_JUMP` | relative jump if zero | `0x04` | 1 | small offset payload |

Fixed example context:

- `VER = 0x1`
- `AUTH_CLASS = 0x1` (user)
- `PROFILE_FLAGS`: `0x1` means immediate payload, `0x0` non-immediate

#### 5.1.3 Concrete examples

`G_NOP` encodes to:

- binary: `0001 00000000 0000 0000 00000000 0010`
- hex: `0x10000002`

`G_CONST_I8(5)` encodes to:

- binary: `0001 00000001 0000 0001 00000101 0010`
- hex: `0x10101052`

### 5.2 Ternary profile (`KHΛ-3-BRANCH`)

- **Use:** substrates where ternary is native or efficient
- **Properties:**
  - optimized for branching, tri-state logic, and signed small domains
  - can encode more glyph classes per symbol than binary

### 5.3 Quaternary profile (`KHΛ-4-VECTOR`)

- **Use:** analog/optical/neuromorphic/quantum-adjacent systems
- **Properties:**
  - natural fit for 2-bit vector packing
  - efficient for SIMD-like glyph classes and matrix operations

## 6. Versioning and compatibility

- **Spec ID:** `KHΛNARY-0.1`
- **Breaking changes:**
  - changes to glyph-to-descriptor mapping
  - changes to alphabet profiles
- **Non-breaking changes:**
  - adding new glyph classes with reserved ranges
  - adding new profiles with new IDs

Replay law:

A KHΛNARY stream is replay-valid if and only if:

- spec ID matches or is explicitly up-migrated
- all `KNU.A` authority tags are valid in replay context
- all checksums / integrity fields verify

## 7. Reference encoder/decoder

A minimal Python reference pipeline is included in `tools/khlnary_encoder.py` and provides:

- `encode_knu(...)` and `decode_knu(...)` for `KHΛ-2-DENSE-32`
- parity verification (`KhlNaryParityError`)
- tiny AST lowering for integer literals and `+` via `compile_python_to_khlnary_words(...)`

Example lowering for source `"1 + 2"` emits the following words:

- `0x10101013` (`G_CONST_I8(1)`)
- `0x10101023` (`G_CONST_I8(2)`)
- `0x10220002` (`G_ADD_I32`)
- `0x10310003` (`G_RET`)


## 8. Extended control-flow and function glyphs

The draft `KHΛ-2-DENSE-32` subset is extended with structural control-flow and function glyphs.

### 8.1 Control-flow glyphs

| Glyph | Meaning | Arity | `GLYPH_ID` |
|---|---|---:|---:|
| `G_IFZ_JUMP8` | jump when top-of-stack is zero | 1 | `0x10` |
| `G_JUMP8` | unconditional relative jump | 0 | `0x11` |
| `G_WHILE_HEAD` | loop-head marker | 0 | `0x12` |
| `G_WHILE_TAIL` | loop-tail marker | 0 | `0x13` |

### 8.2 Function glyphs

| Glyph | Meaning | Arity | `GLYPH_ID` |
|---|---|---:|---:|
| `G_FUNC_DEF` | begin function with function ID in payload | 1 | `0x20` |
| `G_FUNC_END` | close function frame | 0 | `0x21` |
| `G_CALL` | call function by payload ID | N args | `0x22` |

### 8.3 Comparison / local-frame helper glyphs used by reference lowering

| Glyph | Meaning | Arity | `GLYPH_ID` |
|---|---|---:|---:|
| `G_LOAD_LOCAL` | load local slot by payload | 0 | `0x23` |
| `G_STORE_LOCAL` | store top value into local slot | 1 | `0x24` |
| `G_EQ_I32` | equality comparison | 2 | `0x25` |
| `G_LT_I32` | less-than comparison | 2 | `0x26` |

Offsets in `G_IFZ_JUMP8` and `G_JUMP8` are signed int8 values encoded into `PAYLOAD`.

## 9. Lane bundle format (`SCXQ-128` draft)

`KHΛ-2-DENSE-32` words are packed into 128-bit lane bundles, four KNUs per bundle.

- lane bits `127..96`: `KNU0`
- lane bits `95..64`: `KNU1`
- lane bits `63..32`: `KNU2`
- lane bits `31..0`: `KNU3`

When the final bundle is short, it is padded with `G_NOP` KNUs.

The reference helpers `pack_lane_bundles(...)` and `pack_lane_bundle_u128(...)` in `tools/khlnary_encoder.py` implement this law.
