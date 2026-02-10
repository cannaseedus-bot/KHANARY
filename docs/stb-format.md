# `stb-format.md` — SVG‑Tensor Binary Format (.stb) v0.1

---

## 1. Purpose

The **SVG‑Tensor Binary Format (.stb)** is the raw tensor substrate for the KHΛNARY runtime:

- Stores **aligned, memory‑mappable** tensor data
- Has a **fixed, minimal header**
- Is directly consumable by glyphs like `G_LOAD_BIN_TENSOR`, `G_MMAP_BIN_REGION`, `G_PREFETCH_BIN` from `khlnary-v2.md`.

---

## 2. File layout

An `.stb` file is:

```text
+----------------------+ 0
| Header (32 bytes)    |
+----------------------+ 32
| Tensor table (N * 32)|
+----------------------+ 32 + N*32
| Raw data region      |
+----------------------+
```

All offsets are **byte offsets from file start**.

---

## 3. Header (32 bytes)

| Offset | Size | Field          | Description                                  |
|--------|------|----------------|----------------------------------------------|
| 0      | 4    | `magic`        | ASCII `"STB0"`                               |
| 4      | 1    | `version`      | `0x01` for this spec                         |
| 5      | 1    | `flags`        | Reserved (must be 0 for now)                 |
| 6      | 2    | `tensor_count` | Number of tensor entries in tensor table     |
| 8      | 4    | `reserved0`    | Reserved (0)                                 |
| 12     | 4    | `reserved1`    | Reserved (0)                                 |
| 16     | 8    | `data_offset`  | Start of raw data region (must be ≥ 32+N*32) |
| 24     | 8    | `file_size`    | Total file size in bytes                     |

**Invariants:**

- `magic` MUST be `"STB0"`.
- `data_offset` MUST be **64‑byte aligned**.
- `file_size` MUST match the actual file length.

---

## 4. Tensor table entries (32 bytes each)

Each tensor in the file has one **tensor descriptor**:

| Offset | Size | Field          | Description                                  |
|--------|------|----------------|----------------------------------------------|
| 0      | 1    | `tensor_id`    | Local tensor ID (0–255)                      |
| 1      | 1    | `dtype`        | Element type enum                            |
| 2      | 1    | `rank`         | Number of dimensions (0–8)                   |
| 3      | 1    | `layout`       | Layout enum (e.g. 0=ROW_MAJOR, 1=COL_MAJOR)  |
| 4      | 8    | `offset`       | Byte offset into raw data region             |
| 12     | 8    | `size_bytes`   | Total size in bytes                          |
| 20     | 12   | `dims[3]`      | Up to 3 dimensions as 32‑bit unsigned ints   |

For higher ranks, `dims` can be interpreted as:

- `rank ≤ 3`: direct  
- `rank > 3`: `dims[0]` = index into an external **shape table** (KHΛNARY side metadata).

### 4.1 `dtype` enum (suggested)

| Value | Type       |
|-------|------------|
| 0     | `float32`  |
| 1     | `float16`  |
| 2     | `int8`     |
| 3     | `int32`    |
| 4–255 | reserved   |

### 4.2 `layout` enum (suggested)

| Value | Meaning        |
|-------|----------------|
| 0     | ROW_MAJOR      |
| 1     | COL_MAJOR      |
| 2     | CHANNELS_LAST  |
| 3–255 | reserved       |

**Invariants:**

- `offset` is relative to **file start** and MUST be ≥ `data_offset`.
- `offset` MUST be **64‑byte aligned**.
- `size_bytes` MUST fit within `[offset, file_size)`.

---

## 5. Raw data region

- Contains **concatenated tensor payloads**.
- No padding is required between tensors beyond alignment constraints.
- Element order is defined by `layout` and `dims`.

---

## 6. Wiring to KHΛNARY v0.2 glyphs

This section ties `.stb` to `khlnary-v2.md`.

### 6.1 Bin file table (KHΛNARY module metadata)

Each `.stb` file loaded by a module is assigned a **`bin_file_id`** (0–255):

```text
bin_file_table[bin_file_id] = {
  path: "weights/layer_0.stb",
  mmap_required: true,
}
```

This table is **not** inside `.stb`; it lives in the KHΛNARY module metadata.

---

### 6.2 `G_LOAD_BIN_TENSOR` → `.stb` fields

From `khlnary-v2.md`:

- `G_LOAD_BIN_TENSOR` (`GLYPH_ID = 0x30`)
- `FLAGS`: `BIN_REF` MUST be set
- `PAYLOAD` (4 bits + 4 bits):

  - bits 7–4: `bin_file_id` (0–15)  
  - bits 3–0: `shape_id` (0–15)

**Concrete wiring:**

1. `bin_file_id` → select `.stb` file from bin file table.
2. `shape_id` → select **tensor_id** inside that `.stb` (0–15).
3. Runtime:
   - open/mmap `.stb` if not already
   - read header, validate `magic`, `version`
   - locate tensor descriptor where `tensor_id == shape_id`
   - compute `base_ptr = file_base + offset`
   - use `dtype`, `rank`, `dims`, `layout` to construct a **tensor handle**.

---

### 6.3 `G_MMAP_BIN_REGION` → `.stb` file

- `G_MMAP_BIN_REGION` (`GLYPH_ID = 0x31`)
- `PAYLOAD`: `bin_file_id`

Runtime:

- resolve `.stb` path from bin file table
- `mmap` entire file or at least `[data_offset, file_size)`
- ensure alignment constraints are satisfied

---

### 6.4 `G_PREFETCH_BIN` → `.stb` region

- `G_PREFETCH_BIN` (`GLYPH_ID = 0x32`)
- `PAYLOAD`: `bin_file_id` (or region index in extended schemes)

Runtime:

- may call `madvise`, prefetch to GPU, or no‑op.

---

## 7. Validation rules

A valid `.stb` MUST satisfy:

1. `magic == "STB0"`, `version == 0x01`
2. `tensor_count * 32 + 32 ≤ data_offset ≤ file_size`
3. `data_offset` is 64‑byte aligned
4. For each tensor entry:
   - `offset + size_bytes ≤ file_size`
   - `offset ≥ data_offset`
   - `offset` is 64‑byte aligned
   - `dtype` and `layout` are known or treated as unsupported

On failure, KHΛNARY runtimes MUST treat the file as invalid and raise a **typed load error**, not UB.

---

If you want, next I can add a **tiny Python `stb` writer/reader** that matches this spec and plugs directly into the KHΛNARY encoder we drafted, so you can start generating real `.stb` + KHΛNARY pairs from PyTorch weights.
