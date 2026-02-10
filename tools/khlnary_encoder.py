"""Reference KHΛ-2-DENSE-32 encoder/decoder and tiny AST lowering pipeline.

This module provides:
- KNU packing/unpacking for the KHΛ-2-DENSE-32 v0.1 draft profile
- parity validation
- lowering for a compact Python subset including if/while/functions
- 128-bit lane-bundle packing helpers
"""

from __future__ import annotations

import ast
from typing import Dict, List, Tuple

VER = 0x1
AUTH_CLASS_USER = 0x1

GLYPH_IDS: Dict[str, int] = {
    "G_NOP": 0x00,
    "G_CONST_I8": 0x01,
    "G_ADD_I32": 0x02,
    "G_RET": 0x03,
    "G_IFZ_JUMP8": 0x10,
    "G_JUMP8": 0x11,
    "G_WHILE_HEAD": 0x12,
    "G_WHILE_TAIL": 0x13,
    "G_FUNC_DEF": 0x20,
    "G_FUNC_END": 0x21,
    "G_CALL": 0x22,
    "G_LOAD_LOCAL": 0x23,
    "G_STORE_LOCAL": 0x24,
    "G_EQ_I32": 0x25,
    "G_LT_I32": 0x26,
    "G_LOAD_BIN_TENSOR": 0x30,
}

GLYPH_BY_ID = {glyph_id: glyph_name for glyph_name, glyph_id in GLYPH_IDS.items()}

FLAG_IMMEDIATE = 0x1


class KhlNaryParityError(ValueError):
    """Raised when decoded KNU parity does not match computed parity."""


class KhlNaryLoweringError(ValueError):
    """Raised when the source program uses unsupported syntax."""


def parity_even_32(word_without_parity_bit: int) -> int:
    """Return parity bit such that total set bits across [31:0] are even."""

    bits_31_1 = (word_without_parity_bit & 0xFFFFFFFE) >> 1
    return bits_31_1.bit_count() & 0x1


def encode_knu(
    glyph_name: str,
    *,
    arity: int = 0,
    profile_flags: int = 0,
    payload: int = 0,
    auth_class: int = AUTH_CLASS_USER,
    ver: int = VER,
) -> int:
    """Encode one KHΛ-2-DENSE-32 KNU into a 32-bit integer."""

    if glyph_name not in GLYPH_IDS:
        raise KeyError(f"Unknown glyph: {glyph_name}")

    word = 0
    word |= (ver & 0xF) << 28
    word |= (GLYPH_IDS[glyph_name] & 0xFF) << 20
    word |= (arity & 0xF) << 16
    word |= (profile_flags & 0xF) << 12
    word |= (payload & 0xFF) << 4
    word |= (auth_class & 0x7) << 1
    word |= parity_even_32(word)
    return word


def decode_knu(word: int) -> Dict[str, int | str | None]:
    """Decode and validate a KHΛ-2-DENSE-32 KNU."""

    word = word & 0xFFFFFFFF
    parity = word & 0x1
    expected_parity = parity_even_32(word & ~0x1)
    if parity != expected_parity:
        raise KhlNaryParityError("Parity error in KNU")

    glyph_id = (word >> 20) & 0xFF
    return {
        "ver": (word >> 28) & 0xF,
        "glyph_id": glyph_id,
        "glyph_name": GLYPH_BY_ID.get(glyph_id),
        "arity": (word >> 16) & 0xF,
        "profile_flags": (word >> 12) & 0xF,
        "payload": (word >> 4) & 0xFF,
        "auth_class": (word >> 1) & 0x7,
    }


def _signed_to_u8(value: int) -> int:
    if not -128 <= value <= 127:
        raise KhlNaryLoweringError(f"Jump offset out of int8 range: {value}")
    return value & 0xFF


class ExtendedLower(ast.NodeVisitor):
    """Lower a tiny Python subset into glyph tuples.

    Supported constructs:
    - integer literals in [-128, 127]
    - names, assignment to names
    - `+`, `==`, `<`
    - if/else, while
    - function definitions, calls, returns
    - top-level expression statements (auto `G_RET`)
    """

    def __init__(self) -> None:
        self.glyphs: List[List[int | str]] = []
        self.function_ids: Dict[str, int] = {}
        self.next_function_id = 1
        self.locals_stack: List[Dict[str, int]] = []
        self.pending_calls: List[Tuple[int, str]] = []

    def emit(self, name: str, arity: int = 0, flags: int = 0, payload: int = 0) -> int:
        self.glyphs.append([name, arity, flags, payload])
        return len(self.glyphs) - 1

    def patch_jump(self, index: int, target_index: int) -> None:
        offset = target_index - index
        self.glyphs[index][3] = _signed_to_u8(offset)

    def _current_locals(self) -> Dict[str, int]:
        if not self.locals_stack:
            self.locals_stack.append({})
        return self.locals_stack[-1]

    def _local_slot(self, name: str) -> int:
        table = self._current_locals()
        if name not in table:
            table[name] = len(table)
        slot = table[name]
        if slot > 255:
            raise KhlNaryLoweringError("local slot exceeds 8-bit payload")
        return slot

    def _get_function_id(self, name: str) -> int:
        if name not in self.function_ids:
            self.function_ids[name] = self.next_function_id
            self.next_function_id += 1
        return self.function_ids[name]

    def visit_Module(self, node: ast.Module) -> None:  # noqa: N802
        self.locals_stack.append({})
        for stmt in node.body:
            self.visit(stmt)
        self.locals_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        func_id = self._get_function_id(node.name)
        self.emit("G_FUNC_DEF", arity=1, flags=FLAG_IMMEDIATE, payload=func_id)

        fn_locals: Dict[str, int] = {}
        for i, arg in enumerate(node.args.args):
            fn_locals[arg.arg] = i
        self.locals_stack.append(fn_locals)

        for stmt in node.body:
            self.visit(stmt)

        self.locals_stack.pop()
        self.emit("G_FUNC_END")

    def visit_Return(self, node: ast.Return) -> None:  # noqa: N802
        if node.value is None:
            self.emit("G_CONST_I8", flags=FLAG_IMMEDIATE, payload=0)
        else:
            self.visit(node.value)
        self.emit("G_RET", arity=1)

    def visit_Expr(self, node: ast.Expr) -> None:  # noqa: N802
        self.visit(node.value)
        self.emit("G_RET", arity=1)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise KhlNaryLoweringError("Only single-name assignment is supported")
        self.visit(node.value)
        slot = self._local_slot(node.targets[0].id)
        self.emit("G_STORE_LOCAL", arity=1, flags=FLAG_IMMEDIATE, payload=slot)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            slot = self._local_slot(node.id)
            self.emit("G_LOAD_LOCAL", flags=FLAG_IMMEDIATE, payload=slot)
            return
        raise KhlNaryLoweringError("Unsupported name context")

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if isinstance(node.value, int) and -128 <= node.value <= 127:
            self.emit("G_CONST_I8", flags=FLAG_IMMEDIATE, payload=node.value & 0xFF)
            return
        raise KhlNaryLoweringError("Only small integer constants are supported")

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802
        self.visit(node.left)
        self.visit(node.right)
        if isinstance(node.op, ast.Add):
            self.emit("G_ADD_I32", arity=2)
            return
        raise KhlNaryLoweringError("Only '+' binary operator is supported")

    def visit_Compare(self, node: ast.Compare) -> None:  # noqa: N802
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise KhlNaryLoweringError("Only single comparisons are supported")
        self.visit(node.left)
        self.visit(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.Eq):
            self.emit("G_EQ_I32", arity=2)
            return
        if isinstance(op, ast.Lt):
            self.emit("G_LT_I32", arity=2)
            return
        raise KhlNaryLoweringError("Only '==' and '<' comparisons are supported")

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        self.visit(node.test)
        jmp_false = self.emit("G_IFZ_JUMP8", arity=1, flags=FLAG_IMMEDIATE, payload=0)

        for stmt in node.body:
            self.visit(stmt)

        if node.orelse:
            jmp_end = self.emit("G_JUMP8", flags=FLAG_IMMEDIATE, payload=0)
            else_start = len(self.glyphs)
            self.patch_jump(jmp_false, else_start)
            for stmt in node.orelse:
                self.visit(stmt)
            self.patch_jump(jmp_end, len(self.glyphs))
        else:
            self.patch_jump(jmp_false, len(self.glyphs))

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        head = len(self.glyphs)
        self.emit("G_WHILE_HEAD")
        self.visit(node.test)
        jmp_exit = self.emit("G_IFZ_JUMP8", arity=1, flags=FLAG_IMMEDIATE, payload=0)

        for stmt in node.body:
            self.visit(stmt)

        back_jump = self.emit("G_JUMP8", flags=FLAG_IMMEDIATE, payload=0)
        self.patch_jump(back_jump, head)
        tail = len(self.glyphs)
        self.emit("G_WHILE_TAIL")
        self.patch_jump(jmp_exit, tail)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if not isinstance(node.func, ast.Name):
            raise KhlNaryLoweringError("Only simple function-name calls are supported")

        for arg in node.args:
            self.visit(arg)

        idx = self.emit("G_CALL", arity=len(node.args), flags=FLAG_IMMEDIATE, payload=0)
        self.pending_calls.append((idx, node.func.id))

    def finalize(self) -> None:
        for idx, fn_name in self.pending_calls:
            func_id = self._get_function_id(fn_name)
            self.glyphs[idx][3] = func_id


def compile_python_to_khlnary_words(src: str) -> List[int]:
    """Compile a compact Python subset source string to KHΛ-2-DENSE words."""

    tree = ast.parse(src)
    lower = ExtendedLower()
    lower.visit(tree)
    lower.finalize()

    words: List[int] = []
    for glyph_name, arity, flags, payload in lower.glyphs:
        words.append(
            encode_knu(
                str(glyph_name),
                arity=int(arity),
                profile_flags=int(flags),
                payload=int(payload),
                auth_class=AUTH_CLASS_USER,
                ver=VER,
            )
        )
    return words


def pack_lane_bundles(words: List[int]) -> List[List[int]]:
    """Pack words into 128-bit lane bundles (4x 32-bit KNUs, padded with NOP)."""

    bundles: List[List[int]] = []
    nop = encode_knu("G_NOP")
    for i in range(0, len(words), 4):
        chunk = words[i : i + 4]
        while len(chunk) < 4:
            chunk.append(nop)
        bundles.append(chunk)
    return bundles


def pack_lane_bundle_u128(bundle_words: List[int]) -> int:
    """Pack 4 words into one big-endian 128-bit lane integer."""

    if len(bundle_words) != 4:
        raise ValueError("bundle_words must contain exactly 4 KNU words")

    return (
        ((bundle_words[0] & 0xFFFFFFFF) << 96)
        | ((bundle_words[1] & 0xFFFFFFFF) << 64)
        | ((bundle_words[2] & 0xFFFFFFFF) << 32)
        | (bundle_words[3] & 0xFFFFFFFF)
    )


def compile_to_knu(src: str) -> List[int]:
    """Alias for compile_python_to_khlnary_words used by lowering skeletons."""

    return compile_python_to_khlnary_words(src)


__all__ = [
    "VER",
    "AUTH_CLASS_USER",
    "GLYPH_IDS",
    "FLAG_IMMEDIATE",
    "KhlNaryParityError",
    "KhlNaryLoweringError",
    "parity_even_32",
    "encode_knu",
    "decode_knu",
    "compile_python_to_khlnary_words",
    "compile_to_knu",
    "pack_lane_bundles",
    "pack_lane_bundle_u128",
]
