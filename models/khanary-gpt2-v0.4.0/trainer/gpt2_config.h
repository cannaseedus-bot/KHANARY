#pragma once
#include <cstdint>
#include <cmath>

// DistilGPT2 shape (what cli_coder_gpt2 is trained from).
// Full GPT-2 117M uses n_layer=12; override via GPT2Config::from_json().
struct GPT2Config {
    uint32_t vocab_size = 50260;   // 50257 base + 3 special tokens added during fine-tune
    uint32_t n_ctx      = 1024;
    uint32_t n_embd     = 768;
    uint32_t n_head     = 12;
    uint32_t n_layer    = 6;       // DistilGPT2 = 6 layers (GPT-2 117M = 12)
    uint32_t d_head     = 64;      // n_embd / n_head
    uint32_t d_ff       = 3072;    // 4 * n_embd
    float    attn_scale = 0.125f;  // 1/sqrt(64)
};
