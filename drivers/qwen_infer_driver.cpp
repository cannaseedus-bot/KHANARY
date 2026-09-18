//
// qwen_infer_driver.cpp — Qwen3-1.7B D3D11 cs_5_0 inference backend
//
// Architecture (Qwen3-1.7B from GGUF):
//   28 layers, embed=2048, heads=16, kv_heads=8 (GQA), head_dim=128,
//   ff_dim=6144, vocab=151936, rope_theta=1e6, RMSNorm, SiLU, QK-norm per head
//
// Design:
//   - Q8_0 weights loaded into GPU ByteAddressBuffers at model-load time
//     (1.4 GB total, fits within HD 4600 1792 MB ceiling)
//   - GEMV shader dequantizes Q8_0 on-the-fly; no per-token weight uploads
//   - Hidden state and KV cache (float32) are GPU-resident throughout
//   - Embedding lookup and lm_head done on CPU (vocab 151936 too large for VRAM)
//   - QK-norm: CPU readback of Q/K, normalize, re-upload (tiny 24 KB per layer)
//

#define QWEN_INFER_EXPORTS
#include "qwen_infer_driver.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <fstream>
#include <limits>
#include <cstdarg>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

// ============================================================================
// HLSL shaders — cs_5_0
// ============================================================================

// GEMV with inline Q8_0 dequantisation.
// Q8_0 block: [uint16 f16-scale, int8[32] quants] = 34 bytes per 32 elements.
// Blocks cycle between 0 and 2 mod 4 alignment (34 % 4 == 2), so each f16
// scale always fits within a single 4-byte Load word.
// One group per output row; all 64 threads handle different K blocks of the same row.
// Adjacent threads access adjacent 34-byte blocks → ~contiguous memory reads, no stride.
// For K_blocks <= 64: each thread handles <=1 block.
// For K_blocks > 64: stride loop (e.g. K=6144: 3 blocks per thread).
static const char* HLSL_GEMV_Q8 = R"(
Buffer<uint>    W : register(t0); // Q8_0 weights as R32_UINT uints
Buffer<float>   x : register(t1); // input [K]
RWBuffer<float> y : register(u0); // output [N]
cbuffer CB0 : register(b0) { uint N; uint K; uint K_blocks; uint pad; };
groupshared float gs[64];

float dq(uint gb, uint e) {
    uint f16 = (W[gb >> 2u] >> ((gb & 2u) << 3u)) & 0xFFFFu;
    float sc = f16tof32(f16);
    uint qb = gb + 2u + e;
    int  qi = (int)((W[qb >> 2u] >> ((qb & 3u) << 3u)) & 0xFFu);
    if (qi > 127) qi -= 256;
    return (float)qi * sc;
}

[numthreads(64,1,1)]
void CSMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= N) { gs[tid.x] = 0.0f; GroupMemoryBarrierWithGroupSync(); return; }
    float acc = 0.0f;
    for (uint b = tid.x; b < K_blocks; b += 64u) {
        uint gb  = (row * K_blocks + b) * 34u;
        uint col = b * 32u;
        for (uint e = 0u; e < 32u; e++) acc += dq(gb, e) * x[col + e];
    }
    gs[tid.x] = acc;
    GroupMemoryBarrierWithGroupSync();
    if (tid.x < 32) gs[tid.x] += gs[tid.x + 32]; GroupMemoryBarrierWithGroupSync();
    if (tid.x < 16) gs[tid.x] += gs[tid.x + 16]; GroupMemoryBarrierWithGroupSync();
    if (tid.x <  8) gs[tid.x] += gs[tid.x +  8]; GroupMemoryBarrierWithGroupSync();
    if (tid.x <  4) gs[tid.x] += gs[tid.x +  4]; GroupMemoryBarrierWithGroupSync();
    if (tid.x <  2) gs[tid.x] += gs[tid.x +  2]; GroupMemoryBarrierWithGroupSync();
    if (tid.x <  1) gs[tid.x] += gs[tid.x +  1]; GroupMemoryBarrierWithGroupSync();
    if (tid.x == 0) y[row] = gs[0];
}
)";

// Fused QKV GEMV: Q (nH*hD rows from Wq), K (nKV*hD rows from Wk), V (nKV*hD rows from Wv)
// in one dispatch — saves 2 dispatch calls per layer (56 total, ~168ms on HD 4600).
// All groups with gid.x < N_q use Wq→yQ; next N_kv groups use Wk→yK; rest use Wv→yV.
// All 64 threads in a group take the same branch (condition on gid.x, not tid.x).
static const char* HLSL_GEMV_Q8_QKV = R"(
Buffer<uint>    Wq : register(t0);
Buffer<uint>    Wk : register(t1);
Buffer<uint>    Wv : register(t2);
Buffer<float>   x  : register(t3);
RWBuffer<float> yQ : register(u0);
RWBuffer<float> yK : register(u1);
RWBuffer<float> yV : register(u2);
cbuffer CB0 : register(b0) { uint N_q; uint N_kv; uint K_blocks; uint pad; };
groupshared float gs[64];

float dq_q(uint gb, uint e) {
    uint f16 = (Wq[gb >> 2u] >> ((gb & 2u) << 3u)) & 0xFFFFu;
    float sc = f16tof32(f16);
    uint qb = gb + 2u + e;
    int  qi = (int)((Wq[qb >> 2u] >> ((qb & 3u) << 3u)) & 0xFFu);
    if (qi > 127) qi -= 256;
    return (float)qi * sc;
}
float dq_k(uint gb, uint e) {
    uint f16 = (Wk[gb >> 2u] >> ((gb & 2u) << 3u)) & 0xFFFFu;
    float sc = f16tof32(f16);
    uint qb = gb + 2u + e;
    int  qi = (int)((Wk[qb >> 2u] >> ((qb & 3u) << 3u)) & 0xFFu);
    if (qi > 127) qi -= 256;
    return (float)qi * sc;
}
float dq_v(uint gb, uint e) {
    uint f16 = (Wv[gb >> 2u] >> ((gb & 2u) << 3u)) & 0xFFFFu;
    float sc = f16tof32(f16);
    uint qb = gb + 2u + e;
    int  qi = (int)((Wv[qb >> 2u] >> ((qb & 3u) << 3u)) & 0xFFu);
    if (qi > 127) qi -= 256;
    return (float)qi * sc;
}

[numthreads(64,1,1)]
void CSMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint gr = gid.x;
    float acc = 0.0f;
    if (gr < N_q) {
        uint row = gr;
        for (uint b = tid.x; b < K_blocks; b += 64u) {
            uint gb = (row * K_blocks + b) * 34u;
            uint col = b * 32u;
            for (uint e = 0u; e < 32u; e++) acc += dq_q(gb, e) * x[col + e];
        }
        gs[tid.x] = acc;
        GroupMemoryBarrierWithGroupSync();
        if (tid.x < 32) gs[tid.x] += gs[tid.x+32]; GroupMemoryBarrierWithGroupSync();
        if (tid.x < 16) gs[tid.x] += gs[tid.x+16]; GroupMemoryBarrierWithGroupSync();
        if (tid.x <  8) gs[tid.x] += gs[tid.x+ 8]; GroupMemoryBarrierWithGroupSync();
        if (tid.x <  4) gs[tid.x] += gs[tid.x+ 4]; GroupMemoryBarrierWithGroupSync();
        if (tid.x <  2) gs[tid.x] += gs[tid.x+ 2]; GroupMemoryBarrierWithGroupSync();
        if (tid.x <  1) gs[tid.x] += gs[tid.x+ 1]; GroupMemoryBarrierWithGroupSync();
        if (tid.x == 0) yQ[row] = gs[0];
    } else if (gr < N_q + N_kv) {
        uint row = gr - N_q;
        for (uint b = tid.x; b < K_blocks; b += 64u) {
            uint gb = (row * K_blocks + b) * 34u;
            uint col = b * 32u;
            for (uint e = 0u; e < 32u; e++) acc += dq_k(gb, e) * x[col + e];
        }
        gs[tid.x] = acc;
        GroupMemoryBarrierWithGroupSync();
        if (tid.x < 32) gs[tid.x] += gs[tid.x+32]; GroupMemoryBarrierWithGroupSync();
        if (tid.x < 16) gs[tid.x] += gs[tid.x+16]; GroupMemoryBarrierWithGroupSync();
        if (tid.x <  8) gs[tid.x] += gs[tid.x+ 8]; GroupMemoryBarrierWithGroupSync();
        if (tid.x <  4) gs[tid.x] += gs[tid.x+ 4]; GroupMemoryBarrierWithGroupSync();
        if (tid.x <  2) gs[tid.x] += gs[tid.x+ 2]; GroupMemoryBarrierWithGroupSync();
        if (tid.x <  1) gs[tid.x] += gs[tid.x+ 1]; GroupMemoryBarrierWithGroupSync();
        if (tid.x == 0) yK[row] = gs[0];
    } else {
        uint row = gr - N_q - N_kv;
        for (uint b = tid.x; b < K_blocks; b += 64u) {
            uint gb = (row * K_blocks + b) * 34u;
            uint col = b * 32u;
            for (uint e = 0u; e < 32u; e++) acc += dq_v(gb, e) * x[col + e];
        }
        gs[tid.x] = acc;
        GroupMemoryBarrierWithGroupSync();
        if (tid.x < 32) gs[tid.x] += gs[tid.x+32]; GroupMemoryBarrierWithGroupSync();
        if (tid.x < 16) gs[tid.x] += gs[tid.x+16]; GroupMemoryBarrierWithGroupSync();
        if (tid.x <  8) gs[tid.x] += gs[tid.x+ 8]; GroupMemoryBarrierWithGroupSync();
        if (tid.x <  4) gs[tid.x] += gs[tid.x+ 4]; GroupMemoryBarrierWithGroupSync();
        if (tid.x <  2) gs[tid.x] += gs[tid.x+ 2]; GroupMemoryBarrierWithGroupSync();
        if (tid.x <  1) gs[tid.x] += gs[tid.x+ 1]; GroupMemoryBarrierWithGroupSync();
        if (tid.x == 0) yV[row] = gs[0];
    }
}
)";

static const char* HLSL_RMSNORM = R"(
Buffer<float>    x : register(t0);
Buffer<float>    w : register(t1);
RWBuffer<float>  y : register(u0);
cbuffer CB0 : register(b0) { uint D; float eps; uint p0; uint p1; };
groupshared float gs[256];
[numthreads(256,1,1)]
void CSMain(uint3 t3 : SV_GroupThreadID) {
    uint t = t3.x;
    float s = 0.0f;
    for (uint i = t; i < D; i += 256u) { float v = x[i]; s += v * v; }
    gs[t] = s;
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 128u; k > 0u; k >>= 1u) {
        if (t < k) gs[t] += gs[t + k];
        GroupMemoryBarrierWithGroupSync();
    }
    float rms = rsqrt(gs[0] / (float)D + eps);
    for (uint i = t; i < D; i += 256u) y[i] = x[i] * rms * w[i];
}
)";

// RoPE applied in-place to a [nHeads, head_dim] buffer.
// One group per head; each thread handles head_dim/2/64 pairs.
static const char* HLSL_ROPE = R"(
RWBuffer<float> Q : register(u0);
cbuffer CB0 : register(b0) { uint nH; uint hD; uint pos; float theta; };
[numthreads(64,1,1)]
void CSMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint h  = gid.y;
    uint pi = gid.x * 64u + tid.x; // pair index
    if (h >= nH || pi >= hD / 2u) return;
    float freq  = 1.0f / pow(theta, (float)(pi * 2u) / (float)hD);
    float angle = (float)pos * freq;
    float c = cos(angle), s = sin(angle);
    uint  base = h * hD + pi * 2u;
    float q0 = Q[base], q1 = Q[base + 1u];
    Q[base]      = q0 * c - q1 * s;
    Q[base + 1u] = q0 * s + q1 * c;
}
)";

static const char* HLSL_SILU = R"(
Buffer<float>   gate : register(t0);
Buffer<float>   up   : register(t1);
RWBuffer<float> dst  : register(u0);
cbuffer CB0 : register(b0) { uint D; uint p0; uint p1; uint p2; };
[numthreads(64,1,1)]
void CSMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint i = gid.x * 64u + tid.x;
    if (i >= D) return;
    float g = gate[i];
    dst[i] = (g / (1.0f + exp(-g))) * up[i];
}
)";

static const char* HLSL_ADD = R"(
Buffer<float>   b : register(t0);
RWBuffer<float> a : register(u0);
cbuffer CB0 : register(b0) { uint N; uint p0; uint p1; uint p2; };
[numthreads(64,1,1)]
void CSMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint i = gid.x * 64u + tid.x;
    if (i < N) a[i] += b[i];
}
)";

// GQA attention: single token against KV cache.
// Q [nH, hD], K_cache [nKV, cacheStride, hD], V_cache [nKV, cacheStride, hD] → out [nH, hD].
// cacheStride = MAX_SEQ (1024) — the allocated size, not the current T.
// One group dispatched per query head; 64 threads share score reduction.
static const char* HLSL_ATTN = R"(
Buffer<float>   Q  : register(t0);
Buffer<float>   KC : register(t1);
Buffer<float>   VC : register(t2);
RWBuffer<float> O  : register(u0);
cbuffer CB0 : register(b0) { uint nH; uint nKV; uint hD; uint T;
                              float sc; uint cacheStride; uint p1; uint p2; };
groupshared float gs[1024]; // max T=1024 for this kernel
[numthreads(64,1,1)]
void CSMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint h   = gid.x;
    uint kvh = h / (nH / nKV);
    if (h >= nH) return;
    uint qBase  = h * hD;
    uint kvBase = kvh * cacheStride * hD; // stride = MAX_SEQ, not T
    // QK^T scores
    for (uint t = tid.x; t < T; t += 64u) {
        float dot = 0.0f;
        for (uint d = 0u; d < hD; d++) dot += Q[qBase + d] * KC[kvBase + t * hD + d];
        gs[t] = dot * sc;
    }
    GroupMemoryBarrierWithGroupSync();
    // Softmax (serial, tid==0 only — T <= 1024)
    if (tid.x == 0u) {
        float mx = gs[0]; for (uint t=1;t<T;t++) if(gs[t]>mx) mx=gs[t];
        float s = 0.0f;   for (uint t=0;t<T;t++) { gs[t]=exp(gs[t]-mx); s+=gs[t]; }
        for (uint t=0;t<T;t++) gs[t] /= s;
    }
    GroupMemoryBarrierWithGroupSync();
    // Weighted sum of V
    uint oBase = h * hD;
    for (uint d = tid.x; d < hD; d += 64u) {
        float acc = 0.0f;
        for (uint t = 0u; t < T; t++) acc += gs[t] * VC[kvBase + t * hD + d];
        O[oBase + d] = acc;
    }
}
)";

// Per-head in-place RMSNorm for QK-norm: each group = one head (128 threads)
static const char* HLSL_QKNORM = R"(
RWBuffer<float>  Q : register(u0);
Buffer<float>    W : register(t0);
cbuffer CB0 : register(b0) { uint nH; uint hD; float eps; uint pad; };
groupshared float gs[128];
[numthreads(128,1,1)]
void CSMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint h = gid.x;
    if (h >= nH) return;
    uint base = h * hD;
    float val = (tid.x < hD) ? Q[base + tid.x] : 0.0f;
    gs[tid.x] = val * val;
    GroupMemoryBarrierWithGroupSync();
    if (tid.x < 64) gs[tid.x] += gs[tid.x + 64]; GroupMemoryBarrierWithGroupSync();
    if (tid.x < 32) gs[tid.x] += gs[tid.x + 32]; GroupMemoryBarrierWithGroupSync();
    if (tid.x < 16) gs[tid.x] += gs[tid.x + 16]; GroupMemoryBarrierWithGroupSync();
    if (tid.x <  8) gs[tid.x] += gs[tid.x +  8]; GroupMemoryBarrierWithGroupSync();
    if (tid.x <  4) gs[tid.x] += gs[tid.x +  4]; GroupMemoryBarrierWithGroupSync();
    if (tid.x <  2) gs[tid.x] += gs[tid.x +  2]; GroupMemoryBarrierWithGroupSync();
    if (tid.x <  1) gs[tid.x] += gs[tid.x +  1]; GroupMemoryBarrierWithGroupSync();
    float rms = rsqrt(gs[0] / (float)hD + eps);
    if (tid.x < hD) Q[base + tid.x] = val * rms * W[tid.x];
}
)";

// ============================================================================
// Minimal GGUF reader
// ============================================================================
namespace gguf {

struct Tensor {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type; // 0=F32, 1=F16, 8=Q8_0
    uint64_t file_offset; // offset from start of file to data
    uint64_t n_elems;
    uint64_t data_bytes;
};

struct Meta {
    uint32_t block_count = 28;
    uint32_t embed_len   = 2048;
    uint32_t ff_len      = 6144;
    uint32_t n_heads     = 16;
    uint32_t n_kv_heads  = 8;
    float    rope_theta  = 1000000.0f;
    uint32_t eos_token   = 151645;
    uint32_t vocab_size  = 151936;
    std::vector<Tensor> tensors;
    std::unordered_map<std::string,size_t> idx;
    std::vector<std::string> vocab;   // token text by ID (Ġ convention)
    std::vector<std::string> merges;  // BPE merge rules: "tok1 tok2"
};

static uint64_t ru64(std::ifstream& f){ uint64_t v; f.read((char*)&v,8); return v; }
static uint32_t ru32(std::ifstream& f){ uint32_t v; f.read((char*)&v,4); return v; }
static float    rf32(std::ifstream& f){ float    v; f.read((char*)&v,4); return v; }
static uint8_t  ru8 (std::ifstream& f){ uint8_t  v; f.read((char*)&v,1); return v; }
static std::string rstr(std::ifstream& f){
    uint64_t n=ru64(f); std::string s(n,'\0'); f.read(&s[0],n); return s;
}

// Forward-declare
static void skipVal(std::ifstream& f, uint32_t t);
static void skipArr(std::ifstream& f){
    uint32_t et=ru32(f); uint64_t n=ru64(f);
    // Fast path: fixed-size element types (skip all at once)
    uint64_t esz=0;
    switch(et){
        case 0: case 1: case 7: esz=1; break;  // u8,i8,bool
        case 2: case 3:         esz=2; break;  // u16,i16
        case 4: case 5: case 6: esz=4; break;  // u32,i32,f32
        case 10: case 11: case 12: esz=8; break; // u64,i64,f64
    }
    if(esz>0){ f.ignore((std::streamsize)(n*esz)); return; }
    for(uint64_t i=0;i<n;i++) skipVal(f,et);
}
static void skipVal(std::ifstream& f, uint32_t t){
    // GGUF value types (from gguf-py GGUFValueType enum):
    //   0=u8,1=i8,2=u16,3=i16,4=u32,5=i32,6=f32,7=bool,8=string,9=array,
    //   10=u64,11=i64,12=f64
    switch(t){
        case 0: case 1: case 7: f.ignore(1); break;
        case 2: case 3:         f.ignore(2); break;
        case 4: case 5: case 6: f.ignore(4); break;
        case 8:  { uint64_t n=ru64(f); f.ignore((std::streamsize)n); break; }
        case 9:  skipArr(f); break;
        case 10: case 11: case 12: f.ignore(8); break;
        default: break;
    }
}

static uint64_t elemBytes(uint32_t type, uint64_t n){
    if(type==0) return n*4;
    if(type==1) return n*2;
    if(type==8){ uint64_t b=(n+31)/32; return b*34; } // Q8_0
    if(type==2){ uint64_t b=(n+31)/32; return b*18; } // Q4_0
    return n*4;
}
static uint64_t alignUp(uint64_t v,uint64_t a){ return (v+a-1)&~(a-1); }

bool load(const char* path, Meta& m){
    std::ifstream f(path, std::ios::binary);
    if(!f.is_open()) return false;
    char magic[4]; f.read(magic,4);
    if(memcmp(magic,"GGUF",4)!=0) return false;
    uint32_t ver=ru32(f);
    if(ver!=2&&ver!=3) return false;
    uint64_t nt=ru64(f), nkv=ru64(f);

    for(uint64_t i=0;i<nkv;i++){
        std::string key=rstr(f);
        uint32_t vt=ru32(f);
        // extract numeric fields we care about
        auto gu=[&]()->uint32_t{
            switch(vt){
                case 0:  return ru8(f);
                case 1:  { int8_t  v; f.read((char*)&v,1); return (uint32_t)v; }
                case 2:  { uint16_t v; f.read((char*)&v,2); return v; }
                case 3:  { int16_t  v; f.read((char*)&v,2); return (uint32_t)v; }
                case 4:  return ru32(f);
                case 5:  { int32_t  v; f.read((char*)&v,4); return (uint32_t)v; }
                case 10: return (uint32_t)ru64(f);
                case 11: { int64_t  v; f.read((char*)&v,8); return (uint32_t)v; }
                default: skipVal(f,vt); return 0;
            }
        };
        auto gf=[&]()->float{
            switch(vt){
                case 6: return rf32(f);
                case 4: return (float)ru32(f);
                default: skipVal(f,vt); return 0.f;
            }
        };
        if(key=="qwen3.block_count"||key=="llama.block_count") m.block_count=gu();
        else if(key=="qwen3.embedding_length"||key=="llama.embedding_length") m.embed_len=gu();
        else if(key=="qwen3.feed_forward_length"||key=="llama.feed_forward_length") m.ff_len=gu();
        else if(key=="qwen3.attention.head_count"||key=="llama.attention.head_count") m.n_heads=gu();
        else if(key=="qwen3.attention.head_count_kv"||key=="llama.attention.head_count_kv") m.n_kv_heads=gu();
        else if(key=="qwen3.rope.freq_base"||key=="llama.rope.freq_base") m.rope_theta=gf();
        else if(key=="tokenizer.ggml.eos_token_id") m.eos_token=gu();
        else if(key=="tokenizer.ggml.tokens"&&vt==9){
            uint32_t et=ru32(f); uint64_t n=ru64(f);
            m.vocab.resize(n);
            for(uint64_t i=0;i<n;i++) m.vocab[i]=rstr(f);
        }
        else if(key=="tokenizer.ggml.merges"&&vt==9){
            uint32_t et=ru32(f); uint64_t n=ru64(f);
            m.merges.reserve(n);
            for(uint64_t i=0;i<n;i++) m.merges.push_back(rstr(f));
        }
        else { skipVal(f,vt); }
    }

    m.tensors.resize(nt);
    for(uint64_t i=0;i<nt;i++){
        Tensor& t=m.tensors[i];
        t.name=rstr(f);
        uint32_t nd=ru32(f); t.dims.resize(nd);
        for(uint32_t d=0;d<nd;d++) t.dims[d]=ru64(f);
        t.type=ru32(f);
        t.file_offset=ru64(f); // relative to data section
        t.n_elems=1; for(auto d:t.dims) t.n_elems*=d;
        t.data_bytes=elemBytes(t.type,t.n_elems);
    }
    uint64_t data_start=alignUp((uint64_t)f.tellg(),32);
    for(auto& t:m.tensors){
        t.file_offset+=data_start;
        m.idx[t.name]=&t-m.tensors.data();
    }
    return true;
}

// Read raw bytes of a tensor directly from file (no dequant)
static bool readRaw(const char* path, const Tensor& t, std::vector<uint8_t>& out){
    std::ifstream f(path,std::ios::binary);
    if(!f.is_open()) return false;
    f.seekg(t.file_offset);
    // align to 4 bytes for GPU upload
    uint64_t aligned = alignUp(t.data_bytes, 4);
    out.resize(aligned, 0);
    f.read((char*)out.data(), t.data_bytes);
    return (bool)f;
}

// Dequantize to float32
static bool dequant(const char* path, const Tensor& t, std::vector<float>& out){
    std::ifstream f(path,std::ios::binary);
    if(!f.is_open()) return false;
    f.seekg(t.file_offset);
    out.resize(t.n_elems);
    if(t.type==0){ f.read((char*)out.data(),t.n_elems*4); return (bool)f; }
    if(t.type==8){
        uint64_t nb=(t.n_elems+31)/32;
        for(uint64_t b=0;b<nb;b++){
            uint16_t f16; f.read((char*)&f16,2);
            uint32_t sign=(f16>>15)&1, exp=(f16>>10)&0x1F, mant=f16&0x3FF;
            float sc;
            if(exp==0) sc=(mant==0)?0.f:ldexpf((float)mant,-24);
            else if(exp==31) sc=mant?std::numeric_limits<float>::quiet_NaN():
                                     std::numeric_limits<float>::infinity();
            else{ uint32_t fb=(sign<<31)|((exp+112)<<23)|(mant<<13); memcpy(&sc,&fb,4); }
            if(sign&&exp!=31) sc=-fabsf(sc);
            uint64_t base=b*32;
            for(int i=0;i<32;i++){
                int8_t q; f.read((char*)&q,1);
                if(base+i<t.n_elems) out[base+i]=(float)q*sc;
            }
        }
        return true;
    }
    return false;
}
} // namespace gguf

// ============================================================================
// D3D11 types and helpers
// ============================================================================
struct GpuBuf {
    ID3D11Buffer*              buf=nullptr;
    ID3D11ShaderResourceView*  srv=nullptr;
    ID3D11UnorderedAccessView* uav=nullptr;
    uint32_t n=0;
};

static void freeBuf(GpuBuf& g){
    if(g.uav){g.uav->Release();g.uav=nullptr;}
    if(g.srv){g.srv->Release();g.srv=nullptr;}
    if(g.buf){g.buf->Release();g.buf=nullptr;}
    g.n=0;
}

// Typed R32_FLOAT float buffer (SRV + UAV) — avoids structured-buffer UAV driver issues
static GpuBuf makeFloatBuf(ID3D11Device* d, uint32_t n){
    GpuBuf g; g.n=n;
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth=n*4; bd.Usage=D3D11_USAGE_DEFAULT;
    bd.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
    bd.MiscFlags=0;
    bd.StructureByteStride=0;
    d->CreateBuffer(&bd,nullptr,&g.buf);
    if(!g.buf) return g;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format=DXGI_FORMAT_R32_FLOAT;
    sv.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;
    sv.Buffer.FirstElement=0;
    sv.Buffer.NumElements=n;
    d->CreateShaderResourceView(g.buf,&sv,&g.srv);
    D3D11_UNORDERED_ACCESS_VIEW_DESC uv{};
    uv.Format=DXGI_FORMAT_R32_FLOAT;
    uv.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;
    uv.Buffer.FirstElement=0;
    uv.Buffer.NumElements=n;
    d->CreateUnorderedAccessView(g.buf,&uv,&g.uav);
    return g;
}

// Raw ByteAddressBuffer (SRV only, for Q8_0 weights)
// R32_UINT typed buffer — avoids ByteAddressBuffer raw-view driver issues
static GpuBuf makeRawBuf(ID3D11Device* d, const uint8_t* data, uint32_t bytes){
    GpuBuf g;
    uint32_t aligned=(bytes+3)&~3u;
    g.n=aligned/4;
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth=aligned; bd.Usage=D3D11_USAGE_IMMUTABLE;
    bd.BindFlags=D3D11_BIND_SHADER_RESOURCE;
    bd.MiscFlags=0;
    std::vector<uint8_t> padded(aligned,0);
    if(data) memcpy(padded.data(),data,bytes);
    D3D11_SUBRESOURCE_DATA init{padded.data(),0,0};
    d->CreateBuffer(&bd,&init,&g.buf);
    if(!g.buf) return g;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format=DXGI_FORMAT_R32_UINT;
    sv.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;
    sv.Buffer.FirstElement=0;
    sv.Buffer.NumElements=aligned/4;
    d->CreateShaderResourceView(g.buf,&sv,&g.srv);
    return g;
}

static void uploadF32(ID3D11DeviceContext* c, GpuBuf& g, const float* d, uint32_t n){
    // No pDstBox — partial UpdateSubresource is undefined for non-structured buffers on some drivers
    c->UpdateSubresource(g.buf,0,nullptr,d,0,0);
}

static void readbackF32(ID3D11DeviceContext* c, ID3D11Device* d,
                        GpuBuf& src, float* dst, uint32_t n){
    D3D11_BUFFER_DESC bd{}; src.buf->GetDesc(&bd);
    bd.Usage=D3D11_USAGE_STAGING; bd.BindFlags=0;
    bd.CPUAccessFlags=D3D11_CPU_ACCESS_READ; bd.MiscFlags=0;
    ID3D11Buffer* st=nullptr; d->CreateBuffer(&bd,nullptr,&st); if(!st) return;
    c->CopyResource(st,src.buf);
    D3D11_MAPPED_SUBRESOURCE ms{}; c->Map(st,0,D3D11_MAP_READ,0,&ms);
    memcpy(dst,ms.pData,n*4); c->Unmap(st,0); st->Release();
}

static FILE* g_diag = nullptr;
static void diag(const char* fmt, ...){
    if(!g_diag){ g_diag=fopen("C:\\Users\\canna\\_khanary_inspect\\drivers\\qwen_diag.txt","a"); }
    if(!g_diag) return;
    va_list ap; va_start(ap,fmt); vfprintf(g_diag,fmt,ap); va_end(ap);
    fflush(g_diag);
}

struct CS { ID3D11ComputeShader* cs=nullptr; ID3DBlob* blob=nullptr; };
static bool compileCS(ID3D11Device* d, const char* src, CS& s){
    ID3DBlob* err=nullptr;
    HRESULT hr=D3DCompile(src,strlen(src),nullptr,nullptr,nullptr,"CSMain","cs_5_0",
                          D3DCOMPILE_OPTIMIZATION_LEVEL1,0,&s.blob,&err);
    if(FAILED(hr)){
        diag("compileCS D3DCompile hr=0x%08X err=%s\n", (unsigned)hr,
             err?(const char*)err->GetBufferPointer():"(none)");
        if(err)err->Release(); return false;
    }
    hr=d->CreateComputeShader(s.blob->GetBufferPointer(),s.blob->GetBufferSize(),nullptr,&s.cs);
    if(FAILED(hr)){ diag("compileCS CreateComputeShader hr=0x%08X\n",(unsigned)hr); return false; }
    return true;
}

// ============================================================================
// Per-layer GPU weights
// ============================================================================
struct LayerGPU {
    GpuBuf q8_q, q8_k, q8_v, q8_out;   // Q8_0 attention weight buffers
    GpuBuf q8_gate, q8_up, q8_down;     // Q8_0 FFN buffers
    GpuBuf f32_attn_norm, f32_ffn_norm; // F32 RMSNorm scale
    GpuBuf f32_q_norm, f32_k_norm;      // F32 QK-norm scale [head_dim]
    // CPU copies of qk_norm (small, for CPU fallback)
    std::vector<float> cpu_q_norm, cpu_k_norm;
};

// ============================================================================
// Main inference state
// ============================================================================
struct QwenState {
    QwenInferConfig cfg;
    bool model_loaded = false;

    // D3D11
    ID3D11Device*        dev=nullptr;
    ID3D11DeviceContext* ctx=nullptr;
    ID3D11Buffer*        cbuf=nullptr;
    CS cs_gemv_q8, cs_rmsnorm, cs_rope, cs_silu, cs_add, cs_attn, cs_qknorm, cs_gemv_qkv;
    bool gpu_ok = false;

    // GPU layer weights
    std::vector<LayerGPU> layer_gpu;

    // GPU activation buffers
    GpuBuf g_h;        // [embed] hidden state
    GpuBuf g_res;      // [embed] residual
    GpuBuf g_norm_out; // [embed] post-norm
    GpuBuf g_q;        // [num_heads * head_dim]
    GpuBuf g_k;        // [num_kv_heads * head_dim]
    GpuBuf g_v;        // [num_kv_heads * head_dim]
    GpuBuf g_attn;     // [num_heads * head_dim] attention output
    GpuBuf g_gate;     // [ff_dim]
    GpuBuf g_up;       // [ff_dim]
    GpuBuf g_ff;       // [ff_dim] SiLU output
    GpuBuf g_tmp;      // [embed] scratch for down-proj output

    // KV cache: [num_layers][kv_heads * MAX_SEQ * head_dim]
    static constexpr uint32_t MAX_SEQ = 1024;
    std::vector<GpuBuf> kv_k, kv_v;

    // CPU weights (embedding + lm_head only, too large for VRAM)
    std::vector<uint8_t> cpu_embd;    // Q8_0 raw bytes [vocab * embed]
    std::vector<float>   cpu_out_norm; // [embed] F32
    std::vector<uint8_t> cpu_lm_head; // Q8_0 raw bytes [vocab * embed]

    uint32_t kv_pos = 0; // number of tokens already in KV cache

    // Tokenizer data (populated at model-load time)
    std::vector<std::string>             id_to_tok;
    std::unordered_map<std::string,uint32_t> tok_to_id;
    std::unordered_map<uint64_t,uint32_t>    merge_ranks; // key=(id1<<32)|id2 → rank
    uint32_t byte_tok[256]; // byte value → token ID for <0xNN> tokens

    QwenState(){
        cfg.vocab_size   = 151936;
        cfg.max_seq_len  = 32768;
        cfg.embed_dim    = 2048;
        cfg.num_heads    = 16;
        cfg.num_kv_heads = 8;
        cfg.num_layers   = 28;
        cfg.ff_dim       = 6144;
        cfg.head_dim     = 128;
        cfg.rope_theta   = 1000000.0f;
    }

    std::string configJson() const {
        std::ostringstream s;
        s<<"{\"vocab_size\":"<<cfg.vocab_size
         <<",\"embed_dim\":"<<cfg.embed_dim
         <<",\"num_heads\":"<<cfg.num_heads
         <<",\"num_kv_heads\":"<<cfg.num_kv_heads
         <<",\"num_layers\":"<<cfg.num_layers
         <<",\"ff_dim\":"<<cfg.ff_dim
         <<",\"head_dim\":"<<cfg.head_dim
         <<",\"rope_theta\":"<<cfg.rope_theta
         <<",\"model_loaded\":"<<(model_loaded?"true":"false")
         <<",\"gpu_ok\":"<<(gpu_ok?"true":"false")
         <<",\"kv_pos\":"<<kv_pos
         <<",\"architecture\":\"qwen3\"}";
        return s.str();
    }
};

// ============================================================================
// CB update helper
// ============================================================================
static void setCB(ID3D11DeviceContext* c, ID3D11Buffer* cb, const void* data, uint32_t /*bytes*/=32){
    // Constant buffers must be updated with pDstBox=nullptr; partial updates are undefined
    c->UpdateSubresource(cb,0,nullptr,data,0,0);
}

// ============================================================================
// Dispatch helpers
// ============================================================================
static void dispatch_gemv_q8(QwenState& s, GpuBuf& W_q8, GpuBuf& x, GpuBuf& y,
                               uint32_t N, uint32_t K){
    uint32_t Kb=(K+31)/32;
    struct{ uint32_t N,K,Kb,pad; } cb{N,K,Kb,0};
    setCB(s.ctx,s.cbuf,&cb);
    s.ctx->CSSetConstantBuffers(0,1,&s.cbuf);
    ID3D11ShaderResourceView* srvs[2]={W_q8.srv,x.srv};
    s.ctx->CSSetShaderResources(0,2,srvs);
    s.ctx->CSSetUnorderedAccessViews(0,1,&y.uav,nullptr);
    s.ctx->CSSetShader(s.cs_gemv_q8.cs,nullptr,0);
    s.ctx->Dispatch(N,1,1);  // one group per output row for cache-locality
    ID3D11ShaderResourceView*  nsrv[2]={nullptr,nullptr};
    ID3D11UnorderedAccessView* nuav[1]={nullptr};
    s.ctx->CSSetShaderResources(0,2,nsrv);
    s.ctx->CSSetUnorderedAccessViews(0,1,nuav,nullptr);
}

static void dispatch_gemv_qkv(QwenState& s,
                               GpuBuf& Wq, GpuBuf& Wk, GpuBuf& Wv, GpuBuf& x,
                               GpuBuf& gQ, GpuBuf& gK, GpuBuf& gV,
                               uint32_t N_q, uint32_t N_kv, uint32_t K){
    uint32_t Kb = (K+31)/32;
    struct{ uint32_t N_q, N_kv, K_blocks, pad; } cb{N_q, N_kv, Kb, 0};
    setCB(s.ctx, s.cbuf, &cb);
    s.ctx->CSSetConstantBuffers(0, 1, &s.cbuf);
    ID3D11ShaderResourceView* srvs[4] = {Wq.srv, Wk.srv, Wv.srv, x.srv};
    s.ctx->CSSetShaderResources(0, 4, srvs);
    ID3D11UnorderedAccessView* uavs[3] = {gQ.uav, gK.uav, gV.uav};
    s.ctx->CSSetUnorderedAccessViews(0, 3, uavs, nullptr);
    s.ctx->CSSetShader(s.cs_gemv_qkv.cs, nullptr, 0);
    s.ctx->Dispatch(N_q + 2*N_kv, 1, 1);
    ID3D11ShaderResourceView*  nsrv[4] = {nullptr,nullptr,nullptr,nullptr};
    ID3D11UnorderedAccessView* nuav[3] = {nullptr,nullptr,nullptr};
    s.ctx->CSSetShaderResources(0, 4, nsrv);
    s.ctx->CSSetUnorderedAccessViews(0, 3, nuav, nullptr);
}

static void dispatch_rmsnorm(QwenState& s, GpuBuf& x, GpuBuf& w, GpuBuf& y, uint32_t D){
    struct{ uint32_t D; float eps; uint32_t p0,p1; } cb{D,1e-6f,0,0};
    setCB(s.ctx,s.cbuf,&cb);
    s.ctx->CSSetConstantBuffers(0,1,&s.cbuf);
    ID3D11ShaderResourceView* srvs[2]={x.srv,w.srv};
    s.ctx->CSSetShaderResources(0,2,srvs);
    s.ctx->CSSetUnorderedAccessViews(0,1,&y.uav,nullptr);
    s.ctx->CSSetShader(s.cs_rmsnorm.cs,nullptr,0);
    s.ctx->Dispatch(1,1,1);
    ID3D11ShaderResourceView*  nsrv[2]={nullptr,nullptr};
    ID3D11UnorderedAccessView* nuav[1]={nullptr};
    s.ctx->CSSetShaderResources(0,2,nsrv);
    s.ctx->CSSetUnorderedAccessViews(0,1,nuav,nullptr);
}

static void dispatch_rope(QwenState& s, GpuBuf& q, uint32_t nH, uint32_t hD, uint32_t pos){
    struct{ uint32_t nH,hD,pos; float theta; } cb{nH,hD,pos,s.cfg.rope_theta};
    setCB(s.ctx,s.cbuf,&cb);
    s.ctx->CSSetConstantBuffers(0,1,&s.cbuf);
    s.ctx->CSSetUnorderedAccessViews(0,1,&q.uav,nullptr);
    s.ctx->CSSetShader(s.cs_rope.cs,nullptr,0);
    s.ctx->Dispatch((hD/2+63)/64,nH,1);
    ID3D11UnorderedAccessView* nuav[1]={nullptr};
    s.ctx->CSSetUnorderedAccessViews(0,1,nuav,nullptr);
}

static void dispatch_silu(QwenState& s, GpuBuf& gate, GpuBuf& up, GpuBuf& out, uint32_t D){
    struct{ uint32_t D,p0,p1,p2; } cb{D,0,0,0};
    setCB(s.ctx,s.cbuf,&cb);
    s.ctx->CSSetConstantBuffers(0,1,&s.cbuf);
    ID3D11ShaderResourceView* srvs[2]={gate.srv,up.srv};
    s.ctx->CSSetShaderResources(0,2,srvs);
    s.ctx->CSSetUnorderedAccessViews(0,1,&out.uav,nullptr);
    s.ctx->CSSetShader(s.cs_silu.cs,nullptr,0);
    s.ctx->Dispatch((D+63)/64,1,1);
    ID3D11ShaderResourceView*  nsrv[2]={nullptr,nullptr};
    ID3D11UnorderedAccessView* nuav[1]={nullptr};
    s.ctx->CSSetShaderResources(0,2,nsrv);
    s.ctx->CSSetUnorderedAccessViews(0,1,nuav,nullptr);
}

static void dispatch_add(QwenState& s, GpuBuf& acc, GpuBuf& b, uint32_t N){
    struct{ uint32_t N,p0,p1,p2; } cb{N,0,0,0};
    setCB(s.ctx,s.cbuf,&cb);
    s.ctx->CSSetConstantBuffers(0,1,&s.cbuf);
    s.ctx->CSSetShaderResources(0,1,&b.srv);
    s.ctx->CSSetUnorderedAccessViews(0,1,&acc.uav,nullptr);
    s.ctx->CSSetShader(s.cs_add.cs,nullptr,0);
    s.ctx->Dispatch((N+63)/64,1,1);
    ID3D11ShaderResourceView*  nsrv[1]={nullptr};
    ID3D11UnorderedAccessView* nuav[1]={nullptr};
    s.ctx->CSSetShaderResources(0,1,nsrv);
    s.ctx->CSSetUnorderedAccessViews(0,1,nuav,nullptr);
}

static void dispatch_attn(QwenState& s, GpuBuf& Q, GpuBuf& KC, GpuBuf& VC,
                           GpuBuf& out, uint32_t nH, uint32_t nKV, uint32_t hD, uint32_t T){
    struct{ uint32_t nH,nKV,hD,T; float sc; uint32_t cacheStride,p1,p2; } cb;
    // cacheStride = MAX_SEQ: the actual stride in KV cache, not the current T
    cb={nH,nKV,hD,T,1.0f/sqrtf((float)hD),QwenState::MAX_SEQ,0,0};
    setCB(s.ctx,s.cbuf,&cb);
    s.ctx->CSSetConstantBuffers(0,1,&s.cbuf);
    ID3D11ShaderResourceView* srvs[3]={Q.srv,KC.srv,VC.srv};
    s.ctx->CSSetShaderResources(0,3,srvs);
    s.ctx->CSSetUnorderedAccessViews(0,1,&out.uav,nullptr);
    s.ctx->CSSetShader(s.cs_attn.cs,nullptr,0);
    s.ctx->Dispatch(nH,1,1);
    ID3D11ShaderResourceView*  nsrv[3]={nullptr,nullptr,nullptr};
    ID3D11UnorderedAccessView* nuav[1]={nullptr};
    s.ctx->CSSetShaderResources(0,3,nsrv);
    s.ctx->CSSetUnorderedAccessViews(0,1,nuav,nullptr);
}

static void dispatch_qknorm(QwenState& s, GpuBuf& q, GpuBuf& w, uint32_t nH, uint32_t hD){
    struct{ uint32_t nH, hD; float eps; uint32_t pad; } cb{nH, hD, 1e-6f, 0};
    setCB(s.ctx, s.cbuf, &cb);
    s.ctx->CSSetConstantBuffers(0, 1, &s.cbuf);
    s.ctx->CSSetShaderResources(0, 1, &w.srv);
    s.ctx->CSSetUnorderedAccessViews(0, 1, &q.uav, nullptr);
    s.ctx->CSSetShader(s.cs_qknorm.cs, nullptr, 0);
    s.ctx->Dispatch(nH, 1, 1);
    ID3D11ShaderResourceView*  nsrv[1]={nullptr};
    ID3D11UnorderedAccessView* nuav[1]={nullptr};
    s.ctx->CSSetShaderResources(0, 1, nsrv);
    s.ctx->CSSetUnorderedAccessViews(0, 1, nuav, nullptr);
}

// GPU-side KV cache write using CopySubresourceRegion — no CPU readback needed
static void writeKVGPU(ID3D11DeviceContext* c, GpuBuf& cache, GpuBuf& src,
                        uint32_t nKV, uint32_t hD, uint32_t pos){
    for(uint32_t h=0; h<nKV; h++){
        D3D11_BOX srcBox{ h*hD*4u, 0, 0, (h+1)*hD*4u, 1, 1 };
        c->CopySubresourceRegion(cache.buf, 0, (h*QwenState::MAX_SEQ+pos)*hD*4u, 0, 0,
                                  src.buf, 0, &srcBox);
    }
}

// ============================================================================
// D3D11 init
// ============================================================================
static bool initD3D(QwenState* s){
    D3D_FEATURE_LEVEL fl[]={D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL got;
    HRESULT hr=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,
                                  0,fl,1,D3D11_SDK_VERSION,&s->dev,&got,&s->ctx);
    if(FAILED(hr)){
        hr=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,
                              0,fl,1,D3D11_SDK_VERSION,&s->dev,&got,&s->ctx);
        if(FAILED(hr)){ diag("D3D11CreateDevice failed hr=0x%08X\n",(unsigned)hr); return false; }
    }

    D3D11_BUFFER_DESC cd{}; cd.ByteWidth=32; cd.Usage=D3D11_USAGE_DEFAULT;
    cd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    s->dev->CreateBuffer(&cd,nullptr,&s->cbuf);
    if(!s->cbuf){ diag("cbuf creation failed\n"); return false; }

    if(!compileCS(s->dev,HLSL_GEMV_Q8,  s->cs_gemv_q8))  return false;
    if(!compileCS(s->dev,HLSL_RMSNORM,  s->cs_rmsnorm))   return false;
    if(!compileCS(s->dev,HLSL_ROPE,     s->cs_rope))       return false;
    if(!compileCS(s->dev,HLSL_SILU,     s->cs_silu))       return false;
    if(!compileCS(s->dev,HLSL_ADD,      s->cs_add))        return false;
    if(!compileCS(s->dev,HLSL_ATTN,     s->cs_attn))       return false;
    if(!compileCS(s->dev,HLSL_QKNORM,    s->cs_qknorm))     return false;
    if(!compileCS(s->dev,HLSL_GEMV_Q8_QKV, s->cs_gemv_qkv)) return false;

    uint32_t E=s->cfg.embed_dim, F=s->cfg.ff_dim;
    uint32_t nH=s->cfg.num_heads, nKV=s->cfg.num_kv_heads, hD=s->cfg.head_dim;
    uint32_t nL=s->cfg.num_layers;

    s->g_h        = makeFloatBuf(s->dev, E);
    s->g_res      = makeFloatBuf(s->dev, E);
    s->g_norm_out = makeFloatBuf(s->dev, E);
    s->g_q        = makeFloatBuf(s->dev, nH*hD);
    s->g_k        = makeFloatBuf(s->dev, nKV*hD);
    s->g_v        = makeFloatBuf(s->dev, nKV*hD);
    s->g_attn     = makeFloatBuf(s->dev, nH*hD);
    s->g_gate     = makeFloatBuf(s->dev, F);
    s->g_up       = makeFloatBuf(s->dev, F);
    s->g_ff       = makeFloatBuf(s->dev, F);
    s->g_tmp      = makeFloatBuf(s->dev, E);

    // KV cache
    s->kv_k.resize(nL); s->kv_v.resize(nL);
    for(uint32_t l=0;l<nL;l++){
        s->kv_k[l]=makeFloatBuf(s->dev, nKV*QwenState::MAX_SEQ*hD);
        s->kv_v[l]=makeFloatBuf(s->dev, nKV*QwenState::MAX_SEQ*hD);
    }
    s->gpu_ok=true;
    return true;
}

// ============================================================================
// Tokenizer — greedy longest-match BPE (self-contained, no external deps)
// ============================================================================

static void initTokenizer(QwenState* s, gguf::Meta& m){
    s->id_to_tok = m.vocab;
    s->tok_to_id.reserve(m.vocab.size()*2);
    for(uint32_t i=0;i<(uint32_t)m.vocab.size();i++) s->tok_to_id[m.vocab[i]]=i;
    memset(s->byte_tok,0,sizeof(s->byte_tok));
    for(uint32_t i=0;i<(uint32_t)m.vocab.size();i++){
        const auto& t=m.vocab[i];
        // Match <0xNN> byte tokens (exactly 6 chars)
        if(t.size()==6&&t[0]=='<'&&t[1]=='0'&&t[2]=='x'&&t[5]=='>'){
            auto hv=[](char c)->int{ return c>='a'?c-'a'+10:(c>='A'?c-'A'+10:c-'0'); };
            int b=(hv(t[3])<<4)|hv(t[4]);
            if(b>=0&&b<256) s->byte_tok[b]=i;
        }
    }
    // Build merge ranks
    s->merge_ranks.reserve(m.merges.size()*2);
    for(uint32_t r=0;r<(uint32_t)m.merges.size();r++){
        const auto& ms=m.merges[r];
        size_t sp=ms.find(' ');
        if(sp==std::string::npos) continue;
        auto it1=s->tok_to_id.find(ms.substr(0,sp));
        auto it2=s->tok_to_id.find(ms.substr(sp+1));
        if(it1==s->tok_to_id.end()||it2==s->tok_to_id.end()) continue;
        uint64_t key=((uint64_t)it1->second<<32)|(uint64_t)it2->second;
        s->merge_ranks.emplace(key,r);
    }
}

// Ġ (U+0120) UTF-8: 0xC4 0xA0 — represents a leading space in tiktoken vocab
static const char* G_UTF8 = "\xC4\xA0"; // 2 bytes

static std::vector<uint32_t> tok_encode(QwenState* s, const std::string& text){
    std::vector<uint32_t> ids;
    ids.reserve(text.size());
    size_t i=0;
    while(i<text.size()){
        // Try longest match, up to 32 chars (longest Qwen3 token)
        size_t best_len=0; uint32_t best_id=0;
        size_t lim=std::min(text.size()-i,(size_t)48);
        for(size_t l=lim;l>=1;l--){
            // If text[i]==' ', try Ġ-prefixed form (space-prefix convention)
            if(text[i]==' '&&l>1){
                std::string gp=G_UTF8; gp+=text.substr(i+1,l-1);
                auto it=s->tok_to_id.find(gp);
                if(it!=s->tok_to_id.end()){best_len=l;best_id=it->second;break;}
            }
            auto it=s->tok_to_id.find(text.substr(i,l));
            if(it!=s->tok_to_id.end()){best_len=l;best_id=it->second;break;}
        }
        if(best_len>0){ ids.push_back(best_id); i+=best_len; }
        else{ ids.push_back(s->byte_tok[(uint8_t)text[i]]); i++; }
    }
    return ids;
}

static std::string tok_decode(QwenState* s, const std::vector<uint32_t>& ids){
    std::string out;
    out.reserve(ids.size()*4);
    for(uint32_t id:ids){
        if(id>=(uint32_t)s->id_to_tok.size()) continue;
        const std::string& t=s->id_to_tok[id];
        // <0xNN> byte tokens
        if(t.size()==6&&t[0]=='<'&&t[1]=='0'&&t[2]=='x'&&t[5]=='>'){
            auto hv=[](char c)->int{ return c>='a'?c-'a'+10:(c>='A'?c-'A'+10:c-'0'); };
            out+=(char)((hv(t[3])<<4)|hv(t[4]));
        } else {
            // Replace Ġ (0xC4 0xA0) → space, Ċ (0xC4 0x8A) → newline
            for(size_t k=0;k<t.size();k++){
                if((uint8_t)t[k]==0xC4&&k+1<t.size()){
                    if((uint8_t)t[k+1]==0xA0){ out+=' ';  k++; }
                    else if((uint8_t)t[k+1]==0x8A){ out+='\n'; k++; }
                    else { out+=t[k]; }
                } else { out+=t[k]; }
            }
        }
    }
    return out;
}

// ============================================================================
// Load weights: layer Q8_0 → GPU ByteAddressBuffers; embed/lm_head → CPU f32
// ============================================================================
static bool loadWeights(QwenState* s, gguf::Meta& m, const char* path){
    auto getF32=[&](const std::string& name, std::vector<float>& out)->bool{
        auto it=m.idx.find(name); if(it==m.idx.end()) return false;
        return gguf::dequant(path, m.tensors[it->second], out);
    };
    auto getQ8GPU=[&](const std::string& name, GpuBuf& out)->bool{
        auto it=m.idx.find(name); if(it==m.idx.end()) return false;
        const gguf::Tensor& t=m.tensors[it->second];
        std::vector<uint8_t> raw;
        if(!gguf::readRaw(path,t,raw)) return false;
        out=makeRawBuf(s->dev, raw.data(), (uint32_t)raw.size());
        return out.buf!=nullptr;
    };

    // Embedding and lm_head kept as Q8_0 bytes (~330 MB each vs 1.24 GB f32)
    auto getRaw=[&](const std::string& name, std::vector<uint8_t>& out)->bool{
        auto it=m.idx.find(name); if(it==m.idx.end()) return false;
        return gguf::readRaw(path, m.tensors[it->second], out);
    };
    if(!getRaw("token_embd.weight",  s->cpu_embd))    return false;
    if(!getF32("output_norm.weight", s->cpu_out_norm)) return false;
    if(!getRaw("output.weight",      s->cpu_lm_head))  return false;

    s->layer_gpu.resize(m.block_count);
    for(uint32_t l=0;l<m.block_count;l++){
        auto p="blk."+std::to_string(l)+".";
        LayerGPU& lg=s->layer_gpu[l];
        if(!getQ8GPU(p+"attn_q.weight",      lg.q8_q))    return false;
        if(!getQ8GPU(p+"attn_k.weight",      lg.q8_k))    return false;
        if(!getQ8GPU(p+"attn_v.weight",      lg.q8_v))    return false;
        if(!getQ8GPU(p+"attn_output.weight", lg.q8_out))  return false;
        if(!getQ8GPU(p+"ffn_gate.weight",    lg.q8_gate)) return false;
        if(!getQ8GPU(p+"ffn_up.weight",      lg.q8_up))   return false;
        if(!getQ8GPU(p+"ffn_down.weight",    lg.q8_down)) return false;
        // F32 norms
        std::vector<float> tmp;
        if(!getF32(p+"attn_norm.weight", tmp)) return false;
        lg.f32_attn_norm=makeFloatBuf(s->dev,(uint32_t)tmp.size());
        uploadF32(s->ctx,lg.f32_attn_norm,tmp.data(),(uint32_t)tmp.size());
        if(!getF32(p+"ffn_norm.weight", tmp)) return false;
        lg.f32_ffn_norm=makeFloatBuf(s->dev,(uint32_t)tmp.size());
        uploadF32(s->ctx,lg.f32_ffn_norm,tmp.data(),(uint32_t)tmp.size());
        if(!getF32(p+"attn_q_norm.weight", lg.cpu_q_norm)) return false;
        if(!getF32(p+"attn_k_norm.weight", lg.cpu_k_norm)) return false;
        lg.f32_q_norm=makeFloatBuf(s->dev,(uint32_t)lg.cpu_q_norm.size());
        uploadF32(s->ctx,lg.f32_q_norm,lg.cpu_q_norm.data(),(uint32_t)lg.cpu_q_norm.size());
        lg.f32_k_norm=makeFloatBuf(s->dev,(uint32_t)lg.cpu_k_norm.size());
        uploadF32(s->ctx,lg.f32_k_norm,lg.cpu_k_norm.data(),(uint32_t)lg.cpu_k_norm.size());
    }
    return true;
}

// ============================================================================
// Forward pass — one token at position pos
// ============================================================================
static bool forward_token(QwenState* s, int32_t token, uint32_t pos){
    uint32_t E=s->cfg.embed_dim, F=s->cfg.ff_dim;
    uint32_t nH=s->cfg.num_heads, nKV=s->cfg.num_kv_heads, hD=s->cfg.head_dim;
    uint32_t nL=s->cfg.num_layers;
    uint32_t T=pos+1;

    // Embed: dequantize one Q8_0 row for this token
    {
        std::vector<float> emb(E);
        uint64_t K_blocks=(E+31)/32;
        const uint8_t* row_bytes=s->cpu_embd.data()+(uint64_t)token*K_blocks*34;
        for(uint64_t b=0;b<K_blocks;b++){
            const uint8_t* blk=row_bytes+b*34;
            uint16_t f16; memcpy(&f16,blk,2);
            uint32_t sign=(f16>>15)&1,exp=(f16>>10)&0x1F,mant=f16&0x3FF;
            float sc;
            if(exp==0) sc=(mant==0)?0.f:ldexpf((float)mant,-24);
            else if(exp==31) sc=(float)1e30;
            else{ uint32_t fb=(sign<<31)|((exp+112)<<23)|(mant<<13); memcpy(&sc,&fb,4); }
            if(sign&&exp!=31) sc=-fabsf(sc);
            for(uint32_t e=0;e<32&&b*32+e<E;e++)
                emb[b*32+e]=(float)(int8_t)blk[2+e]*sc;
        }
        uploadF32(s->ctx, s->g_h, emb.data(), E);
    }

    for(uint32_t l=0;l<nL;l++){
        LayerGPU& lg=s->layer_gpu[l];

        // Save residual
        s->ctx->CopyResource(s->g_res.buf, s->g_h.buf);

        // Attention pre-norm
        dispatch_rmsnorm(*s, s->g_h, lg.f32_attn_norm, s->g_norm_out, E);

        // QKV projections (fused — 1 dispatch instead of 3, saves ~168ms/forward)
        dispatch_gemv_qkv(*s, lg.q8_q, lg.q8_k, lg.q8_v, s->g_norm_out,
                          s->g_q, s->g_k, s->g_v, nH*hD, nKV*hD, E);

        // QK-norm (GPU)
        dispatch_qknorm(*s, s->g_q, lg.f32_q_norm, nH,  hD);
        dispatch_qknorm(*s, s->g_k, lg.f32_k_norm, nKV, hD);

        // RoPE
        dispatch_rope(*s, s->g_q, nH,  hD, pos);
        dispatch_rope(*s, s->g_k, nKV, hD, pos);

        // Update KV cache (GPU-side copy — no CPU readback)
        writeKVGPU(s->ctx, s->kv_k[l], s->g_k, nKV, hD, pos);
        writeKVGPU(s->ctx, s->kv_v[l], s->g_v, nKV, hD, pos);

        // GQA attention
        dispatch_attn(*s, s->g_q, s->kv_k[l], s->kv_v[l],
                      s->g_attn, nH, nKV, hD, T);

        // Attention output projection
        dispatch_gemv_q8(*s, lg.q8_out, s->g_attn, s->g_h, E, nH*hD);

        // Residual
        dispatch_add(*s, s->g_h, s->g_res, E);

        // FFN pre-norm
        s->ctx->CopyResource(s->g_res.buf, s->g_h.buf);
        dispatch_rmsnorm(*s, s->g_h, lg.f32_ffn_norm, s->g_norm_out, E);

        // Gate + Up
        dispatch_gemv_q8(*s, lg.q8_gate, s->g_norm_out, s->g_gate, F, E);
        dispatch_gemv_q8(*s, lg.q8_up,   s->g_norm_out, s->g_up,   F, E);

        // SiLU-gate
        dispatch_silu(*s, s->g_gate, s->g_up, s->g_ff, F);

        // Down projection
        dispatch_gemv_q8(*s, lg.q8_down, s->g_ff, s->g_tmp, E, F);

        // Residual + advance hidden state
        dispatch_add(*s, s->g_tmp, s->g_res, E);
        s->ctx->CopyResource(s->g_h.buf, s->g_tmp.buf);
    }
    return true;
}

// Helper: f16 bits → f32
static float f16_to_f32(uint16_t f16){
    uint32_t sign=(f16>>15)&1,exp=(f16>>10)&0x1F,mant=f16&0x3FF;
    if(exp==0) return (mant==0)?0.f:(sign?-1.f:1.f)*ldexpf((float)mant,-24);
    if(exp==31) return sign?-1e30f:1e30f;
    uint32_t fb=(sign<<31)|((exp+112)<<23)|(mant<<13); float v; memcpy(&v,&fb,4); return v;
}

// Compute logits on CPU — lm_head is Q8_0 bytes, hidden is F32
static void cpu_lm_head(QwenState* s, std::vector<float>& logits){
    uint32_t E=s->cfg.embed_dim, V=s->cfg.vocab_size;
    uint64_t K_blocks=(E+31)/32;

    // Read back hidden state
    std::vector<float> h(E);
    readbackF32(s->ctx,s->dev,s->g_h,h.data(),E);

    // Final RMSNorm on CPU
    float ss=0; for(uint32_t i=0;i<E;i++) ss+=h[i]*h[i];
    float rms=1.f/sqrtf(ss/E+1e-6f);
    for(uint32_t i=0;i<E;i++) h[i]*=rms*s->cpu_out_norm[i];

    // lm_head with inline Q8_0 dequant: logits[v] = dot(W_q8[v], h)
    logits.resize(V);
    const uint8_t* W=s->cpu_lm_head.data();
    for(uint32_t v=0;v<V;v++){
        float acc=0;
        const uint8_t* row=W+(uint64_t)v*K_blocks*34;
        for(uint64_t b=0;b<K_blocks;b++){
            const uint8_t* blk=row+b*34;
            float sc=f16_to_f32(*(uint16_t*)blk);
            for(uint32_t e=0;e<32&&b*32+e<E;e++)
                acc+=(float)(int8_t)blk[2+e]*sc*h[b*32+e];
        }
        logits[v]=acc;
    }
}

// ============================================================================
// Top-k sampling
// ============================================================================
static int32_t topk_sample(const std::vector<float>& logits, int k){
    int V=(int)logits.size();
    if(k<=0||k>V) k=40;
    std::vector<int> idx(V); std::iota(idx.begin(),idx.end(),0);
    std::partial_sort(idx.begin(),idx.begin()+k,idx.end(),
        [&](int a,int b){return logits[a]>logits[b];});
    float mx=logits[idx[0]], s=0;
    std::vector<float> p(k);
    for(int i=0;i<k;i++){p[i]=expf(logits[idx[i]]-mx);s+=p[i];}
    float r=(float)rand()/RAND_MAX, cdf=0;
    for(int i=0;i<k;i++){cdf+=p[i]/s;if(r<=cdf) return idx[i];}
    return idx[0];
}

// ============================================================================
// Public API
// ============================================================================
extern "C" {

QWEN_INFER_API void* qw_create(const char*){
    return new QwenState();
}

QWEN_INFER_API void qw_destroy(void* ctx){
    delete static_cast<QwenState*>(ctx);
}

QWEN_INFER_API int qw_load_model(void* ctx, const char* path,
                                  char* err_buf, int err_sz){
    auto* s=static_cast<QwenState*>(ctx);
    if(!s||!path){if(err_buf)strncpy(err_buf,"null",err_sz-1);return 0;}

    gguf::Meta m;
    if(!gguf::load(path,m)){if(err_buf)strncpy(err_buf,"gguf_parse_fail",err_sz-1);return 0;}

    // Update config from file
    s->cfg.num_layers  =m.block_count;
    s->cfg.embed_dim   =m.embed_len;
    s->cfg.ff_dim      =m.ff_len;
    s->cfg.num_heads   =m.n_heads;
    s->cfg.num_kv_heads=m.n_kv_heads;
    s->cfg.rope_theta  =m.rope_theta;
    s->cfg.head_dim    =m.embed_len/m.n_heads;
    s->cfg.vocab_size  =m.vocab_size;

    if(!initD3D(s)){if(err_buf)strncpy(err_buf,"d3d11_init_fail",err_sz-1);return 0;}
    if(!loadWeights(s,m,path)){if(err_buf)strncpy(err_buf,"weight_load_fail",err_sz-1);return 0;}
    initTokenizer(s,m);

    s->model_loaded=true;
    return 1;
}

QWEN_INFER_API char* qw_forward(void* ctx, const int32_t* tokens, uint32_t tc){
    auto* s=static_cast<QwenState*>(ctx);
    if(!s||!s->model_loaded||!tokens||tc==0){
        const char* e="{\"error\":\"not_ready\"}";
        auto* o=new char[strlen(e)+1];strcpy(o,e);return o;
    }
    std::vector<float> logits;
    s->kv_pos=0;
    for(uint32_t i=0;i<tc&&s->kv_pos<QwenState::MAX_SEQ;i++){
        if(!forward_token(s,tokens[i],s->kv_pos)){break;}
        s->kv_pos++;
    }
    cpu_lm_head(s,logits);
    int best=0;float bv=logits[0];
    for(int i=1;i<(int)logits.size();i++)if(logits[i]>bv){bv=logits[i];best=i;}
    std::ostringstream r;
    r<<"{\"token\":"<<best<<",\"logit\":"<<bv<<",\"token_count\":"<<tc<<"}";
    std::string js=r.str();
    auto* o=new char[js.size()+1];memcpy(o,js.data(),js.size());o[js.size()]=0;
    return o;
}

QWEN_INFER_API char* qw_sample(void* ctx, const int32_t* tokens, uint32_t tc, int k){
    auto* s=static_cast<QwenState*>(ctx);
    if(!s||!s->model_loaded||!tokens||tc==0){
        std::ostringstream r;
        r<<"{\"token\":0,\"logit\":0.0,\"k\":"<<k<<",\"status\":\"not_loaded\"}";
        std::string js=r.str();
        auto* o=new char[js.size()+1];memcpy(o,js.data(),js.size());o[js.size()]=0;
        return o;
    }

    // Determine starting position
    // If tc <= kv_pos: new sequence (shorter than cached) — reset
    if(tc<=s->kv_pos){
        s->kv_pos=0;
        // Zero KV caches
        uint32_t nKV=s->cfg.num_kv_heads, hD=s->cfg.head_dim;
        uint32_t csz=nKV*QwenState::MAX_SEQ*hD;
        std::vector<float> z(csz,0.f);
        for(auto& b:s->kv_k) uploadF32(s->ctx,b,z.data(),csz);
        for(auto& b:s->kv_v) uploadF32(s->ctx,b,z.data(),csz);
    }

    // Process only new tokens (kv_pos to tc-1)
    for(uint32_t i=s->kv_pos;i<tc&&i<QwenState::MAX_SEQ;i++){
        if(!forward_token(s,tokens[i],i)) break;
        s->kv_pos=i+1;
    }

    std::vector<float> logits;
    cpu_lm_head(s,logits);

    if(k<=0) k=40;
    int32_t next=topk_sample(logits,k);
    float   logv=logits[next];

    std::ostringstream r;
    r<<"{\"token\":"<<next<<",\"logit\":"<<logv<<",\"k\":"<<k<<"}";
    std::string js=r.str();
    auto* o=new char[js.size()+1];memcpy(o,js.data(),js.size());o[js.size()]=0;
    return o;
}

QWEN_INFER_API char* qw_get_config(void* ctx){
    auto* s=static_cast<QwenState*>(ctx);
    if(!s){auto* o=new char[5];strcpy(o,"null");return o;}
    std::string js=s->configJson();
    auto* o=new char[js.size()+1];memcpy(o,js.data(),js.size());o[js.size()]=0;
    return o;
}

QWEN_INFER_API int  qw_probe(){ return 1; }
QWEN_INFER_API void qw_free_string(char* s){ delete[] s; }

// Tokenize text → JSON array of token IDs: "[1,2,3]"
QWEN_INFER_API char* qw_tokenize(void* handle, const char* text){
    auto* s=static_cast<QwenState*>(handle);
    if(!s||!text||s->id_to_tok.empty()) return nullptr;
    auto ids=tok_encode(s,text);
    std::ostringstream os; os<<'[';
    for(size_t i=0;i<ids.size();i++){ if(i) os<<','; os<<ids[i]; }
    os<<']';
    std::string js=os.str();
    auto* o=new char[js.size()+1]; memcpy(o,js.data(),js.size()); o[js.size()]=0;
    return o;
}

// Detokenize JSON array of token IDs → JSON: {"content":"..."}
QWEN_INFER_API char* qw_detokenize(void* handle, const char* tokens_json){
    auto* s=static_cast<QwenState*>(handle);
    if(!s||!tokens_json||s->id_to_tok.empty()) return nullptr;
    // Parse simple JSON array of integers
    std::vector<uint32_t> ids;
    const char* p=tokens_json;
    while(*p&&*p!='[') p++;
    if(*p=='[') p++;
    while(*p&&*p!=']'){
        while(*p==' '||*p==',') p++;
        if(*p==']') break;
        char* end; uint32_t v=(uint32_t)strtoul(p,&end,10);
        if(end==p) break;
        ids.push_back(v); p=end;
    }
    std::string text=tok_decode(s,ids);
    // Build JSON: {"content":"<escaped>"}
    std::ostringstream os; os<<"{\"content\":\"";
    for(char c:text){
        if(c=='"')  os<<"\\\"";
        else if(c=='\\') os<<"\\\\";
        else if(c=='\n') os<<"\\n";
        else if(c=='\r') os<<"\\r";
        else os<<c;
    }
    os<<"\"}";
    std::string js=os.str();
    auto* o=new char[js.size()+1]; memcpy(o,js.data(),js.size()); o[js.size()]=0;
    return o;
}

} // extern "C"
