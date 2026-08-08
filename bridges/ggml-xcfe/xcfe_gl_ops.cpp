// xcfe_gl_ops.cpp — KHANARY GL tensor runtime: the op-generic GPU seam behind ggml-xcfe.
//
// Contract (see ggml-xcfe.cpp):
//   int xcfe_gl_run(const char* op, const float* const* inputs, int n_inputs,
//                   float* out, const int64_t* ne_out, int n_dims);
//     returns 0 = GPU handled; nonzero = caller falls back to its CPU reference.
//
// Driver: wgpu_native (WebGPU C API) with the GL backend preferred, then D3D11 — the same
// wgpu_native-release.dll the project's WGSL pipeline already verified on this rig
// (OpenGL fallback, 10 ops PASS). Loaded lazily so this DLL has zero link dependencies.
//
// Ops: mul_mat, get_rows, norm, rms_norm, gelu, gelu_quick, silu, relu, tanh, sigmoid,
//      add, sub, mul, soft_max, rope, concat, cpy.
//
// Every kernel uses one fixed binding layout:
//   @binding(0) uniform  Params     (dims + op scalars, 64 bytes, std140-ish)
//   @binding(1) storage  in0
//   @binding(2) storage  in1        (binary ops / weights / mask / positions; dummy for unary)
//   @binding(3) storage  out        (read_write)
#include "webgpu.h"
#include "wgpu.h"

#include <windows.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <map>

// ---------------------------------------------------------------------------------------------
// wgpu C API function pointers (loaded from wgpu_native-release.dll at first use)
// ---------------------------------------------------------------------------------------------
#define WGPU_FN(name) decltype(&::name) name = nullptr

static struct WgpuApi {
    WGPU_FN(wgpuCreateInstance);
    WGPU_FN(wgpuInstanceRequestAdapter);
    WGPU_FN(wgpuInstanceProcessEvents);
    WGPU_FN(wgpuInstanceRelease);
    WGPU_FN(wgpuAdapterRequestDevice);
    WGPU_FN(wgpuAdapterRelease);
    WGPU_FN(wgpuDeviceCreateShaderModule);
    WGPU_FN(wgpuDeviceCreateBuffer);
    WGPU_FN(wgpuDeviceCreateBindGroupLayout);
    WGPU_FN(wgpuDeviceCreatePipelineLayout);
    WGPU_FN(wgpuDeviceCreateBindGroup);
    WGPU_FN(wgpuDeviceCreateComputePipeline);
    WGPU_FN(wgpuDeviceCreateCommandEncoder);
    WGPU_FN(wgpuDeviceGetQueue);
    WGPU_FN(wgpuDevicePoll);
    WGPU_FN(wgpuDeviceRelease);
    WGPU_FN(wgpuQueueWriteBuffer);
    WGPU_FN(wgpuQueueSubmit);
    WGPU_FN(wgpuQueueRelease);
    WGPU_FN(wgpuCommandEncoderBeginComputePass);
    WGPU_FN(wgpuCommandEncoderCopyBufferToBuffer);
    WGPU_FN(wgpuCommandEncoderFinish);
    WGPU_FN(wgpuCommandEncoderRelease);
    WGPU_FN(wgpuComputePassEncoderSetPipeline);
    WGPU_FN(wgpuComputePassEncoderSetBindGroup);
    WGPU_FN(wgpuComputePassEncoderDispatchWorkgroups);
    WGPU_FN(wgpuComputePassEncoderEnd);
    WGPU_FN(wgpuComputePassEncoderRelease);
    WGPU_FN(wgpuBufferMapAsync);
    WGPU_FN(wgpuBufferGetMappedRange);
    WGPU_FN(wgpuBufferUnmap);
    WGPU_FN(wgpuBufferRelease);
    WGPU_FN(wgpuCommandBufferRelease);
    WGPU_FN(wgpuShaderModuleRelease);
    WGPU_FN(wgpuBindGroupLayoutRelease);
    WGPU_FN(wgpuPipelineLayoutRelease);
    WGPU_FN(wgpuBindGroupRelease);
    WGPU_FN(wgpuComputePipelineRelease);
} wg;

static bool wgpu_load() {
    static bool tried = false;
    static bool ok = false;
    if (tried) return ok;
    tried = true;

    const char * env = getenv("WGPU_NATIVE_DLL");
    HMODULE h = LoadLibraryA(env ? env : "wgpu_native-release.dll");
    if (!h) h = LoadLibraryA("wgpu_native.dll");
    if (!h) {
        fprintf(stderr, "[xcfe_gl_ops] wgpu_native.dll not found — GPU tensor ops disabled\n");
        return false;
    }
#define LOAD(fn) do { wg.fn = (decltype(&::fn)) GetProcAddress(h, #fn); if (!wg.fn) { fprintf(stderr, "[xcfe_gl_ops] missing export %s\n", #fn); return false; } } while (0)
    LOAD(wgpuCreateInstance);
    LOAD(wgpuInstanceRequestAdapter);
    LOAD(wgpuInstanceProcessEvents);
    LOAD(wgpuInstanceRelease);
    LOAD(wgpuAdapterRequestDevice);
    LOAD(wgpuAdapterRelease);
    LOAD(wgpuDeviceCreateShaderModule);
    LOAD(wgpuDeviceCreateBuffer);
    LOAD(wgpuDeviceCreateBindGroupLayout);
    LOAD(wgpuDeviceCreatePipelineLayout);
    LOAD(wgpuDeviceCreateBindGroup);
    LOAD(wgpuDeviceCreateComputePipeline);
    LOAD(wgpuDeviceCreateCommandEncoder);
    LOAD(wgpuDeviceGetQueue);
    LOAD(wgpuDevicePoll);
    LOAD(wgpuDeviceRelease);
    LOAD(wgpuQueueWriteBuffer);
    LOAD(wgpuQueueSubmit);
    LOAD(wgpuQueueRelease);
    LOAD(wgpuCommandEncoderBeginComputePass);
    LOAD(wgpuCommandEncoderCopyBufferToBuffer);
    LOAD(wgpuCommandEncoderFinish);
    LOAD(wgpuCommandEncoderRelease);
    LOAD(wgpuComputePassEncoderSetPipeline);
    LOAD(wgpuComputePassEncoderSetBindGroup);
    LOAD(wgpuComputePassEncoderDispatchWorkgroups);
    LOAD(wgpuComputePassEncoderEnd);
    LOAD(wgpuComputePassEncoderRelease);
    LOAD(wgpuBufferMapAsync);
    LOAD(wgpuBufferGetMappedRange);
    LOAD(wgpuBufferUnmap);
    LOAD(wgpuBufferRelease);
    LOAD(wgpuCommandBufferRelease);
    LOAD(wgpuShaderModuleRelease);
    LOAD(wgpuBindGroupLayoutRelease);
    LOAD(wgpuPipelineLayoutRelease);
    LOAD(wgpuBindGroupRelease);
    LOAD(wgpuComputePipelineRelease);
#undef LOAD
    ok = true;
    return true;
}

// ---------------------------------------------------------------------------------------------
// Device state + pipeline cache
// ---------------------------------------------------------------------------------------------
static WGPUInstance      g_instance  = nullptr;
static WGPUAdapter       g_adapter   = nullptr;
static WGPUDevice        g_device    = nullptr;
static WGPUQueue         g_queue     = nullptr;
static WGPUBindGroupLayout g_bgl     = nullptr;
static WGPUPipelineLayout  g_pl      = nullptr;
static std::map<std::string, WGPUComputePipeline> g_pipelines;

static const char * g_device_desc = "uninitialized";

static WGPUStringView sv(const char * s) {
    WGPUStringView v;
    v.data = s;
    v.length = s ? strlen(s) : 0;
    return v;
}

static WGPUInstanceBackend g_backend_flags = WGPUInstanceBackend_GL;

static void on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                       WGPUStringView message, void * u1, void * u2) {
    if (status == WGPURequestAdapterStatus_Success && adapter) {
        g_adapter = adapter;
    } else {
        fprintf(stderr, "[xcfe_gl_ops] adapter request failed: %.*s\n", (int) message.length, message.data ? message.data : "");
    }
}

static void on_device(WGPURequestDeviceStatus status, WGPUDevice device,
                      WGPUStringView message, void * u1, void * u2) {
    if (status == WGPURequestDeviceStatus_Success && device) {
        g_device = device;
    } else {
        fprintf(stderr, "[xcfe_gl_ops] device request failed: %.*s\n", (int) message.length, message.data ? message.data : "");
    }
}

static void on_map(WGPUMapAsyncStatus status, WGPUStringView message, void * u1, void * u2) {
    volatile int * flag = (volatile int *) u1;
    *flag = status == WGPUMapAsyncStatus_Success ? 1 : -1;
}

static void on_uncaptured_error(WGPUDevice const * device, WGPUErrorType type, WGPUStringView message, void * u1, void * u2) {
    (void) device; (void) type; (void) u1; (void) u2;
    fprintf(stderr, "[xcfe_gl_ops] wgpu uncaptured error: %.*s\n", (int) message.length, message.data ? message.data : "");
}

// Try to bring up the device with the given instance-backend flags. Returns true on success.
static bool device_init_with(WGPUInstanceBackend backend_flags, WGPUBackendType adapter_backend) {
    g_backend_flags = backend_flags;

    WGPUInstanceExtras extras = {};
    extras.chain.sType = (WGPUSType) 0x00030006; // WGPUSType_InstanceExtras (native)
    extras.chain.next  = nullptr;
    extras.backends    = backend_flags;

    WGPUInstanceDescriptor idesc = {};
    idesc.nextInChain = &extras.chain;
    idesc.features    = {}; // WGPUInstanceCapabilities

    g_instance = wg.wgpuCreateInstance(&idesc);
    if (!g_instance) { fprintf(stderr, "[xcfe_gl_ops] wgpuCreateInstance failed\n"); return false; }

    WGPURequestAdapterOptions aopts = {};
    aopts.featureLevel        = WGPUFeatureLevel_Core;
    aopts.powerPreference     = WGPUPowerPreference_HighPerformance;
    aopts.forceFallbackAdapter = false;
    aopts.backendType         = adapter_backend;

    WGPURequestAdapterCallbackInfo acb = {};
    acb.mode     = WGPUCallbackMode_AllowProcessEvents;
    acb.callback = on_adapter;
    wg.wgpuInstanceRequestAdapter(g_instance, &aopts, acb);
    for (int i = 0; i < 1000 && !g_adapter; ++i) wg.wgpuInstanceProcessEvents(g_instance);
    if (!g_adapter) { fprintf(stderr, "[xcfe_gl_ops] no adapter (backend 0x%x)\n", (unsigned) backend_flags); return false; }

    WGPUDeviceDescriptor ddesc = {};
    ddesc.label = sv("xcfe-gl-device");
    ddesc.requiredFeatureCount = 0;
    ddesc.requiredFeatures = nullptr;
    ddesc.requiredLimits   = nullptr;
    ddesc.defaultQueue     = {};
    ddesc.deviceLostCallbackInfo      = {};
    WGPUUncapturedErrorCallbackInfo uerr = {};
    uerr.callback = on_uncaptured_error;
    ddesc.uncapturedErrorCallbackInfo = uerr;

    WGPURequestDeviceCallbackInfo dcb = {};
    dcb.mode     = WGPUCallbackMode_AllowProcessEvents;
    dcb.callback = on_device;
    wg.wgpuAdapterRequestDevice(g_adapter, &ddesc, dcb);
    for (int i = 0; i < 2000 && !g_device; ++i) wg.wgpuInstanceProcessEvents(g_instance);
    if (!g_device) { fprintf(stderr, "[xcfe_gl_ops] device request failed\n"); return false; }

    g_queue = wg.wgpuDeviceGetQueue(g_device);
    if (!g_queue) { fprintf(stderr, "[xcfe_gl_ops] no queue\n"); return false; }
    return true;
}

static bool gl_ensure_device() {
    if (g_device) return true;
    if (!wgpu_load()) return false;

    // The user's route: OpenGL first (covers the whole tensor set on this rig), then D3D11,
    // then D3D12, then let wgpu pick.
    if (device_init_with(WGPUInstanceBackend_GL, WGPUBackendType_OpenGL)) {
        g_device_desc = "OpenGL (wgpu_native GL backend)";
    } else if (device_init_with(WGPUInstanceBackend_DX11, WGPUBackendType_D3D11)) {
        g_device_desc = "D3D11 (wgpu_native)";
    } else if (device_init_with(WGPUInstanceBackend_DX12, WGPUBackendType_D3D12)) {
        g_device_desc = "D3D12 (wgpu_native)";
    } else {
        fprintf(stderr, "[xcfe_gl_ops] all backends failed — GPU tensor ops disabled\n");
        return false;
    }

    // Fixed bind group layout: uniform(0), in0(1), in1(2), out(3).
    WGPUBindGroupLayoutEntry entries[4] = {};
    entries[0].binding    = 0;
    entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;
    entries[0].buffer.minBindingSize = 64;
    entries[1].binding    = 1;
    entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[2].binding    = 2;
    entries[2].visibility = WGPUShaderStage_Compute;
    entries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[3].binding    = 3;
    entries[3].visibility = WGPUShaderStage_Compute;
    entries[3].buffer.type = WGPUBufferBindingType_Storage;

    WGPUBindGroupLayoutDescriptor bgl = {};
    bgl.label = sv("xcfe-gl-bgl");
    bgl.entryCount = 4;
    bgl.entries = entries;
    g_bgl = wg.wgpuDeviceCreateBindGroupLayout(g_device, &bgl);
    if (!g_bgl) { fprintf(stderr, "[xcfe_gl_ops] bind group layout failed\n"); return false; }

    WGPUPipelineLayoutDescriptor pl = {};
    pl.label = sv("xcfe-gl-pl");
    pl.bindGroupLayoutCount = 1;
    pl.bindGroupLayouts = &g_bgl;
    g_pl = wg.wgpuDeviceCreatePipelineLayout(g_device, &pl);
    if (!g_pl) { fprintf(stderr, "[xcfe_gl_ops] pipeline layout failed\n"); return false; }

    fprintf(stderr, "[xcfe_gl_ops] device up: %s\n", g_device_desc);
    return true;
}

// ---------------------------------------------------------------------------------------------
// WGSL kernels
// ---------------------------------------------------------------------------------------------
static const char * PARAMS_HEAD =
    "struct Params { ne0: u32, ne1: u32, ne2: u32, ne3: u32, n_dims: u32, dim: u32, aux: u32, pad: u32, eps: f32, scale: f32, freq_base: f32, freq_scale: f32 };\n"
    "@group(0) @binding(0) var<uniform> p : Params;\n"
    "@group(0) @binding(1) var<storage, read> in0 : array<f32>;\n"
    "@group(0) @binding(2) var<storage, read> in1 : array<f32>;\n"
    "@group(0) @binding(3) var<storage, read_write> out : array<f32>;\n";

// erf via Abramowitz-Stegun 7.1.26 (~1e-7), matches erff closely enough for gelu.
static const char * ERF_FN =
    "fn erf_approx(x: f32) -> f32 {\n"
    "  let ax = abs(x);\n"
    "  let t = 1.0 / (1.0 + 0.3275911 * ax);\n"
    "  let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;\n"
    "  let y = 1.0 - poly * exp(-ax * ax);\n"
    "  return select(-y, y, x >= 0.0);\n"
    "}\n";

static const char * WGSL_CPY =
    "@compute @workgroup_size(256)\n"
    "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
    "  let n = p.ne0 * p.ne1 * p.ne2 * p.ne3;\n"
    "  if (gid.x >= n) { return; }\n"
    "  out[gid.x] = in0[gid.x];\n"
    "}\n";

static const char * WGSL_UNARY_BODY_TPL =
    "@compute @workgroup_size(256)\n"
    "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
    "  let n = p.ne0 * p.ne1 * p.ne2 * p.ne3;\n"
    "  if (gid.x >= n) { return; }\n"
    "  let x = in0[gid.x];\n"
    "  out[gid.x] = %EXPR%;\n"
    "}\n";

static const char * WGSL_BINARY_BODY_TPL =
    "@compute @workgroup_size(256)\n"
    "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
    "  let n = p.ne0 * p.ne1 * p.ne2 * p.ne3;\n"
    "  if (gid.x >= n) { return; }\n"
    "  out[gid.x] = %EXPR%;\n"
    "}\n";

static const char * WGSL_NORM =
    "@compute @workgroup_size(256)\n"
    "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
    "  let rows = p.ne1 * p.ne2 * p.ne3;\n"
    "  if (gid.x >= rows) { return; }\n"
    "  let base = gid.x * p.ne0;\n"
    "  var sum = 0.0;\n"
    "  for (var j = 0u; j < p.ne0; j++) { let v = in0[base + j]; sum = sum + v * v; }\n"
    "  let inv = 1.0 / sqrt(sum / f32(p.ne0) + p.eps);\n"
    "  for (var j = 0u; j < p.ne0; j++) {\n"
    "    let w = select(1.0, in1[base + j], p.aux != 0u);\n"
    "    out[base + j] = in0[base + j] * inv * w;\n"
    "  }\n"
    "}\n";

static const char * WGSL_RMS_NORM =
    "@compute @workgroup_size(256)\n"
    "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
    "  let rows = p.ne1 * p.ne2 * p.ne3;\n"
    "  if (gid.x >= rows) { return; }\n"
    "  let base = gid.x * p.ne0;\n"
    "  var sum = 0.0;\n"
    "  for (var j = 0u; j < p.ne0; j++) { let v = in0[base + j]; sum = sum + v * v; }\n"
    "  let inv = 1.0 / sqrt(sum / f32(p.ne0) + p.eps);\n"
    "  for (var j = 0u; j < p.ne0; j++) {\n"
    "    let w = select(1.0, in1[j], p.aux != 0u);\n"
    "    out[base + j] = in0[base + j] * inv * w;\n"
    "  }\n"
    "}\n";

static const char * WGSL_SOFT_MAX =
    "@compute @workgroup_size(256)\n"
    "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
    "  let rows = p.ne1 * p.ne2 * p.ne3;\n"
    "  if (gid.x >= rows) { return; }\n"
    "  let base = gid.x * p.ne0;\n"
    "  var mx = -1000000000.0;\n"
    "  for (var j = 0u; j < p.ne0; j++) {\n"
    "    let m = select(0.0, in1[base + j], p.aux != 0u);\n"
    "    let v = in0[base + j] * p.scale + m;\n"
    "    mx = max(mx, v);\n"
    "  }\n"
    "  var sum = 0.0;\n"
    "  for (var j = 0u; j < p.ne0; j++) {\n"
    "    let m = select(0.0, in1[base + j], p.aux != 0u);\n"
    "    let v = in0[base + j] * p.scale + m;\n"
    "    let e = exp(v - mx);\n"
    "    out[base + j] = e;\n"
    "    sum = sum + e;\n"
    "  }\n"
    "  for (var j = 0u; j < p.ne0; j++) { out[base + j] = out[base + j] / sum; }\n"
    "}\n";

static const char * WGSL_GET_ROWS =
    "@compute @workgroup_size(256)\n"
    "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
    "  let n = p.ne0 * p.ne1;\n"
    "  if (gid.x >= n) { return; }\n"
    "  let c = gid.x % p.ne0;\n"
    "  let r = gid.x / p.ne0;\n"
    "  let idx = u32(in1[r]);\n"
    "  out[gid.x] = select(0.0, in0[idx * p.ne0 + c], idx < p.pad);\n"
    "}\n";

static const char * WGSL_ROPE =
    "@compute @workgroup_size(256)\n"
    "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
    "  let rows = p.ne1 * p.ne2 * p.ne3;\n"
    "  if (gid.x >= rows) { return; }\n"
    "  let i1 = gid.x % p.ne1;\n"
    "  let pos = select(i1, u32(in1[i1]), p.aux != 0u);\n"
    "  let base = gid.x * p.ne0;\n"
    "  var j = 0u;\n"
    "  for (; j + 1u < p.n_dims; j = j + 2u) {\n"
    "    let theta = f32(pos) * p.freq_scale * pow(p.freq_base, -2.0 * f32(j / 2u) / f32(p.n_dims));\n"
    "    let c = cos(theta); let s = sin(theta);\n"
    "    let x0 = in0[base + j]; let x1 = in0[base + j + 1u];\n"
    "    out[base + j] = x0 * c - x1 * s;\n"
    "    out[base + j + 1u] = x0 * s + x1 * c;\n"
    "  }\n"
    "  for (; j < p.ne0; j++) { out[base + j] = in0[base + j]; }\n"
    "}\n";

static const char * WGSL_CONCAT =
    "@compute @workgroup_size(256)\n"
    "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
    "  let ne0a = p.n_dims; let ne0b = p.ne0 - ne0a;\n"
    "  let ne1a = p.pad;    let ne1b = p.ne1 - ne1a;\n"
    "  let rows = ne1a + ne1b;\n"
    "  let cols = ne0a + ne0b;\n"
    "  if (gid.x >= rows * cols) { return; }\n"
    "  let r = gid.x / cols; let c = gid.x % cols;\n"
    "  if (p.dim == 1u) {\n"
    "    if (r < ne1a) { out[gid.x] = in0[r * ne0a + c]; }\n"
    "    else          { out[gid.x] = in1[(r - ne1a) * ne0b + c]; }\n"
    "  } else {\n"
    "    if (c < ne0a) { out[gid.x] = select(0.0, in0[r * ne0a + c], r < ne1a); }\n"
    "    else          { out[gid.x] = select(0.0, in1[r * ne0b + (c - ne0a)], r >= ne1a); }\n"
    "  }\n"
    "}\n";

static const char * WGSL_MUL_MAT =
    "@compute @workgroup_size(16, 16)\n"
    "fn main(@builtin(global_invocation_id) dtid : vec3<u32>) {\n"
    "  let M = p.ne1; let N = p.ne0; let K = p.ne2;\n"
    "  let row = dtid.y; let col = dtid.x;\n"
    "  var acc = 0.0;\n"
    "  for (var k = 0u; k < K; k = k + 1u) {\n"
    "    let av = select(0.0, in0[row * K + k], row < M && k < K);\n"
    "    let bv = select(0.0, in1[col * K + k], col < N && k < K);\n"
    "    acc = acc + av * bv;\n"
    "  }\n"
    "  if (row < M && col < N) { out[row * N + col] = acc; }\n"
    "}\n";

// Simple per-element unary kernels (expr substituted into the template).
static std::string unary_shader(const char * expr) {
    std::string s = PARAMS_HEAD;
    s += WGSL_UNARY_BODY_TPL;
    size_t at = s.find("%EXPR%");
    s.replace(at, 6, expr);
    return s;
}

static std::string binary_shader(const char * expr) {
    std::string s = PARAMS_HEAD;
    s += WGSL_BINARY_BODY_TPL;
    size_t at = s.find("%EXPR%");
    s.replace(at, 6, expr);
    return s;
}

static std::map<std::string, std::string> build_shaders() {
    std::map<std::string, std::string> m;
    m["cpy"]        = std::string(PARAMS_HEAD) + WGSL_CPY;
    m["gelu"]       = std::string(PARAMS_HEAD) + ERF_FN + unary_shader("0.5 * x * (1.0 + erf_approx(x * 0.70710678118654752440))");
    m["gelu_quick"] = std::string(PARAMS_HEAD) + unary_shader("x / (1.0 + exp(-1.702 * x))");
    m["silu"]       = std::string(PARAMS_HEAD) + unary_shader("x / (1.0 + exp(-x))");
    m["relu"]       = std::string(PARAMS_HEAD) + unary_shader("max(x, 0.0)");
    m["tanh"]       = std::string(PARAMS_HEAD) + unary_shader("tanh(x)");
    m["sigmoid"]    = std::string(PARAMS_HEAD) + unary_shader("1.0 / (1.0 + exp(-x))");
    m["add"]        = std::string(PARAMS_HEAD) + binary_shader("in0[gid.x] + in1[gid.x]");
    m["sub"]        = std::string(PARAMS_HEAD) + binary_shader("in0[gid.x] - in1[gid.x]");
    m["mul"]        = std::string(PARAMS_HEAD) + binary_shader("in0[gid.x] * in1[gid.x]");
    m["norm"]       = std::string(PARAMS_HEAD) + WGSL_NORM;
    m["rms_norm"]   = std::string(PARAMS_HEAD) + WGSL_RMS_NORM;
    m["soft_max"]   = std::string(PARAMS_HEAD) + WGSL_SOFT_MAX;
    m["get_rows"]   = std::string(PARAMS_HEAD) + WGSL_GET_ROWS;
    m["rope"]       = std::string(PARAMS_HEAD) + WGSL_ROPE;
    m["concat"]     = std::string(PARAMS_HEAD) + WGSL_CONCAT;
    m["mul_mat"]    = std::string(PARAMS_HEAD) + WGSL_MUL_MAT;
    return m;
}

// ---------------------------------------------------------------------------------------------
// Per-op pipeline creation
// ---------------------------------------------------------------------------------------------
static WGPUComputePipeline op_pipeline(const std::string & op) {
    auto it = g_pipelines.find(op);
    if (it != g_pipelines.end()) return it->second;

    static const std::map<std::string, std::string> shaders = build_shaders();
    auto sit = shaders.find(op);
    if (sit == shaders.end()) return nullptr;

    WGPUShaderSourceWGSL wgsl = {};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code        = sv(sit->second.c_str());

    WGPUShaderModuleDescriptor smd = {};
    smd.nextInChain = &wgsl.chain;
    smd.label       = sv(op.c_str());
    WGPUShaderModule module = wg.wgpuDeviceCreateShaderModule(g_device, &smd);
    if (!module) { fprintf(stderr, "[xcfe_gl_ops] shader compile failed: %s\n", op.c_str()); return nullptr; }

    WGPUProgrammableStageDescriptor stage = {};
    stage.module      = module;
    stage.entryPoint  = sv("main");
    stage.constantCount = 0;

    WGPUComputePipelineDescriptor cpd = {};
    cpd.label   = sv(op.c_str());
    cpd.layout  = g_pl;
    cpd.compute = stage;
    WGPUComputePipeline pipeline = wg.wgpuDeviceCreateComputePipeline(g_device, &cpd);
    wg.wgpuShaderModuleRelease(module);
    if (!pipeline) { fprintf(stderr, "[xcfe_gl_ops] pipeline failed: %s\n", op.c_str()); return nullptr; }

    g_pipelines[op] = pipeline;
    return pipeline;
}

// ---------------------------------------------------------------------------------------------
// The seam entry point
// ---------------------------------------------------------------------------------------------
struct Params {
    uint32_t ne0, ne1, ne2, ne3;
    uint32_t n_dims, dim, aux, pad;
    float    eps, scale, freq_base, freq_scale;
};

static uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

extern "C" __declspec(dllexport) int xcfe_gl_run(const char * op, const float * const * inputs,
                                                 int n_inputs, float * out,
                                                 const int64_t * ne_out, int n_dims,
                                                 const float * params, int n_params) {
    if (!op || !inputs || n_inputs < 1 || !out) return 1;
    if (!gl_ensure_device()) return 1;

    const uint64_t ne0 = ne_out && n_dims > 0 ? (uint64_t) ne_out[0] : 1;
    const uint64_t ne1 = ne_out && n_dims > 1 ? (uint64_t) ne_out[1] : 1;
    const uint64_t ne2 = ne_out && n_dims > 2 ? (uint64_t) ne_out[2] : 1;
    const uint64_t ne3 = ne_out && n_dims > 3 ? (uint64_t) ne_out[3] : 1;

    WGPUComputePipeline pipeline = op_pipeline(op);
    if (!pipeline) return 1;

    // Params (op-specific wiring matches ggml's op_params usage in the bridge).
    Params p = {};
    p.ne0 = (uint32_t) ne0; p.ne1 = (uint32_t) ne1; p.ne2 = (uint32_t) ne2; p.ne3 = (uint32_t) ne3;
    p.eps = 1e-5f; p.scale = 1.0f; p.freq_base = 10000.0f; p.freq_scale = 1.0f;
    p.aux = (uint32_t) (n_inputs >= 2 ? 1 : 0); // weights/mask/positions present?

    // Contract params -> Params fields (per-op).
    if (params && n_params > 0) {
        if (strcmp(op, "mul_mat") == 0) {
            p.ne2 = (uint32_t) params[0]; // K = reduction dim
        } else if (strcmp(op, "norm") == 0 || strcmp(op, "rms_norm") == 0) {
            p.eps = params[0];
        } else if (strcmp(op, "soft_max") == 0) {
            p.scale = params[0];
        } else if (strcmp(op, "rope") == 0 && n_params >= 3) {
            p.n_dims     = (uint32_t) params[0];
            p.freq_base  = params[1];
            p.freq_scale = params[2];
        } else if (strcmp(op, "concat") == 0 && n_params >= 3) {
            p.dim  = (uint32_t) params[0]; // concat dim
            p.n_dims = (uint32_t) params[1]; // ne0a
            p.pad    = (uint32_t) params[2]; // ne1a
        } else if (strcmp(op, "get_rows") == 0) {
            p.pad = (uint32_t) params[0]; // table row count
        }
    }

    const uint64_t out_elems = ne0 * ne1 * ne2 * ne3;
    const uint64_t out_bytes = round_up(out_elems * sizeof(float), 16);
    uint64_t in0_bytes = out_bytes, in1_bytes = 16;
    if (strcmp(op, "mul_mat") == 0) {
        // in0 = src1 [M,K], in1 = src0 [N,K]; M=ne1, N=ne0, K=ne2 (set by bridge)
        p.aux = 0;
        in0_bytes = round_up((uint64_t) p.ne1 * p.ne2 * sizeof(float), 16);
        in1_bytes = round_up((uint64_t) p.ne0 * p.ne2 * sizeof(float), 16);
    } else if (strcmp(op, "get_rows") == 0) {
        // in0 = table [rows, ne0]; table rows come via params (p.pad)
        in0_bytes = round_up((uint64_t) p.pad * ne0 * sizeof(float), 16);
        in1_bytes = round_up(ne1 * sizeof(float), 16);
    } else if (strcmp(op, "concat") == 0) {
        // ne0a=p.n_dims, ne1a=p.pad; ne0b = ne0-ne0a, ne1b = ne1-ne1a
        const uint64_t ne0a = p.n_dims, ne1a = p.pad;
        const uint64_t ne0b = ne0 > ne0a ? ne0 - ne0a : 0;
        const uint64_t ne1b = ne1 > ne1a ? ne1 - ne1a : 0;
        in0_bytes = round_up(ne0a * ne1a * sizeof(float), 16);
        in1_bytes = round_up(ne0b * ne1b * sizeof(float), 16);
    } else if (strcmp(op, "norm") == 0 || strcmp(op, "rms_norm") == 0 || strcmp(op, "soft_max") == 0) {
        in0_bytes = out_bytes;
        in1_bytes = n_inputs >= 2 ? out_bytes : 16; // weights/mask
    } else if (strcmp(op, "rope") == 0) {
        in0_bytes = out_bytes;
        in1_bytes = n_inputs >= 2 ? round_up(ne1 * sizeof(float), 16) : 16; // positions per row
    } else if (strcmp(op, "add") == 0 || strcmp(op, "sub") == 0 || strcmp(op, "mul") == 0) {
        in0_bytes = out_bytes;
        in1_bytes = out_bytes;
    } else {
        in0_bytes = out_bytes;
        in1_bytes = 16;
    }

    // --- create buffers ------------------------------------------------------
    WGPUBufferDescriptor bd = {};
    bd.label = sv("xcfe-in0");
    bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage;
    bd.size   = in0_bytes;
    bd.mappedAtCreation = false;
    WGPUBuffer buf0 = wg.wgpuDeviceCreateBuffer(g_device, &bd);
    if (!buf0) return 1;

    bd.label = sv("xcfe-in1");
    bd.size  = in1_bytes;
    WGPUBuffer buf1 = wg.wgpuDeviceCreateBuffer(g_device, &bd);
    if (!buf1) { wg.wgpuBufferRelease(buf0); return 1; }

    bd.label = sv("xcfe-params");
    bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
    bd.size  = 64;
    WGPUBuffer bufP = wg.wgpuDeviceCreateBuffer(g_device, &bd);
    if (!bufP) { wg.wgpuBufferRelease(buf0); wg.wgpuBufferRelease(buf1); return 1; }

    bd.label = sv("xcfe-out");
    bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
    bd.size  = out_bytes;
    WGPUBuffer bufO = wg.wgpuDeviceCreateBuffer(g_device, &bd);
    if (!bufO) { wg.wgpuBufferRelease(buf0); wg.wgpuBufferRelease(buf1); wg.wgpuBufferRelease(bufP); return 1; }

    // Separate readback buffer — wgpu forbids MAP_READ combined with STORAGE.
    bd.label = sv("xcfe-readback");
    bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    bd.size  = out_bytes;
    WGPUBuffer bufR = wg.wgpuDeviceCreateBuffer(g_device, &bd);
    if (!bufR) { wg.wgpuBufferRelease(buf0); wg.wgpuBufferRelease(buf1); wg.wgpuBufferRelease(bufP); wg.wgpuBufferRelease(bufO); return 1; }

    // --- upload --------------------------------------------------------------
    wg.wgpuQueueWriteBuffer(g_queue, buf0, 0, inputs[0], in0_bytes);
    wg.wgpuQueueWriteBuffer(g_queue, buf1, 0, n_inputs >= 2 ? inputs[1] : (const float *) &p, in1_bytes);
    wg.wgpuQueueWriteBuffer(g_queue, bufP, 0, &p, 64);

    // --- bind group + dispatch ----------------------------------------------
    WGPUBindGroupEntry entries[4] = {};
    entries[0].binding = 0; entries[0].buffer = bufP; entries[0].offset = 0; entries[0].size = 64;
    entries[1].binding = 1; entries[1].buffer = buf0; entries[1].offset = 0; entries[1].size = in0_bytes;
    entries[2].binding = 2; entries[2].buffer = buf1; entries[2].offset = 0; entries[2].size = in1_bytes;
    entries[3].binding = 3; entries[3].buffer = bufO; entries[3].offset = 0; entries[3].size = out_bytes;

    WGPUBindGroupDescriptor bgd = {};
    bgd.label = sv("xcfe-bg");
    bgd.layout = g_bgl;
    bgd.entryCount = 4;
    bgd.entries = entries;
    WGPUBindGroup bg = wg.wgpuDeviceCreateBindGroup(g_device, &bgd);
    if (!bg) { wg.wgpuBufferRelease(buf0); wg.wgpuBufferRelease(buf1); wg.wgpuBufferRelease(bufP); wg.wgpuBufferRelease(bufO); return 1; }

    WGPUCommandEncoderDescriptor ced = {};
    ced.label = sv("xcfe-enc");
    WGPUCommandEncoder enc = wg.wgpuDeviceCreateCommandEncoder(g_device, &ced);
    if (!enc) { wg.wgpuBindGroupRelease(bg); wg.wgpuBufferRelease(buf0); wg.wgpuBufferRelease(buf1); wg.wgpuBufferRelease(bufP); wg.wgpuBufferRelease(bufO); wg.wgpuBufferRelease(bufR); return 1; }

    WGPUComputePassDescriptor cpd = {};
    cpd.label = sv("xcfe-pass");
    WGPUComputePassEncoder pass = wg.wgpuCommandEncoderBeginComputePass(enc, &cpd);

    uint32_t gx = 1, gy = 1, gz = 1;
    if (strcmp(op, "mul_mat") == 0) {
        gx = (uint32_t) ((ne0 + 15) / 16);
        gy = (uint32_t) ((ne1 + 15) / 16);
    } else if (strcmp(op, "norm") == 0 || strcmp(op, "rms_norm") == 0 ||
               strcmp(op, "soft_max") == 0 || strcmp(op, "rope") == 0) {
        gx = (uint32_t) ((ne1 * ne2 * ne3 + 255) / 256);
    } else {
        gx = (uint32_t) ((out_elems + 255) / 256);
    }

    wg.wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wg.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wg.wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, gz);
    wg.wgpuComputePassEncoderEnd(pass);
    wg.wgpuComputePassEncoderRelease(pass);

    // Copy the compute output into the mappable readback buffer.
    wg.wgpuCommandEncoderCopyBufferToBuffer(enc, bufO, 0, bufR, 0, out_bytes);

    WGPUCommandBufferDescriptor cbd = {};
    cbd.label = sv("xcfe-cmdbuf");
    WGPUCommandBuffer cmd = wg.wgpuCommandEncoderFinish(enc, &cbd);
    wg.wgpuCommandEncoderRelease(enc);
    if (!cmd) { wg.wgpuBindGroupRelease(bg); wg.wgpuBufferRelease(buf0); wg.wgpuBufferRelease(buf1); wg.wgpuBufferRelease(bufP); wg.wgpuBufferRelease(bufO); wg.wgpuBufferRelease(bufR); return 1; }

    wg.wgpuQueueSubmit(g_queue, 1, &cmd);
    wg.wgpuCommandBufferRelease(cmd);

    // --- sync readback -------------------------------------------------------
    volatile int mapped = 0;
    WGPUBufferMapCallbackInfo mcb = {};
    mcb.mode     = WGPUCallbackMode_AllowProcessEvents;
    mcb.callback = on_map;
    mcb.userdata1 = (void *) &mapped;
    wg.wgpuBufferMapAsync(bufR, WGPUMapMode_Read, 0, out_bytes, mcb);
    for (int i = 0; i < 4000 && mapped == 0; ++i) wg.wgpuDevicePoll(g_device, true, nullptr);
    if (mapped <= 0) {
        fprintf(stderr, "[xcfe_gl_ops] map failed for %s\n", op);
        wg.wgpuBindGroupRelease(bg);
        wg.wgpuBufferRelease(buf0); wg.wgpuBufferRelease(buf1); wg.wgpuBufferRelease(bufP); wg.wgpuBufferRelease(bufO); wg.wgpuBufferRelease(bufR);
        return 1;
    }

    const void * range = wg.wgpuBufferGetMappedRange(bufR, 0, out_bytes);
    if (range) std::memcpy(out, range, (size_t) (out_elems * sizeof(float)));
    wg.wgpuBufferUnmap(bufR);

    wg.wgpuBindGroupRelease(bg);
    wg.wgpuBufferRelease(buf0);
    wg.wgpuBufferRelease(buf1);
    wg.wgpuBufferRelease(bufP);
    wg.wgpuBufferRelease(bufO);
    wg.wgpuBufferRelease(bufR);
    return range ? 0 : 1;
}
