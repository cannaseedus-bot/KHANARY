#!/usr/bin/env node
'use strict';

/*
 * KUHUL SERVER — NNC-K Stack Gateway
 * Micronaut Registry + kuhul_engine auto-start + MCP server
 *
 * Ports:
 *   8081 — this server (registry, MCP, engine proxy)
 *   17474 — kuhul_engine --serve (auto-started on boot)
 */

const fs      = require('fs');
const path    = require('path');
const http    = require('http');
const https   = require('https');
const net     = require('net');
const os      = require('os');
const crypto  = require('crypto');
const { spawn, execFile } = require('child_process');

// ── Driver DLL (khanary_driver.dll) — load if available ──────────────
let driverDll = null;
const DRIVER_DLL_PATH = path.join(__dirname, '..', '..', 'drivers', 'khanary_driver.dll');
try {
  if (fs.existsSync(DRIVER_DLL_PATH)) {
    const ffi = require('./ffi-shim');
    driverDll = ffi.Library(DRIVER_DLL_PATH, {
      kd_create:      ['pointer', ['string']],
      kd_load_tasks:  ['int',    ['pointer', 'string', 'pointer', 'int']],
      kd_plan:        ['string', ['pointer']],
      kd_run:         ['string', ['pointer']],
      kd_dispatch:    ['string', ['pointer', 'string']],
      kd_task_count:  ['int',    ['pointer']],
      kd_destroy:     ['void',   ['pointer']],
      kd_free_string: ['void',   ['string']],
    });
    console.log('[driver] khanary_driver.dll loaded');
  }
} catch (e) {
  console.log('[driver] DLL not available, using engine execFile fallback:', e.message);
}

// Driver instance (created on first use with default providers)
let _driverHandle = null;
function getDriverHandle() {
  if (!driverDll) return null;
  if (_driverHandle) return _driverHandle;
  const providers = JSON.stringify([
    { id: 'micronaut-coder',   library: 'bot.py', available: true },
    { id: 'micronaut-factory', library: 'bot.py', available: true },
    { id: 'micronaut-base',    library: 'bot.py', available: true },
    { id: 'json-runtime',      library: null,     available: true },
    { id: 'kuhul-engine',      library: null,     available: true },
    { id: 'kuhul-server',      library: null,     available: true },
    { id: 'ollama',            library: null,     available: true },
    { id: 'python-distil',     library: null,     available: true },
  ]);
  _driverHandle = driverDll.kd_create(providers);
  return _driverHandle;
}

function driverPlan(taskListJson) {
  const h = getDriverHandle();
  if (!h) return null;
  const errBuf = Buffer.alloc(1024);
  const ok = driverDll.kd_load_tasks(h, taskListJson, errBuf, errBuf.length);
  if (!ok) {
    const errStr = require('./ffi-shim').ref.readCString(errBuf, 0);
    return { error: errStr || 'driver_load_failed' };
  }
  const result = driverDll.kd_plan(h);
  try {
    return JSON.parse(result);
  } finally {
    driverDll.kd_free_string(result);
  }
}

function driverRun(taskListJson) {
  const h = getDriverHandle();
  if (!h) return null;
  const errBuf = Buffer.alloc(1024);
  const ok = driverDll.kd_load_tasks(h, taskListJson, errBuf, errBuf.length);
  if (!ok) {
    const errStr = require('./ffi-shim').ref.readCString(errBuf, 0);
    return { error: errStr || 'driver_load_failed' };
  }
  const result = driverDll.kd_run(h);
  try {
    return JSON.parse(result);
  } finally {
    driverDll.kd_free_string(result);
  }
}

function driverDispatch(taskJson) {
  const h = getDriverHandle();
  if (!h) return null;
  const result = driverDll.kd_dispatch(h, taskJson);
  try {
    return JSON.parse(result);
  } finally {
    driverDll.kd_free_string(result);
  }
}

// ── Glyph engine DLL (khanary_glyph_driver.dll) — load if available ──
let glyphDll = null;
const GLYPH_DLL_PATH = path.join(__dirname, '..', '..', 'drivers', 'khanary_glyph_driver.dll');
try {
  if (fs.existsSync(GLYPH_DLL_PATH)) {
    const ffi = require('./ffi-shim');
    glyphDll = ffi.Library(GLYPH_DLL_PATH, {
      kd_glyph_phase_count:      ['uint32', []],
      kd_glyph_get_phase:        ['int32',  ['uint32', 'pointer']],
      kd_glyph_dispatch_phase:   ['void',   ['uint32', 'string', 'pointer']],
      kd_glyph_process_sequence: ['string', ['pointer', 'uint32']],
      kd_glyph_lane_count:       ['uint32', []],
      kd_glyph_get_lane:         ['int32',  ['uint32', 'pointer']],
      kd_glyph_find_lane_by_opcode: ['int32', ['uint32', 'pointer']],
      kd_glyph_find_lane_by_ggml:   ['int32', ['string', 'pointer']],
      kd_glyph_dump_registry:    ['string', []],
      kd_glyph_total_entries:    ['uint32', []],
      kd_glyph_free_string:      ['void',   ['string']],
    });
    console.log('[glyph] khanary_glyph_driver.dll loaded (' +
      glyphDll.kd_glyph_phase_count() + ' phases, ' +
      glyphDll.kd_glyph_lane_count() + ' lanes)');
  }
} catch (e) {
  console.log('[glyph] DLL not available:', e.message);
}

function glyphPhaseCount() {
  return glyphDll ? glyphDll.kd_glyph_phase_count() : 0;
}

function glyphLaneCount() {
  return glyphDll ? glyphDll.kd_glyph_lane_count() : 0;
}

function glyphDispatchPhase(opcode, providerJson) {
  if (!glyphDll) return null;
  const resultBuf = Buffer.alloc(640);
  glyphDll.kd_glyph_dispatch_phase(opcode, providerJson || '', resultBuf);
  return {
    opcode:  resultBuf.readUInt32LE(0),
    name:    require('./ffi-shim').ref.readCString(resultBuf, 8),
    status:  require('./ffi-shim').ref.readCString(resultBuf, 40),
    detail:  require('./ffi-shim').ref.readCString(resultBuf, 72),
  };
}

function glyphDumpRegistry() {
  if (!glyphDll) return null;
  const json = glyphDll.kd_glyph_dump_registry();
  try { return JSON.parse(json); } finally { glyphDll.kd_glyph_free_string(json); }
}

// ── kuhul_engine driver DLL — load if available ────────────────────
let engineDll = null;
const ENGINE_DLL_PATH = path.join(__dirname, '..', '..', 'drivers', 'kuhul_engine_driver.dll');
try {
  if (fs.existsSync(ENGINE_DLL_PATH)) {
    const ffi = require('./ffi-shim');
    engineDll = ffi.Library(ENGINE_DLL_PATH, {
      ke_create:            ['pointer', ['string']],
      ke_destroy:           ['void',    ['pointer']],
      ke_load_model:        ['int',     ['pointer', 'string', 'pointer', 'int']],
      ke_load_dom:          ['int',     ['pointer', 'string', 'pointer', 'int']],
      ke_get_tools:         ['string',  ['pointer']],
      ke_chat:              ['string',  ['pointer', 'string']],
      ke_status:            ['string',  ['pointer']],
      ke_is_model_loaded:   ['int',     ['pointer']],
      ke_package_wwa:       ['int',     ['string', 'string', 'pointer', 'int']],
      ke_launch_wwa:        ['int',     ['string', 'pointer', 'int']],
      ke_wwa_runtime_available: ['int', []],
      ke_free_string:       ['void',    ['string']],
    });
    console.log('[engine] kuhul_engine_driver.dll loaded');
  }
} catch (e) { console.log('[engine] DLL not available:', e.message); }

// ── gl_infer driver DLL — load if available ────────────────────────
let glDll = null;
const GL_DLL_PATH = path.join(__dirname, '..', '..', 'drivers', 'gl_infer_driver.dll');
try {
  if (fs.existsSync(GL_DLL_PATH)) {
    const ffi = require('./ffi-shim');
    glDll = ffi.Library(GL_DLL_PATH, {
      gli_create:     ['pointer', ['string']],
      gli_destroy:    ['void',    ['pointer']],
      gli_load_model: ['int',     ['pointer', 'string', 'pointer', 'int']],
      gli_forward:    ['string',  ['pointer', 'pointer', 'uint32']],
      gli_sample:     ['string',  ['pointer', 'pointer', 'uint32', 'int']],
      gli_probe:      ['int',     []],
      gli_gpu_info:   ['string',  []],
      gli_free_string:['void',    ['string']],
    });
    console.log('[gl] gl_infer_driver.dll loaded (probe=' + glDll.gli_probe() + ')');
  }
} catch (e) { console.log('[gl] DLL not available:', e.message); }

// ── qwen_infer driver DLL — load if available ──────────────────────
let qwDll = null;
const QW_DLL_PATH = path.join(__dirname, '..', '..', 'drivers', 'qwen_infer_driver.dll');
try {
  if (fs.existsSync(QW_DLL_PATH)) {
    const ffi = require('./ffi-shim');
    qwDll = ffi.Library(QW_DLL_PATH, {
      qw_create:      ['pointer', ['string']],
      qw_destroy:     ['void',    ['pointer']],
      qw_load_model:  ['int',     ['pointer', 'string', 'pointer', 'int']],
      qw_forward:     ['string',  ['pointer', 'pointer', 'uint32']],
      qw_sample:      ['string',  ['pointer', 'pointer', 'uint32', 'int']],
      qw_probe:       ['int',     []],
      qw_free_string: ['void',    ['string']],
    });
    console.log('[qw] qwen_infer_driver.dll loaded (probe=' + qwDll.qw_probe() + ')');
  }
} catch (e) { console.log('[qw] DLL not available:', e.message); }

const pkg = require('./package.json');

// =============================================================================
// Paths — NNC-K Stack
// =============================================================================
const PROJECT_ROOT = path.resolve(__dirname, '..', '..');

// ── UI host → dist folder map ─────────────────────────────────────────────────
// Add one entry per UI project. The folder must contain a built index.html.
// New project: npm run build into its folder, add an entry here, add a
// Cloudflare tunnel route for the subdomain → localhost:8764.
const UI_DIST    = path.join(PROJECT_ROOT, 'studio-dist');
const UI_BY_HOST = { 'kuhul.dev': UI_DIST, 'localhost': UI_DIST, '127.0.0.1': UI_DIST };
const UI_DEFAULT = UI_DIST;
const WEBX_ROOT_CANDIDATES = [
  path.join(PROJECT_ROOT, 'dist', 'v3.5.0-WebX'),
  'C:\\Users\\canna\\.NNC-K\\bin\\v3.5.0-WebX',
];
const NNC_K_WEBX = WEBX_ROOT_CANDIDATES.find((root) => (
  fs.existsSync(path.join(root, 'build', 'bin', 'Release', 'kuhul_engine.exe'))
  || fs.existsSync(path.join(root, 'build-llama', 'bin', 'Release', 'kuhul_engine.exe'))
)) || WEBX_ROOT_CANDIDATES[0];
const NNC_K_BIN_CANDIDATES = [
  path.join(NNC_K_WEBX, 'build', 'bin', 'Release'),
  path.join(NNC_K_WEBX, 'build-llama', 'bin', 'Release'),
];
const NNC_K_BIN = NNC_K_BIN_CANDIDATES.find((dir) => fs.existsSync(path.join(dir, 'kuhul_engine.exe')))
  || NNC_K_BIN_CANDIDATES[0];
const ENGINE_EXE  = path.join(NNC_K_BIN, 'kuhul_engine.exe');
const HELPER_EXE  = path.join(NNC_K_BIN, 'kuhul_task_helper.exe');
const JSON_RT_EXE = path.join(NNC_K_WEBX, 'bin', 'json_runtime.exe');
const JSON_RT_DIR = path.join(NNC_K_WEBX, 'bin', 'json-runtime');
const WWAHOSTEXE  = path.join(NNC_K_WEBX, 'bin', 'WWAHost.exe');
const SDK_PS1     = path.join(NNC_K_WEBX, 'native', 'semantic-kernel', 'MicrosoftSDK.ps1');
const CSHARP_SDK  = 'C:\\Users\\canna\\.NNC-K\\bin\\csharp-sdk';

// Engine HTTP API — port is auto-detected at startup (17xxx range is Windows-excluded on this machine)
const ENGINE_PORT_DEFAULT    = 17480;
const ENGINE_PORT_CANDIDATES = [14800, 14810, 15000, 13480, 12480, 11480, 10480, 9480];
const ENGINE_MODEL           = 'gpt-oss-20b-MXFP4.gguf';

// llama-server — proper llama.cpp server used as local teacher (kuhul_engine can't load GGUF
// because its binary doesn't call ggml_backend_load_all()). Gemma-3-1B-QAT fits in 1792 MB VRAM.
const LLAMA_SERVER_EXE  = 'C:\\Users\\canna\\_khanary_inspect\\khanary-llama-build\\llama.cpp\\build\\bin\\Release\\llama-server.exe';
const LLAMA_SERVER_DIR  = 'C:\\Users\\canna\\_khanary_inspect\\khanary-llama-build\\llama.cpp\\build\\bin\\Release';
const LLAMA_SERVER_PORT = 17481;
const LLAMA_SERVER_MODEL = 'C:\\Users\\canna\\.lmstudio\\models\\lmstudio-community\\gemma-3-1B-it-QAT-GGUF\\gemma-3-1B-it-QAT-Q4_0.gguf';
let llamaServerProcess  = null;
let _llamaServerReady   = false;

// ── Quantization-tier routing ─────────────────────────────────────────────────
// Tiers: fast=Q4 (speed), standard=Q8 (quality), quality=F16 (reasoning)
// Models live in QUANT_MODEL_DIR; naming convention: <model>_Q4.gguf / _Q8.gguf / _F16.gguf
const QUANT_MODEL_DIR = 'E:\\models\\khanary-gguf';

// Tier → filename suffix used by make_gguf.py
const QUANT_TIER_SUFFIX = { fast: 'Q4', standard: 'Q8', quality: 'F16' };

// Universal fallback per tier (from_zero base or gemma teacher).
// Paths update once GGUF conversion runs; server falls back gracefully if missing.
const QUANT_TIER_FALLBACK = {
  fast:     'C:\\Users\\canna\\.lmstudio\\models\\lmstudio-community\\gemma-3-1B-it-QAT-GGUF\\gemma-3-1B-it-QAT-Q4_0.gguf',
  standard: path.join(QUANT_MODEL_DIR, 'from_zero_v0.6_Q8.gguf'),
  quality:  path.join(QUANT_MODEL_DIR, 'from_zero_v0.6_F16.gguf'),
};

// Resolve the GGUF path for a given micronaut registry entry
function resolveQuantModel(mn) {
  const tier   = (mn && mn.quant_tier) || 'standard';
  const hint   = mn && mn.model_hint;   // e.g. "ts_coder_v0.1"
  const suffix = QUANT_TIER_SUFFIX[tier] || 'Q8';

  if (hint) {
    const candidate = path.join(QUANT_MODEL_DIR, hint + '_' + suffix + '.gguf');
    if (fs.existsSync(candidate)) return { path: candidate, tier, hint };
    // also try hint as literal filename
    const literal = path.join(QUANT_MODEL_DIR, hint);
    if (fs.existsSync(literal)) return { path: literal, tier, hint };
  }

  // Tier fallback
  const fb = QUANT_TIER_FALLBACK[tier];
  if (fb && fs.existsSync(fb)) return { path: fb, tier, hint: null };

  // Last resort: always-present gemma teacher model
  return { path: LLAMA_SERVER_MODEL, tier: 'fast', hint: null };
}

// Track which model the llama-server currently has loaded
let _activeLlamaModel = null;

// Hot-swap: kill server and restart with a different model file
async function _swapLlamaModel(modelPath) {
  if (_activeLlamaModel === modelPath && _llamaServerReady) return true;
  if (llamaServerProcess) {
    llamaServerProcess.kill('SIGTERM');
    llamaServerProcess = null;
    _llamaServerReady  = false;
    _activeLlamaModel  = null;
    await new Promise(r => setTimeout(r, 1500));
  }
  const ok = await _ensureLlamaServer(modelPath);
  if (ok) _activeLlamaModel = modelPath;
  return ok;
}

// Grammar files
const KUHUL_EBNF_EXCERPT = path.join(NNC_K_WEBX, 'kuhul', 'grammar', 'kuhul.ebnf');
const KUHUL_EBNF_FULL    = path.join(NNC_K_WEBX, 'grammar-ebnf', 'KUHUL-LLM.ebnf');
const GRAMMAR_VALIDATOR  = path.join(NNC_K_WEBX, 'kuhul', 'grammar', 'grammar-validator.js');
const EBNF_PARSER_JS     = path.join(NNC_K_WEBX, 'kuhul', 'grammar', 'ebnf-parser.js');

// This server — abstract port: 0 = OS assigns, port written to .port file
const REGISTRY_PORT  = Number(process.env.KUHUL_REGISTRY_PORT || 0);
const PORT_FILE      = path.join(__dirname, '.kuhul-server.port');

// =============================================================================
// Server State
// =============================================================================
const state = {
  version:    pkg.version,
  pid:        process.pid,
  startedAt:  new Date().toISOString(),
  cwd:        PROJECT_ROOT,
  enginePid:  null,
  engineUp:   false,
  enginePort: ENGINE_PORT_DEFAULT
};

let engineProcess = null;

// SSE clients subscribed to /micronauts/events
const sseClients     = new Set();
// MCP SSE clients subscribed to /mcp/sse
const mcpSseClients  = new Set();

const autoCreated = [];

// =============================================================================
// Boot Banner
// =============================================================================
console.log(`
+==========================================+
|  KUHUL SERVER v${pkg.version}
|  NNC-K Stack Gateway + MCP Server
|  Micronaut Registry Service
+==========================================+
`);
console.log('PID:     ', state.pid);
console.log('CWD:     ', state.cwd);
console.log('Started: ', state.startedAt);
console.log('Engine:  ', ENGINE_EXE);
console.log('MCP SDK: ', CSHARP_SDK);

// =============================================================================
// Trace
// =============================================================================
// Replay-stable tracing: the provenance hash is computed over a payload that
// excludes wall-clock time. A monotonic sequence number (trace_seq) provides
// ordering; wall_ts is kept for display only and is NOT part of the hash.
let _traceSeq = 0;

// JROM replay stream: every model output, tool request, and tool result is
// appended here so all backends (llama.cpp, kuhul_engine, LM Studio, native
// micronauts, etc.) communicate through a single ordered, hash-chained log.
const JROM_STREAM = path.resolve(__dirname, '..', '..', 'programs', 'tools', 'replay', 'tool-events.jsonl');
const JROM_REQUIRED = new Set([
  'replay_id', 'sequence', 'timestamp', 'tool_id', 'verb', 'target',
  'args', 'phase', 'lane', 'status'
]);
const JROM_PHASES = ['Pop', 'Wo', 'Yax', 'Sek', 'Chen', 'Xul'];

function canonicalJrom(value) {
  return JSON.stringify(value, (key, val) => {
    if (val && typeof val === 'object' && !Array.isArray(val) && !(val instanceof Date)) {
      const sorted = {};
      for (const k of Object.keys(val).sort()) sorted[k] = val[k];
      return sorted;
    }
    return val;
  });
}

function eventHash(event) {
  const body = { ...event };
  delete body.event_hash;
  return crypto.createHash('sha256').update(canonicalJrom(body)).digest('hex');
}

function readJromEvents(streamPath) {
  if (!streamPath) streamPath = JROM_STREAM;
  if (!fs.existsSync(streamPath)) return [];
  const raw = fs.readFileSync(streamPath, 'utf8');
  // Strip UTF-8 BOM if present
  const text = raw.charCodeAt(0) === 0xFEFF ? raw.slice(1) : raw;
  return text.split('\n')
    .filter(line => line.trim())
    .map((line, idx) => {
      try { return JSON.parse(line); }
      catch (e) { throw new Error(`JROM parse error at ${streamPath}:${idx + 1}: ${e.message}`); }
    });
}

function validateJrom(events) {
  let previous = '';
  for (let i = 0; i < events.length; i++) {
    const ev = events[i];
    const missing = [...JROM_REQUIRED].filter(k => ev[k] === undefined);
    if (missing.length) throw new Error(`JROM event ${i}: missing ${missing.join(', ')}`);
    if (ev.sequence !== i) throw new Error(`JROM event ${i}: sequence ${ev.sequence} !== ${i}`);
    if (!JROM_PHASES.includes(ev.phase)) throw new Error(`JROM event ${i}: invalid phase ${ev.phase}`);
    if ((ev.previous_event_hash || '') !== previous) throw new Error(`JROM event ${i}: broken hash chain`);
    if (ev.event_hash !== eventHash(ev)) throw new Error(`JROM event ${i}: invalid event_hash`);
    previous = ev.event_hash;
  }
  return previous;
}

function appendJromEvent(payload) {
  const events = readJromEvents(JROM_STREAM);
  validateJrom(events);
  const required = new Set(JROM_REQUIRED);
  required.delete('sequence');
  const missing = [...required].filter(k => payload[k] === undefined);
  if (missing.length) throw new Error(`JROM append missing ${missing.join(', ')}`);
  const event = {
    replay_id: payload.replay_id || ('jrom-' + crypto.randomUUID()),
    ...payload,
    sequence: events.length,
    previous_event_hash: events.length ? events[events.length - 1].event_hash : '',
    timestamp: new Date().toISOString()
  };
  event.event_hash = eventHash(event);
  const dir = path.dirname(JROM_STREAM);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.appendFileSync(JROM_STREAM, canonicalJrom(event) + '\n');
  return event;
}

function readMicronautRegistry() {
  const micronautsDir = path.resolve(PROJECT_ROOT, 'micronauts');
  const registryPath = path.join(micronautsDir, 'registry.json');
  const semanticDir = path.join(micronautsDir, 'semantic');
  const out = { files: {}, semantic: [], factory_dlls: [] };
  if (fs.existsSync(registryPath)) {
    try { out.registry = JSON.parse(fs.readFileSync(registryPath, 'utf8')); }
    catch (_) {}
  }
  // Load every top-level micronaut/*.json file (factory, evolution, memory, chat, etc.)
  if (fs.existsSync(micronautsDir)) {
    for (const f of fs.readdirSync(micronautsDir)) {
      if (!f.endsWith('.json')) continue;
      try {
        const p = path.join(micronautsDir, f);
        out.files[f] = JSON.parse(fs.readFileSync(p, 'utf8'));
      } catch (_) {}
    }
  }
  if (fs.existsSync(semanticDir)) {
    for (const f of fs.readdirSync(semanticDir)) {
      if (!f.endsWith('.json')) continue;
      try {
        const p = path.join(semanticDir, f);
        out.semantic.push({ file: f, data: JSON.parse(fs.readFileSync(p, 'utf8')) });
      } catch (_) {}
    }
  }
  const factoryDllDir = path.resolve(PROJECT_ROOT, 'dist', 'micronaut-factory', 'build-local', 'bin');
  if (fs.existsSync(factoryDllDir)) {
    for (const f of fs.readdirSync(factoryDllDir)) {
      if (f.endsWith('.dll')) out.factory_dlls.push({ name: f, path: path.join(factoryDllDir, f) });
    }
  }
  const factoryRegistry = path.resolve(PROJECT_ROOT, 'dist', 'micronaut-factory', 'micronauts', '.registry.json');
  if (fs.existsSync(factoryRegistry)) {
    try { out.factory_registry = JSON.parse(fs.readFileSync(factoryRegistry, 'utf8')); }
    catch (_) {}
  }
  return out;
}

function recordMicronautMetadataToJrom(source = 'startup') {
  const meta = readMicronautRegistry();
  try {
    appendJromEvent({
      tool_id: 'micronaut.registry',
      verb: 'registry.snapshot',
      target: 'jrom://micronaut-metadata',
      args: { source },
      phase: 'Pop',
      lane: 'MEMORY',
      status: 'ok',
      result: meta,
      model: 'kuhul-server',
      micronaut: 'µ-memory',
      jrom_program: 'atomic_mcp'
    });
  } catch (e) { writeTrace('jrom.append.error', { source: 'recordMicronautMetadataToJrom', error: e.message }); }
}

// =============================================================================
// NNC-K /micronaut-v4 memory integration
// =============================================================================
const V4_NNC_HOST = process.env.V4_NNC_HOST || '127.0.0.1';
const V4_NNC_PORT = parseInt(process.env.V4_NNC_PORT || '17889', 10);

function lastUserContent(messages) {
  if (!Array.isArray(messages)) return '';
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role === 'user' && (typeof m.content === 'string' || m.content == null)) {
      return String(m.content || '');
    }
  }
  return '';
}

function queryNncKMemory(query, topK = 5, timeoutMs = 1500) {
  return new Promise((resolve) => {
    if (!query) return resolve([]);
    const payload = JSON.stringify({ query, topK });
    const opts = {
      hostname: V4_NNC_HOST, port: V4_NNC_PORT,
      path: '/nnc/search', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
      timeout: timeoutMs
    };
    let data = '';
    const req = http.request(opts, res2 => {
      res2.on('data', c => { data += c; });
      res2.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          resolve(parsed.results || []);
        } catch (_) { resolve([]); }
      });
    });
    req.on('error', (e) => { writeTrace('nnc-k.query.error', { query, error: e.message }); resolve([]); });
    req.on('timeout', () => { req.destroy(); resolve([]); });
    req.write(payload); req.end();
  });
}

function formatMemoryContext(memories) {
  if (!Array.isArray(memories) || memories.length === 0) return '';
  const lines = memories.map((r, i) => {
    const m = r.memory || r;
    const conf = m.confidence != null ? m.confidence : r.score;
    return `${i + 1}. ${m.content || m.text || JSON.stringify(m)} (confidence ${conf ?? '?'})`;
  });
  return `Total Recall memory context:\n${lines.join('\n')}\nUse the above only when relevant.`;
}

async function injectNncKMemory(messages, query, topK = 5) {
  if (!Array.isArray(messages) || messages.length === 0 || !query) return messages;
  try {
    const memories = await queryNncKMemory(query, topK);
    if (!memories.length) return messages;
    const context = formatMemoryContext(memories);
    const out = messages.filter(m => m.role !== 'system');
    return [{ role: 'system', content: context }, ...out];
  } catch (e) {
    writeTrace('nnc-k.memory.error', { error: e.message });
    return messages;
  }
}

// =============================================================================
// /micronaut-v4 JROM → central replay merge
// =============================================================================
const V4_JROM_PATH = path.resolve(PROJECT_ROOT, 'micronaut-v4', 'JROM', 'replay', 'tool-events.jsonl');

function syncV4JromToCentral() {
  if (!fs.existsSync(V4_JROM_PATH)) return { ok: true, synced: 0, v4_events: 0 };
  let central;
  try { central = readJromEvents(JROM_STREAM); }
  catch (e) { writeTrace('jrom.sync.error', { phase: 'read_central', error: e.message }); return { ok: false, synced: 0, error: e.message }; }
  const syncedV4Seqs = new Set(central.filter(e => e.v4_source).map(e => e.v4_sequence));
  let v4Events;
  try { v4Events = readJromEvents(V4_JROM_PATH); }
  catch (e) { writeTrace('jrom.sync.error', { phase: 'read_v4', error: e.message }); return { ok: false, synced: 0, error: e.message }; }
  let previous = central.length ? central[central.length - 1].event_hash : '';
  let count = 0;
  const dir = path.dirname(JROM_STREAM);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  for (const v4 of v4Events) {
    if (syncedV4Seqs.has(v4.sequence)) continue;
    const event = {
      replay_id: v4.replay_id || ('v4-' + crypto.randomUUID()),
      sequence: central.length + count,
      previous_event_hash: previous,
      timestamp: v4.timestamp || new Date().toISOString(),
      tool_id: v4.tool_id,
      verb: v4.verb,
      target: v4.target,
      args: { ...(v4.args || {}), v4_source: true, v4_sequence: v4.sequence, v4_event_hash: v4.event_hash },
      phase: v4.phase,
      lane: v4.lane,
      status: v4.status
    };
    if (v4.result !== undefined) event.result = v4.result;
    if (v4.error !== undefined) event.error = v4.error;
    if (v4.model) event.model = v4.model;
    if (v4.micronaut) event.micronaut = v4.micronaut;
    if (v4.jrom_program) event.jrom_program = v4.jrom_program;
    event.event_hash = eventHash(event);
    fs.appendFileSync(JROM_STREAM, canonicalJrom(event) + '\n');
    previous = event.event_hash;
    count++;
  }
  if (count) writeTrace('jrom.sync.v4', { synced: count, v4_events: v4Events.length });
  return { ok: true, synced: count, v4_events: v4Events.length };
}

function writeTrace(event, payload = {}) {
  const trace = { event, payload, seq: ++_traceSeq };
  // Hash over event+payload+seq only — replay-stable across runs
  trace.hash = crypto.createHash('sha256')
    .update(JSON.stringify({ event, payload, seq: trace.seq }))
    .digest('hex');
  trace.wall_ts = new Date().toISOString();  // display only, not hashed
  const logPath = path.resolve(state.cwd, 'kuhul-server.log');
  try { fs.appendFileSync(logPath, JSON.stringify(trace) + '\n'); } catch (_) {}
}

// =============================================================================
// Port probe
// =============================================================================
function isPortOpen(port) {
  return new Promise(resolve => {
    const s = net.createConnection({ host: '127.0.0.1', port });
    s.on('connect', () => { s.destroy(); resolve(true); });
    s.on('error',   () => resolve(false));
    setTimeout(() => { s.destroy(); resolve(false); }, 1000);
  });
}

// isPortAvailable: tries to *bind* a TCP server — unlike isPortOpen (which connects),
// this detects Windows-excluded ports (excluded ports look free but refuse binding).
function isPortAvailable(port) {
  return new Promise(resolve => {
    const srv = net.createServer();
    srv.once('error', () => resolve(false));
    srv.once('listening', () => { srv.close(() => resolve(true)); });
    srv.listen(port, '127.0.0.1');
  });
}

async function findEnginePort() {
  // If something is already listening on the default port, engine is already running — use it
  if (await isPortOpen(ENGINE_PORT_DEFAULT)) return ENGINE_PORT_DEFAULT;
  // Check if default port is actually bindable (Windows may exclude it)
  if (await isPortAvailable(ENGINE_PORT_DEFAULT)) return ENGINE_PORT_DEFAULT;
  console.warn(`[engine] port ${ENGINE_PORT_DEFAULT} is Windows-excluded — scanning candidates`);
  for (const p of ENGINE_PORT_CANDIDATES) {
    if (!(await isPortOpen(p)) && await isPortAvailable(p)) {
      console.log(`[engine] found available port: ${p}`);
      return p;
    }
  }
  return null;
}

// =============================================================================
// kuhul_engine auto-start
// =============================================================================
async function ensureEngineRunning() {
  // Find a port we can actually bind (Windows excludes some ranges incl. 17xxx on this machine)
  const port = await findEnginePort();
  if (!port) {
    console.warn('[engine] no bindable port found from candidates:', [ENGINE_PORT_DEFAULT, ...ENGINE_PORT_CANDIDATES]);
    return;
  }

  if (await isPortOpen(port)) {
    state.engineUp   = true;
    state.enginePort = port;
    console.log('[engine] already running on port', port);
    return;
  }

  if (!fs.existsSync(ENGINE_EXE)) {
    console.warn('[engine] NOT FOUND:', ENGINE_EXE);
    return;
  }

  state.enginePort = port;
  console.log('[engine] starting kuhul_engine --serve', port, '...');
  engineProcess = spawn(ENGINE_EXE, ['--serve', String(port)], {
    detached: false,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
    cwd: NNC_K_BIN,
  });
  state.enginePid = engineProcess.pid;
  engineProcess.stdout.on('data', d => process.stdout.write('[engine] ' + d));
  engineProcess.stderr.on('data', d => process.stderr.write('[engine] ' + d));
  engineProcess.on('exit', code => {
    console.log('[engine] exited with code', code);
    state.enginePid = null;
    state.engineUp  = false;
    engineProcess   = null;
  });
  writeTrace('engine.start', { pid: engineProcess.pid, port });

  // Wait up to 5 seconds for engine to be ready
  for (let i = 0; i < 10; i++) {
    await new Promise(r => setTimeout(r, 500));
    if (await isPortOpen(port)) { state.engineUp = true; break; }
  }
  console.log('[engine]', state.engineUp ? 'READY on port ' + port : 'still starting...');

  // Update active-model.json so other tools know the actual engine port
  try {
    const amPath = path.join(PROJECT_ROOT, 'active-model.json');
    if (fs.existsSync(amPath)) {
      const am = JSON.parse(fs.readFileSync(amPath, 'utf8'));
      am.engine_endpoint = `http://127.0.0.1:${port}/v1/chat/completions`;
      am.engine_port     = port;
      fs.writeFileSync(amPath, JSON.stringify(am, null, 2) + '\n');
    }
  } catch (_) {}
}

// =============================================================================
// Engine health probe
// =============================================================================
async function checkEngineHealth() {
  state.engineUp = await isPortOpen(state.enginePort);
  return state.engineUp;
}

// =============================================================================
// Proxy a request to kuhul_engine
// =============================================================================
function proxyToEngine(method, pathSuffix, body, cb) {
  const opts = {
    hostname: '127.0.0.1',
    port: state.enginePort,
    path: pathSuffix,
    method: method,
    headers: { 'Content-Type': 'application/json' }
  };
  const req = http.request(opts, res => {
    let data = '';
    res.on('data', c => { data += c; });
    res.on('end', () => {
      try { cb(null, JSON.parse(data), res.statusCode); }
      catch (e) { cb(null, data, res.statusCode); }
    });
  });
  req.on('error', e => cb(e));
  if (body) req.write(typeof body === 'string' ? body : JSON.stringify(body));
  req.end();
}

// =============================================================================
// PowerShell helper
// =============================================================================
function runPs1(scriptPath, args, cwd, cb) {
  execFile('powershell.exe',
    ['-NoLogo', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', scriptPath, ...args],
    { cwd, timeout: 30000, maxBuffer: 4 * 1024 * 1024 },
    (err, stdout, stderr) => {
      if (err) return cb(err, null);
      try { cb(null, JSON.parse(stdout.trim())); }
      catch (_) { cb(null, stdout.trim()); }
    }
  );
}

// =============================================================================
// Micronaut Registry
// =============================================================================
const MICRONAUT_KEYWORDS = {
  coder:            ['code', 'program', 'bug', 'refactor', 'compile', 'function', 'error', 'fix', 'implement', 'script', 'cpp', 'c++', 'python', 'javascript', 'js'],
  memory:           ['memory', 'remember', 'recall', 'forget', 'save this', 'stored', 'retain'],
  ui:               ['ui', 'html', 'css', 'interface', 'frontend', 'canvas', 'display', 'render', 'webview', 'react'],
  stack_doc:        ['stack', 'kuhul', 'khanary', 'system', 'architecture', 'how does', 'what is', 'native', 'driver', 'dll', 'micronauts', 'registry', 'boss', 'webx', 'orchestrat', 'fieldgraph', 'directml', 'gemm', 'intel hd', 'hd 4600', 'atomic dom', 'atomic block', 'klsl', 'shader', 'planner', 'executor', 'taskengine'],
  primeos_guide:    ['primeos', 'desktop', 'wpf', 'app shell', 'window', 'model selector', 'webview2', 'winui'],
  scx_guide:        ['scx', 'scxq2', 'bytecode', 'compile', 'decompile', 'runtime', 'opcode', '@op', 'instruction', 'xcfe'],
  asx_guide:        ['asx', 'gravity', 'entropy', 'pressure', 'physics', 'routing', 'attention', 'state', 'shared memory'],
  distillation_guide: ['distill', 'lora', 'teacher', 'train', 'fine-tune', 'from_zero', 'safetensors', 'oss_distillation'],
  tool_call:        ['tool', 'function call', 'call', 'invoke', 'mcp', 'json-rpc', 'kuhul_task_boss'],
  chat:             ['chat', 'talk', 'hello', 'hi', 'hey', 'conversation'],
  compiled_model:   ['micronaut model', 'compiled model', 'brain router', 'vault keeper', 'swarm mesh', 'training runtime', 'mayan orchestrat', 'glyph opcode', 'gpu dispatch', 'fabric graph', 'bson brain', 'micronaut_brain', 'worker host', 'phase boundary', 'scope gate', 'collapse timing', 'proof generation', 'replay identity', 'token signal', 'stream tokens', 'emit token', 'factory create', 'register service', 'executor', 'coordinator', 'mn-coord', 'mn-factory', 'mn-executor', 'mn-worker', 'co-1', 'fg-1', 'ex-1', 'dotnet worker', 'simd', 'matmul kernel'],
};

const ATOMIC_BLOCKS = ['HEADER', 'MENU', 'BODY', 'FEED', 'FOOTER'];

// Load behavioral n-gram profiles from compiled micronaut model (cached)
let _compiledProfiles = null;
function _getCompiledProfiles() {
  if (_compiledProfiles) return _compiledProfiles;
  const profilesPath = 'E:\\models\\micronaut\\micronaut\\micronaut-profiles.json';
  try {
    if (fs.existsSync(profilesPath)) {
      const raw = JSON.parse(fs.readFileSync(profilesPath, 'utf8'));
      _compiledProfiles = raw.profiles || {};
    }
  } catch (_) {}
  return _compiledProfiles || {};
}

// Score prompt against compiled behavioral profile n-grams; returns highest-scoring profile id
function _scoreProfileNgrams(text) {
  const profiles = _getCompiledProfiles();
  let bestProfile = null;
  let bestScore = 0;
  for (const [id, profile] of Object.entries(profiles)) {
    const ngrams = profile.ngrams || {};
    let score = 0;
    for (const bg of (ngrams.bigrams || [])) { if (text.includes(bg.toLowerCase())) score += 1; }
    for (const tg of (ngrams.trigrams || [])) { if (text.includes(tg.toLowerCase())) score += 1.5; }
    if (score > bestScore) { bestScore = score; bestProfile = id; }
  }
  return bestScore > 0 ? { id: bestProfile, score: bestScore } : null;
}

function selectMicronaut(prompt) {
  const text = (prompt || '').toLowerCase();
  const registry = loadRegistry();
  let best = { name: 'khanary', confidence: 0.5, quant_tier: 'fast', reason: 'default', blocks: ['BODY'] };
  let bestScore = 0;

  for (const mn of registry) {
    const keywords = MICRONAUT_KEYWORDS[mn.name] || [];
    let score = 0;
    for (const kw of keywords) {
      if (text.includes(kw)) score += 1;
    }
    if (score > 0) {
      score *= (mn.confidence || 0.7);
      if (score > bestScore) {
        bestScore = score;
        best = {
          name: mn.name,
          confidence: mn.confidence || 0.7,
          quant_tier: mn.quant_tier || 'standard',
          reason: 'keyword match',
          blocks: blockForMicronaut(mn)
        };
      }
    }
  }

  // Secondary: compiled behavioral profile n-gram scoring
  // If profile n-grams score higher, prefer compiled_model as a tie-breaker
  const profileHit = _scoreProfileNgrams(text);
  if (profileHit && profileHit.score * 0.5 > bestScore) {
    const compiledMn = registry.find(m => m.name === 'compiled_model');
    if (compiledMn) {
      best = {
        name: 'compiled_model',
        confidence: compiledMn.confidence || 0.7,
        reason: 'profile:' + profileHit.id,
        blocks: blockForMicronaut(compiledMn)
      };
    }
  }

  // Fold routing: if prompt names a phase/fold explicitly, boost matching fold micronaut
  const foldMatch = text.match(/\b(pop|wo|yax|sek|chen|xul)\b/);
  if (foldMatch) {
    const foldName = foldMatch[1];
    const foldMn = registry.find(m => m.name === foldName && m.category === 'fold');
    if (foldMn) {
      best = {
        name: foldMn.name,
        confidence: foldMn.confidence || 0.7,
        quant_tier: foldMn.quant_tier || 'standard',
        reason: 'explicit fold',
        blocks: blockForMicronaut(foldMn)
      };
    }
  }

  return best;
}

function blockForMicronaut(mn) {
  // Map micronaut category/fold to the Atomic DOM block(s) it renders in
  switch (mn.category) {
    case 'system':    return ['BODY'];
    case 'persona':   return ['HEADER', 'BODY'];
    case 'stack':     return ['MENU', 'BODY'];
    case 'specialist':
      if (mn.name === 'ui') return ['FOOTER', 'BODY'];
      if (mn.name === 'coder') return ['BODY'];
      if (mn.name === 'memory') return ['MENU', 'BODY'];
      return ['BODY'];
    case 'fold':
      if (mn.name === 'pop')  return ['HEADER'];
      if (mn.name === 'wo')   return ['MENU'];
      if (mn.name === 'yax')  return ['MENU', 'BODY'];
      if (mn.name === 'sek')  return ['BODY'];
      if (mn.name === 'chen') return ['BODY'];
      if (mn.name === 'xul')  return ['FOOTER'];
      return ['BODY'];
    default: return ['BODY'];
  }
}

function loadRegistry() {
  const registryPath = path.join(PROJECT_ROOT, 'micronauts', 'registry.json');
  if (fs.existsSync(registryPath)) {
    try {
      const raw = JSON.parse(fs.readFileSync(registryPath, 'utf8'));
      return Array.isArray(raw.micronauts) ? raw.micronauts : [];
    } catch (_) {}
  }
  const dir = path.join(PROJECT_ROOT, 'micronauts');
  const entries = [];
  if (!fs.existsSync(dir)) return entries;
  for (const file of fs.readdirSync(dir)) {
    if (!file.endsWith('.json') || file === 'registry.json') continue;
    try {
      const j = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
      if (j.name) entries.push({
        name: j.name, fold: j.fold || null, category: j.category || 'unknown',
        description: j.description || '', confidence: j.confidence || 0.7, sampling: j.sampling || {}
      });
    } catch (_) {}
  }
  return entries;
}

// =============================================================================
// Factory: auto-create micronaut
// =============================================================================
function factoryCreate(name, opts = {}) {
  if (!name || typeof name !== 'string') return null;
  const safeName = name.toLowerCase().replace(/[^a-z0-9_-]/g, '_');
  const curiosity  = opts.curiosity  ?? 0.6;
  const concern    = opts.concern    ?? 0.5;
  const valence    = opts.valence    ?? 0.1;
  const confidence = opts.confidence ?? 0.65;
  const attachment = opts.attachment ?? 0.4;
  const temp  = Math.max(0.05, Math.min(1.4, curiosity * 0.6 + valence * 0.2 + 0.2));
  const pen   = Math.max(1.0,  Math.min(1.5, 1.0 + concern * 0.3 + (1.0 - confidence) * 0.2));
  const lastN = attachment > 0.5 ? 96 : 64;
  const entry = {
    name: safeName, fold: opts.fold || null, category: opts.category || 'auto',
    description: opts.description || `Auto-generated micronaut: ${name}`,
    confidence, sampling: { repeat_penalty: +pen.toFixed(3), temperature: +temp.toFixed(3), repeat_last_n: lastN },
    created: new Date().toISOString(), auto: true
  };
  const outDir = path.resolve(PROJECT_ROOT, 'micronauts');
  if (fs.existsSync(outDir)) {
    const outPath = path.join(outDir, safeName + '.json');
    if (!fs.existsSync(outPath)) {
      fs.writeFileSync(outPath, JSON.stringify({
        name: entry.name, fold: entry.fold, category: entry.category,
        description: entry.description, confidence: entry.confidence, sampling: entry.sampling
      }, null, 2) + '\n');
      console.log('[factory] created:', safeName);
    }
  }
  autoCreated.push(entry);
  writeTrace('factory.create', { name: safeName, sampling: entry.sampling });
  try {
    appendJromEvent({
      tool_id: 'micronaut.factory',
      verb: 'factory.create',
      target: 'micronaut://factory',
      args: { name: safeName, ...opts },
      phase: 'Yax',
      lane: 'TOOL',
      status: 'ok',
      result: entry,
      model: opts.model || 'kuhul-server',
      micronaut: safeName,
      jrom_program: 'atomic_mcp'
    });
  } catch (e) { writeTrace('jrom.append.error', { source: 'factory.create', error: e.message }); }
  broadcastSSE({ type: 'factory.created', micronaut: entry });
  return entry;
}

// =============================================================================
// SSE Broadcast — micronaut events
// =============================================================================
function broadcastSSE(data) {
  const msg = 'data: ' + JSON.stringify(data) + '\n\n';
  for (const res of sseClients) {
    try { res.write(msg); } catch (_) { sseClients.delete(res); }
  }
}

// =============================================================================
// MCP Protocol — Model Context Protocol 2025-06-18
// Streamable HTTP transport: POST /mcp  + GET /mcp/sse
// =============================================================================
const MCP_PROTOCOL_VERSION = '2025-06-18';

const MCP_TOOLS = [
  {
    name: 'kuhul_chat',
    description: 'Send a prompt to the kuhul_engine local inference API (OpenAI-compatible). Acts as a cloud model endpoint for the NNC-K stack.',
    inputSchema: {
      type: 'object',
      properties: {
        prompt:  { type: 'string', description: 'The user prompt to send' },
        model:   { type: 'string', description: 'Model name (default: gpt-oss-20b-MXFP4.gguf)' },
        tokens:  { type: 'number', description: 'Max tokens (default: 512)' },
        system:  { type: 'string', description: 'Optional system message' }
      },
      required: ['prompt']
    }
  },
  {
    name: 'kuhul_tasklist',
    description: 'Generate a declarative TaskList JSON from a natural-language prompt using MicrosoftSDK.ps1. Returns a structured task plan for kuhul_engine task-boss execution.',
    inputSchema: {
      type: 'object',
      properties: {
        prompt: { type: 'string', description: 'Natural language description of what to build or plan' },
        tokens: { type: 'number', description: 'Max tokens for the planner (default: 512)' }
      },
      required: ['prompt']
    }
  },
  {
    name: 'kuhul_task_boss',
    description: 'Execute an admitted TaskList through kuhul_engine task-boss. Dispatches and runs allowlisted NNC-K tasks via the BOSS FieldGraph.',
    inputSchema: {
      type: 'object',
      properties: {
        tasks: {
          type: 'array',
          description: 'TaskList tasks array (same shape as kuhul_tasklist output)',
          items: {
            type: 'object',
            properties: {
              id:          { type: 'string' },
              action:      { type: 'string' },
              description: { type: 'string' },
              provider:    { type: 'string' },
              depends_on:  { type: 'array', items: { type: 'string' } }
            }
          }
        },
        verb:        { type: 'string', description: 'Task verb: task.plan, app.create, build.game, build.website, build.program, build.micronaut' },
        target_kind: { type: 'string', description: 'Target kind: app, game, website, program, micronaut, none' },
        prompt:      { type: 'string', description: 'Natural-language request to forward to TaskEngine admission (included in the task list)' },
        model:       { type: 'string', description: 'Model / AtomicDOM alias hint for TaskEngine admission' }
      },
      required: ['tasks']
    }
  },
  {
    name: 'kuhul_json_runtime',
    description: 'Execute a JSON runtime program through json_runtime.exe. Runs manifest-backed execution plans in the NNC-K native runtime.',
    inputSchema: {
      type: 'object',
      properties: {
        program: { type: 'string', description: 'Program name or path (relative to json-runtime/programs/)' },
        args:    { type: 'array', items: { type: 'string' }, description: 'Additional runtime args' }
      },
      required: ['program']
    }
  },
  {
    name: 'kuhul_manifest',
    description: 'Get the full NNC-K stack manifest including capabilities, micronauts, SDK info, runtime details, and toolchain via MicrosoftSDK.ps1.',
    inputSchema: {
      type: 'object',
      properties: {
        section: {
          type: 'string',
          enum: ['full', 'persona', 'stack', 'actions', 'toolchain'],
          description: 'Which section of the manifest to return (default: full)'
        }
      }
    }
  },
  {
    name: 'kuhul_engine_status',
    description: 'Check the status of kuhul_engine.exe, json_runtime.exe, and WWAHost. Returns PID, port status, and engine health.',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  {
    name: 'kuhul_wwa_host',
    description: 'Launch a WWA (Windows Web Application) app through WWAHost.exe in the NNC-K WebX runtime.',
    inputSchema: {
      type: 'object',
      properties: {
        app:  { type: 'string', description: 'App manifest path or name to launch' },
        args: { type: 'array', items: { type: 'string' }, description: 'Additional launch args' }
      },
      required: ['app']
    }
  },
  {
    name: 'kuhul_wwa_package',
    description: 'Package an app folder into a .wwa zip container. Uses native ke_package_wwa from kuhul_engine_driver.dll when available; falls back to PowerShell Compress-Archive.',
    inputSchema: {
      type: 'object',
      properties: {
        app_root:      { type: 'string', description: 'Folder containing the app (index.html, manifest.json, assets)' },
        out_wwa_path:  { type: 'string', description: 'Output .wwa file path' }
      },
      required: ['app_root', 'out_wwa_path']
    }
  },
  {
    name: 'kuhul_wwa_launch',
    description: 'Launch a .wwa bundle or app folder through WWAHost.exe. Uses native ke_launch_wwa when available.',
    inputSchema: {
      type: 'object',
      properties: {
        app:  { type: 'string', description: 'Path to .wwa file or app folder' },
        path: { type: 'string', description: 'Alias for app' },
        args: { type: 'array', items: { type: 'string' }, description: 'Extra WWAHost args' }
      },
      required: []
    }
  },
  {
    name: 'kuhul_grammar',
    description: 'Get the K\'UHUL language grammar in EBNF format. Use `full` for the complete 81KB KUHUL-LLM.ebnf or `excerpt` for the reference excerpt. For llama.cpp constrained generation, also returns a GBNF-compatible header note.',
    inputSchema: {
      type: 'object',
      properties: {
        mode: {
          type: 'string',
          enum: ['excerpt', 'full', 'gbnf_note'],
          description: 'excerpt=reference 200-line excerpt, full=complete 77KB grammar, gbnf_note=instructions for GBNF constrained generation'
        }
      }
    }
  },
  {
    name: 'kuhul_grammar_validate',
    description: 'Validate a K\'UHUL source file or inline string using kuhul_engine. Returns AST JSON or validation errors.',
    inputSchema: {
      type: 'object',
      properties: {
        source:    { type: 'string', description: 'K\'UHUL source code to validate (written to a temp file)' },
        sourcePath:{ type: 'string', description: 'Absolute path to an existing .kuhul file to validate' },
        mode:      { type: 'string', enum: ['validate', 'ast', 'analyze', 'ebnf'], description: 'kuhul_engine command to run (default: validate)' }
      }
    }
  },
  {
    name: 'kuhul_forge',
    description: 'Forge a memory micronaut artifact via kuhul_engine --forge. Call this when the user says "save memory", "memory update", or wants to capture a point of interest. Creates a named micronaut in the registry and runs kuhul_engine field forge.',
    inputSchema: {
      type: 'object',
      properties: {
        text:     { type: 'string', description: 'The memory/context text to forge into a micronaut artifact' },
        name:     { type: 'string', description: 'Optional micronaut name (auto-generated if omitted)' },
        category: { type: 'string', description: 'Category tag (default: memory)' }
      },
      required: ['text']
    }
  },
  {
    name: 'kuhul_driver_plan',
    description: 'Plan tasks using the native khanary_driver.dll (in-process TaskEngine + DAG). Returns a topological plan with provider resolution. Faster than execFile to kuhul_engine.',
    inputSchema: {
      type: 'object',
      properties: {
        tasks: {
          type: 'array',
          description: 'TaskList tasks array [{id, action, description, provider, depends_on}]',
          items: { type: 'object' }
        },
        verb: { type: 'string', description: 'Task verb (default: task.plan)' }
      },
      required: ['tasks']
    }
  },
  {
    name: 'kuhul_driver_dispatch',
    description: 'Dispatch a single task to a micronaut provider via khanary_driver.dll. Returns dispatch result with provider status.',
    inputSchema: {
      type: 'object',
      properties: {
        id:          { type: 'string', description: 'Task ID' },
        action:      { type: 'string', description: 'Task action (e.g. app.create, build.website)' },
        description: { type: 'string', description: 'Task description' },
        provider:    { type: 'string', description: 'Target provider (micronaut-coder, micronaut-factory, json-runtime, etc.)' }
      },
      required: ['id', 'action', 'provider']
    }
  },
  {
    name: 'kuhul_glyph_phase',
    description: 'Dispatch a K\'UHUL phase/fold glyph opcode. Accepts hex opcode (0x5000-0x5005 for phases POP-XUL, 0x6000-0x6004 for folds FOLD_0-FOLD_4, 0x5006 for SEP). Returns admitted status with target provider for MCP routing.',
    inputSchema: {
      type: 'object',
      properties: {
        opcode: { type: 'string', description: 'Hex opcode (e.g. "0x5003" for SEK/execute)' },
        provider_json: { type: 'string', description: 'Optional provider override JSON: {"action":"app.create","provider":"micronaut-factory"}' }
      },
      required: ['opcode']
    }
  },
  {
    name: 'kuhul_gpu_probe',
    description: 'Probe for OpenGL 4.3 compute shader availability via gl_infer_driver.dll. Returns GPU vendor, renderer, version, and shader backend info.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'kuhul_qwen_probe',
    description: 'Probe for Qwen 1.8B D3D11 compute shader availability via qwen_infer_driver.dll. Returns architecture info and shader support.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'kuhul_engine_dom',
    description: 'Load an Atomic DOM manifest and extract model identity via kuhul_engine_driver.dll. Returns tools, persona, execution gating, and chat template.',
    inputSchema: {
      type: 'object',
      properties: {
        manifest_path: { type: 'string', description: 'Path to atomic.manifest.json' }
      },
      required: ['manifest_path']
    }
  },
  {
    name: 'kuhul_build_micronaut',
    description: 'Create a new micronaut profile on disk. Writes a JSON file to micronauts/<name>.json with sampling config and registers it in the MCP tool list.',
    inputSchema: {
      type: 'object',
      properties: {
        name:            { type: 'string', description: 'Micronaut name (e.g. "researcher", "validator")' },
        purpose:         { type: 'string', description: 'What this micronaut does' },
        temperature:     { type: 'number', description: 'Sampling temperature (default 0.5)' },
        repeat_penalty:  { type: 'number', description: 'Repeat penalty (default 1.0)' },
        repeat_last_n:   { type: 'number', description: 'Repeat last N tokens (default 64)' },
        stop:            { type: 'array',  items: { type: 'string' }, description: 'Stop tokens' }
      },
      required: ['name']
    }
  },
  {
    name: 'kuhul_micronaut_registry',
    description: 'Read the full micronaut registry (micronauts/registry.json + semantic/ + factory/evolution DLL manifest) and append a snapshot to JROM so model metadata is replayable.',
    inputSchema: {
      type: 'object',
      properties: {
        include_semantic: { type: 'boolean', description: 'Include semantic micronaut files (default true)' },
        include_factory_dlls: { type: 'boolean', description: 'Include dist/micronaut-factory DLL manifest (default true)' }
      }
    }
  },
  {
    name: 'kuhul_glyph_registry',
    description: 'Dump the full K\'UHUL glyph + lane registry (12 phases + 13 compute lanes = 25 entries). Returns JSON with phases[] and lanes[] arrays, each entry carrying opcode, name, glyph symbol, GGML op mapping, lane kind, fold class, arity, and WebGPU shader path.',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  }
];

function mcpError(id, code, message) {
  return { jsonrpc: '2.0', id, error: { code, message } };
}

function mcpResult(id, result) {
  return { jsonrpc: '2.0', id, result };
}

// Validate toolArgs against the declared inputSchema before dispatch.
function validateToolArgs(toolName, args) {
  const tool = MCP_TOOLS.find(t => t.name === toolName);
  if (!tool || !tool.inputSchema || !tool.inputSchema.properties) return null;
  const schema = tool.inputSchema;
  const required = Array.isArray(schema.required) ? schema.required : [];
  const missing = required.filter(k => args[k] === undefined);
  if (missing.length > 0) {
    return `missing required field(s): ${missing.join(', ')}`;
  }
  for (const [key, prop] of Object.entries(schema.properties)) {
    if (args[key] === undefined) continue;
    const t = prop.type;
    if (t === 'array' && !Array.isArray(args[key])) return `field "${key}" must be an array`;
    if (t === 'object' && (typeof args[key] !== 'object' || args[key] === null || Array.isArray(args[key]))) return `field "${key}" must be an object`;
    if (t === 'number' && typeof args[key] !== 'number') return `field "${key}" must be a number`;
    if (t === 'integer' && (!Number.isInteger(args[key]))) return `field "${key}" must be an integer`;
    if (t === 'string' && typeof args[key] !== 'string') return `field "${key}" must be a string`;
    if (t === 'boolean' && typeof args[key] !== 'boolean') return `field "${key}" must be a boolean`;
  }
  return null;
}

async function _dispatchMcpToolRaw(toolName, toolArgs) {
  const schemaError = validateToolArgs(toolName, toolArgs);
  if (schemaError) {
    return { content: [{ type: 'text', text: 'schema_error: ' + schemaError }], isError: true };
  }
  switch (toolName) {
    case 'kuhul_engine_status': {
      const up = await checkEngineHealth();
      const jrExists  = fs.existsSync(JSON_RT_EXE);
      const wwaExists = fs.existsSync(WWAHOSTEXE);
      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            engine: { pid: state.enginePid, port: ENGINE_PORT, up, exe: ENGINE_EXE },
            json_runtime: { exe: JSON_RT_EXE, exists: jrExists, workDir: JSON_RT_DIR },
            wwa_host: { exe: WWAHOSTEXE, exists: wwaExists },
            microsoft_sdk: { ps1: SDK_PS1, exists: fs.existsSync(SDK_PS1) },
            csharp_sdk: { root: CSHARP_SDK, exists: fs.existsSync(CSHARP_SDK) }
          }, null, 2)
        }]
      };
    }

    case 'kuhul_chat': {
      const up = await checkEngineHealth();
      if (!up) {
        await ensureEngineRunning();
        await new Promise(r => setTimeout(r, 1000));
      }
      const model  = toolArgs.model  || ENGINE_MODEL;
      const tokens = toolArgs.tokens || 512;
      const messages = [];
      if (toolArgs.system) messages.push({ role: 'system', content: toolArgs.system });
      messages.push({ role: 'user', content: toolArgs.prompt });
      const payload = { model, messages, max_tokens: tokens, stream: false };
      return new Promise(resolve => {
        proxyToEngine('POST', '/v1/chat/completions', payload, (err, data, status) => {
          if (err) return resolve({ content: [{ type: 'text', text: 'Engine error: ' + err.message }], isError: true });
          const content = (status < 300 && data?.choices?.[0]?.message?.content) || JSON.stringify(data);
          resolve({ content: [{ type: 'text', text: content }] });
        });
      });
    }

    case 'kuhul_tasklist': {
      if (!fs.existsSync(SDK_PS1)) {
        return { content: [{ type: 'text', text: 'MicrosoftSDK.ps1 not found: ' + SDK_PS1 }], isError: true };
      }
      const tokens = toolArgs.tokens || 512;
      return new Promise(resolve => {
        runPs1(SDK_PS1, [
          '-Command', 'tasklist',
          '-Prompt', toolArgs.prompt,
          '-Endpoint', `http://127.0.0.1:${state.enginePort}/v1/chat/completions`,
          '-Model', ENGINE_MODEL,
          '-Tokens', String(tokens)
        ], path.dirname(SDK_PS1), (err, data) => {
          if (err) return resolve({ content: [{ type: 'text', text: 'SDK error: ' + err.message }], isError: true });
          resolve({ content: [{ type: 'text', text: typeof data === 'string' ? data : JSON.stringify(data, null, 2) }] });
        });
      });
    }

    case 'kuhul_task_boss': {
      const taskList = {
        verb:        toolArgs.verb        || 'task.plan',
        target_kind: toolArgs.target_kind || 'none',
        tasks:       toolArgs.tasks       || [],
        prompt:      toolArgs.prompt      || '',
        model:       toolArgs.model       || ''
      };

      // Try driver DLL first (fast, in-process)
      if (toolArgs.verb === 'task.plan' || toolArgs.verb === 'app.inspect') {
        const plan = driverPlan(JSON.stringify(taskList));
        if (plan && !plan.error) {
          return { content: [{ type: 'text', text: JSON.stringify(plan, null, 2) }] };
        }
      }

      // Try driver dispatch for single-task execution
      if (taskList.tasks && taskList.tasks.length === 1) {
        const disp = driverDispatch(JSON.stringify(taskList.tasks[0]));
        if (disp && !disp.error) {
          return { content: [{ type: 'text', text: JSON.stringify(disp, null, 2) }] };
        }
      }

      // Fallback: kuhul_engine.exe task-boss
      if (!fs.existsSync(ENGINE_EXE)) {
        return { content: [{ type: 'text', text: 'kuhul_engine.exe not found: ' + ENGINE_EXE }], isError: true };
      }
      const tmpFile = path.join(state.cwd, '_kuhul_tasklist_tmp.json');
      fs.writeFileSync(tmpFile, JSON.stringify(taskList, null, 2));
      return new Promise(resolve => {
        execFile(ENGINE_EXE, ['task-boss', tmpFile], { timeout: 30000, maxBuffer: 4 * 1024 * 1024 },
          (err, stdout, stderr) => {
            try { fs.unlinkSync(tmpFile); } catch (_) {}
            if (err && !stdout) return resolve({ content: [{ type: 'text', text: 'task-boss error: ' + err.message }], isError: true });
            resolve({ content: [{ type: 'text', text: stdout || stderr || 'done' }] });
          }
        );
      });
    }

    case 'kuhul_driver_plan': {
      const taskListJson = JSON.stringify({
        tasks: toolArgs.tasks || [],
        verb: toolArgs.verb || 'task.plan'
      });
      const plan = driverPlan(taskListJson);
      if (!plan) {
        return { content: [{ type: 'text', text: 'driver not available' }], isError: true };
      }
      if (plan.error) {
        return { content: [{ type: 'text', text: plan.error }], isError: true };
      }
      return { content: [{ type: 'text', text: JSON.stringify(plan, null, 2) }] };
    }

    case 'kuhul_driver_dispatch': {
      const taskJson = JSON.stringify(toolArgs);
      const result = driverDispatch(taskJson);
      if (!result) {
        return { content: [{ type: 'text', text: 'driver not available' }], isError: true };
      }
      if (result.error) {
        return { content: [{ type: 'text', text: result.error }], isError: true };
      }
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }

    case 'kuhul_glyph_phase': {
      if (!glyphDll) return { content: [{ type: 'text', text: 'glyph driver not available' }], isError: true };
      const opcodeStr = toolArgs.opcode || '0x5000';
      const opcode = parseInt(opcodeStr, 16);
      const result = glyphDispatchPhase(opcode, toolArgs.provider_json);
      if (!result) return { content: [{ type: 'text', text: 'glyph dispatch failed' }], isError: true };
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }

    case 'kuhul_micronaut_registry': {
      const meta = readMicronautRegistry();
      recordMicronautMetadataToJrom('tools/call');
      return { content: [{ type: 'text', text: JSON.stringify(meta, null, 2) }] };
    }

    case 'kuhul_glyph_registry': {
      if (!glyphDll) return { content: [{ type: 'text', text: 'glyph driver not available' }], isError: true };
      const registry = glyphDumpRegistry();
      if (!registry) return { content: [{ type: 'text', text: 'registry dump failed' }], isError: true };
      return { content: [{ type: 'text', text: JSON.stringify(registry, null, 2) }] };
    }

    case 'kuhul_gpu_probe': {
      const gpuInfo = glDll ? glDll.gli_gpu_info() : null;
      const probe = glDll ? glDll.gli_probe() : 0;
      if (glDll && gpuInfo) {
        try { return { content: [{ type: 'text', text: gpuInfo }] }; } finally { glDll.gli_free_string(gpuInfo); }
      }
      return { content: [{ type: 'text', text: JSON.stringify({ available: false, probe, backend: 'OpenGL 4.3 (driver not loaded)' }) }] };
    }

    case 'kuhul_qwen_probe': {
      const probe = qwDll ? qwDll.qw_probe() : 0;
      return { content: [{ type: 'text', text: JSON.stringify({
        available: probe === 1,
        architecture: 'qwen2',
        backend: 'D3D11 cs_5_0',
        shader_ops: 9,
        layers: 24, embed_dim: 2048, heads: 16, head_dim: 128
      }) }] };
    }

    case 'kuhul_engine_dom': {
      if (!engineDll) return { content: [{ type: 'text', text: 'engine driver not available' }], isError: true };
      const mp = toolArgs.manifest_path;
      if (!mp) return { content: [{ type: 'text', text: 'manifest_path required' }], isError: true };
      const errBuf = Buffer.alloc(512);
      const h = engineDll.ke_create(JSON.stringify({}));
      const ok = engineDll.ke_load_dom(h, mp, errBuf, errBuf.length);
      if (!ok) {
        engineDll.ke_destroy(h);
        return { content: [{ type: 'text', text: require('./ffi-shim').ref.readCString(errBuf, 0) }], isError: true };
      }
      const tools = engineDll.ke_get_tools(h);
      const status = engineDll.ke_status(h);
      const result = { tools: tools ? JSON.parse(tools) : [], status: JSON.parse(status) };
      if (tools) engineDll.ke_free_string(tools);
      engineDll.ke_free_string(status);
      engineDll.ke_destroy(h);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }

    case 'kuhul_user_profile_get': {
      const userId = String(toolArgs.user_id || '').trim();
      if (!userId) return { content: [{ type: 'text', text: 'user_id required' }], isError: true };
      const safe = userId.replace(/[^a-zA-Z0-9_-]/g, '') || 'unknown';
      const profilePath = path.join(PROJECT_ROOT, 'micronauts', `user-${safe}.micronaut.json`);
      const templatePath = path.join(PROJECT_ROOT, 'micronauts', 'user-template.micronaut.json');
      let profile = null;
      if (fs.existsSync(profilePath)) {
        try { profile = JSON.parse(fs.readFileSync(profilePath, 'utf-8')); } catch (_) {}
      }
      if (!profile && fs.existsSync(templatePath)) {
        try {
          profile = JSON.parse(fs.readFileSync(templatePath, 'utf-8'));
          profile.user_id = userId; profile.name = `user-${userId}`;
        } catch (_) {}
      }
      if (!profile) return { content: [{ type: 'text', text: JSON.stringify({ error: 'no profile and no template' }) }], isError: true };
      return { content: [{ type: 'text', text: JSON.stringify(profile, null, 2) }] };
    }

    case 'kuhul_user_profile_save': {
      const userId = String(toolArgs.user_id || '').trim();
      if (!userId) return { content: [{ type: 'text', text: 'user_id required' }], isError: true };
      const patch = toolArgs.profile || toolArgs.patch || {};
      if (typeof patch !== 'object' || Array.isArray(patch)) return { content: [{ type: 'text', text: 'profile must be an object' }], isError: true };

      const safe = userId.replace(/[^a-zA-Z0-9_-]/g, '') || 'unknown';
      const micronautDir = path.join(PROJECT_ROOT, 'micronauts');
      if (!fs.existsSync(micronautDir)) fs.mkdirSync(micronautDir, { recursive: true });
      const profilePath = path.join(micronautDir, `user-${safe}.micronaut.json`);
      const templatePath = path.join(micronautDir, 'user-template.micronaut.json');

      // Base = existing file, else template
      let base = {};
      if (fs.existsSync(profilePath)) {
        try { base = JSON.parse(fs.readFileSync(profilePath, 'utf-8')); } catch (_) { base = {}; }
      }
      if (Object.keys(base).length === 0 && fs.existsSync(templatePath)) {
        try { base = JSON.parse(fs.readFileSync(templatePath, 'utf-8')); } catch (_) { base = {}; }
      }
      // Deep merge (recursive; nested dicts merge field-by-field)
      function deepMerge(b, o) {
        const out = { ...b };
        for (const [k, v] of Object.entries(o || {})) {
          if (v && typeof v === 'object' && !Array.isArray(v) && out[k] && typeof out[k] === 'object' && !Array.isArray(out[k])) {
            out[k] = deepMerge(out[k], v);
          } else {
            out[k] = v;
          }
        }
        return out;
      }
      const merged = deepMerge(base, patch);
      merged.name = `user-${userId}`;
      merged.kind = 'user-profile';
      merged.user_id = userId;
      merged.updated_at = new Date().toISOString();

      fs.writeFileSync(profilePath, JSON.stringify(merged, null, 2), 'utf-8');
      console.log('[profile] saved:', profilePath);

      // Mirror into users table via IDB (mx2db helper, if importable)
      try {
        const { execFileSync } = require('child_process');
        const py = process.env.PYTHON || 'python';
        execFileSync(py, ['-c',
          "import sys; sys.path.insert(0, r'" + path.join(PROJECT_ROOT, 'brain2', 'idb').replace(/'/g, "\\'") + "'); " +
          "from mx2db import get_default; db = get_default(); " +
          "import json; " +
          "db.save_user_micronaut(" + JSON.stringify(userId) + ", json.loads(" + JSON.stringify(JSON.stringify(merged)) + ")); " +
          "print('OK')"],
          { timeout: 10000, stdio: ['ignore', 'pipe', 'pipe'] });
      } catch (e) { console.log('[profile] IDB mirror skipped:', e.message); }

      return { content: [{ type: 'text', text: JSON.stringify({ status: 'saved', path: profilePath, profile: merged }, null, 2) }] };
    }

    case 'kuhul_build_micronaut': {
      // GPT-OSS self-extending: create a new micronaut JSON profile on disk
      const name = (toolArgs.name || 'auto_' + Date.now().toString(36)).replace(/[^a-z0-9_]/gi, '_');
      const micronautDir = path.join(PROJECT_ROOT, 'micronauts');
      if (!fs.existsSync(micronautDir)) fs.mkdirSync(micronautDir, { recursive: true });

      const profile = {
        name,
        sampling: {
          repeat_penalty: toolArgs.repeat_penalty || 1.0,
          temperature:    toolArgs.temperature    || 0.5,
          repeat_last_n:  toolArgs.repeat_last_n  || 64,
          ...(toolArgs.stop ? { stop: Array.isArray(toolArgs.stop) ? toolArgs.stop : [toolArgs.stop] } : {}),
        },
        created_by: 'gpt-oss',
        created_at: new Date().toISOString(),
        purpose: toolArgs.purpose || '',
      };

      const filePath = path.join(micronautDir, name + '.json');
      fs.writeFileSync(filePath, JSON.stringify(profile, null, 2));
      console.log('[micronaut] created:', filePath);

      // Register in MCP_TOOLS dynamically
      if (!MCP_TOOLS.find(t => t.name === name)) {
        MCP_TOOLS.push({
          name,
          description: `Auto-created micronaut: ${toolArgs.purpose || name}`,
          inputSchema: { type: 'object', properties: {} },
          _dynamic: true,
        });
      }

      return { content: [{ type: 'text', text: JSON.stringify({
        status: 'created',
        name,
        path: filePath,
        profile,
        message: `Micronaut "${name}" created. Available immediately for dispatch.`
      }, null, 2) }] };
    }

    case 'kuhul_json_runtime': {
      if (!fs.existsSync(JSON_RT_EXE)) {
        return { content: [{ type: 'text', text: 'json_runtime.exe not found: ' + JSON_RT_EXE }], isError: true };
      }
      const progArg  = toolArgs.program || 'main';
      const extraArgs = toolArgs.args || [];
      return new Promise(resolve => {
        execFile(JSON_RT_EXE, [progArg, ...extraArgs],
          { cwd: JSON_RT_DIR, timeout: 30000, maxBuffer: 4 * 1024 * 1024 },
          (err, stdout, stderr) => {
            if (err && !stdout) return resolve({ content: [{ type: 'text', text: 'json_runtime error: ' + err.message + '\n' + stderr }], isError: true });
            resolve({ content: [{ type: 'text', text: stdout || stderr || 'done' }] });
          }
        );
      });
    }

    case 'kuhul_manifest': {
      if (!fs.existsSync(SDK_PS1)) {
        return { content: [{ type: 'text', text: 'MicrosoftSDK.ps1 not found: ' + SDK_PS1 }], isError: true };
      }
      const section = toolArgs.section || 'full';
      const cmdMap = { full: 'manifest', persona: 'persona', stack: 'stack-manifest', actions: 'actions', toolchain: 'toolchain' };
      const cmd = cmdMap[section] || 'manifest';
      return new Promise(resolve => {
        runPs1(SDK_PS1, ['-Command', cmd], path.dirname(SDK_PS1), (err, data) => {
          if (err) return resolve({ content: [{ type: 'text', text: 'SDK error: ' + err.message }], isError: true });
          resolve({ content: [{ type: 'text', text: typeof data === 'string' ? data : JSON.stringify(data, null, 2) }] });
        });
      });
    }

    case 'kuhul_wwa_host': {
      if (!fs.existsSync(WWAHOSTEXE)) {
        return { content: [{ type: 'text', text: 'WWAHost.exe not found: ' + WWAHOSTEXE }], isError: true };
      }
      const args = [toolArgs.app, ...(toolArgs.args || [])];
      spawn(WWAHOSTEXE, args, { detached: true, stdio: 'ignore' }).unref();
      return { content: [{ type: 'text', text: 'WWAHost launched: ' + toolArgs.app }] };
    }

    case 'kuhul_wwa_package': {
      // Native packaging via kuhul_engine_driver.dll (ke_package_wwa)
      const appRoot = toolArgs.app_root, outPath = toolArgs.out_wwa_path;
      if (!appRoot || !outPath) return { content: [{ type: 'text', text: 'app_root and out_wwa_path required' }], isError: true };
      if (engineDll) {
        const errBuf = Buffer.alloc(512);
        const ok = engineDll.ke_package_wwa(appRoot, outPath, errBuf, errBuf.length);
        if (!ok) return { content: [{ type: 'text', text: 'package failed: ' + require('./ffi-shim').ref.readCString(errBuf, 0) }], isError: true };
        return { content: [{ type: 'text', text: JSON.stringify({ status: 'packaged', out: outPath }) }] };
      }
      // Fallback: zip via PowerShell Compress-Archive (folder → .wwa)
      try {
        require('child_process').execFileSync('powershell', ['-NoProfile', '-Command',
          `Compress-Archive -Path "${appRoot}\\*" -DestinationPath "${outPath}" -Force`],
          { timeout: 20000, stdio: 'ignore' });
        return { content: [{ type: 'text', text: JSON.stringify({ status: 'packaged', out: outPath, backend: 'powershell' }) }] };
      } catch (e) {
        return { content: [{ type: 'text', text: 'package failed: ' + e.message }], isError: true };
      }
    }

    case 'kuhul_wwa_launch': {
      const target = toolArgs.app || toolArgs.path;
      if (!target) return { content: [{ type: 'text', text: 'app (path or .wwa) required' }], isError: true };
      if (engineDll) {
        const errBuf = Buffer.alloc(512);
        const ok = engineDll.ke_launch_wwa(target, errBuf, errBuf.length);
        if (!ok) return { content: [{ type: 'text', text: 'launch failed: ' + require('./ffi-shim').ref.readCString(errBuf, 0) }], isError: true };
        return { content: [{ type: 'text', text: JSON.stringify({ status: 'launched', app: target, backend: 'native' }) }] };
      }
      if (!fs.existsSync(WWAHOSTEXE)) return { content: [{ type: 'text', text: 'WWAHost.exe not found: ' + WWAHOSTEXE }], isError: true };
      spawn(WWAHOSTEXE, [target], { detached: true, stdio: 'ignore' }).unref();
      return { content: [{ type: 'text', text: JSON.stringify({ status: 'launched', app: target, backend: 'WWAHost' }) }] };
    }

    case 'kuhul_grammar': {
      const mode = toolArgs.mode || 'excerpt';
      if (mode === 'gbnf_note') {
        return { content: [{ type: 'text', text: [
          'GBNF Constrained Generation with K\'UHUL Grammar',
          '=================================================',
          'llama.cpp llama-server accepts a "grammar" field in the completion request body.',
          'The grammar must be in GBNF format (a subset of BNF used by llama.cpp).',
          '',
          'To use K\'UHUL-constrained generation:',
          '  1. Convert kuhul.ebnf → kuhul.gbnf using llama-gbnf tools or the ebnf-parser.js',
          '  2. Pass the GBNF string as the "grammar" field in your /v1/chat/completions request:',
          '     { "model": "...", "messages": [...], "grammar": "<gbnf content>" }',
          '',
          'Grammar files:',
          '  Excerpt: ' + KUHUL_EBNF_EXCERPT,
          '  Full:    ' + KUHUL_EBNF_FULL,
          '  Parser:  ' + EBNF_PARSER_JS,
          '',
          'llama.cpp GBNF examples: C:\\Users\\canna\\_khanary_inspect\\khanary-llama-build\\llama.cpp\\grammars\\',
          'kuhul_engine validate: can validate .kuhul source against the grammar'
        ].join('\n') }] };
      }
      const gramPath = mode === 'full' ? KUHUL_EBNF_FULL : KUHUL_EBNF_EXCERPT;
      if (!fs.existsSync(gramPath)) {
        return { content: [{ type: 'text', text: 'Grammar file not found: ' + gramPath }], isError: true };
      }
      const content = fs.readFileSync(gramPath, 'utf8');
      return { content: [{ type: 'text', text: content }] };
    }

    case 'kuhul_grammar_validate': {
      if (!fs.existsSync(ENGINE_EXE)) {
        return { content: [{ type: 'text', text: 'kuhul_engine.exe not found' }], isError: true };
      }
      let filePath = toolArgs.sourcePath;
      let tmpCreated = false;
      if (!filePath && toolArgs.source) {
        filePath = path.join(state.cwd, '_kuhul_validate_tmp.kuhul');
        fs.writeFileSync(filePath, toolArgs.source, 'utf8');
        tmpCreated = true;
      }
      if (!filePath) {
        return { content: [{ type: 'text', text: 'Provide source or sourcePath' }], isError: true };
      }
      const cmd = toolArgs.mode || 'validate';
      return new Promise(resolve => {
        execFile(ENGINE_EXE, [cmd, filePath], { timeout: 15000, maxBuffer: 2 * 1024 * 1024 },
          (err, stdout, stderr) => {
            if (tmpCreated) try { fs.unlinkSync(filePath); } catch (_) {}
            const out = (stdout || '') + (stderr ? '\n[stderr] ' + stderr : '');
            if (err && !stdout) return resolve({ content: [{ type: 'text', text: 'validate error: ' + err.message + '\n' + stderr }], isError: true });
            resolve({ content: [{ type: 'text', text: out }] });
          }
        );
      });
    }

    case 'kuhul_forge': {
      if (!fs.existsSync(ENGINE_EXE)) {
        return { content: [{ type: 'text', text: 'kuhul_engine.exe not found: ' + ENGINE_EXE }], isError: true };
      }
      const forgeText = toolArgs.text || '';
      const forgeName = (toolArgs.name || ('memory_' + Date.now())).toLowerCase().replace(/[^a-z0-9_-]/g, '_');
      const forgeCat  = toolArgs.category || 'memory';
      const entry = factoryCreate(forgeName, {
        category: forgeCat,
        description: forgeText.slice(0, 200),
        curiosity: 0.7, concern: 0.3, valence: 0.2, confidence: 0.8, attachment: 0.6,
        model: toolArgs.model || 'kuhul-server'
      });
      return new Promise(resolve => {
        execFile(ENGINE_EXE, ['--forge', forgeText], { timeout: 15000, maxBuffer: 2 * 1024 * 1024 },
          (err, stdout, stderr) => {
            const out = (stdout || '') + (stderr ? '\n[stderr] ' + stderr : '');
            writeTrace('engine.forge', { name: forgeName, text: forgeText.slice(0, 80) });
            try {
              appendJromEvent({
                tool_id: 'micronaut.evolution',
                verb: 'evolution.forge',
                target: 'micronaut://evolution',
                args: { name: forgeName, category: forgeCat, text: forgeText.slice(0, 200) },
                phase: 'Chen',
                lane: 'MEMORY',
                status: err ? 'error' : 'ok',
                result: { micronaut: entry, forge_output: out.trim() },
                error: err ? err.message : undefined,
                model: toolArgs.model || 'kuhul-server',
                micronaut: forgeName,
                jrom_program: 'atomic_mcp'
              });
            } catch (e) { writeTrace('jrom.append.error', { source: 'kuhul_forge', error: e.message }); }
            resolve({
              content: [{
                type: 'text',
                text: JSON.stringify({ micronaut: entry, forge_output: out.trim() }, null, 2)
              }]
            });
          }
        );
      });
    }

    default:
      return { content: [{ type: 'text', text: 'Unknown tool: ' + toolName }], isError: true };
  }
}

// JROM-wrapped MCP dispatch: every tool call/result is appended to the replay
// stream so model↔tool traffic is auditable and replayable regardless of backend.
async function dispatchMcpTool(toolName, toolArgs) {
  const safeArgs = toolArgs || {};
  const start = Date.now();
  let status = 'ok';
  let result;
  try {
    result = await _dispatchMcpToolRaw(toolName, safeArgs);
    if (result && result.isError) status = 'error';
  } catch (e) {
    status = 'error';
    result = { content: [{ type: 'text', text: 'dispatch error: ' + e.message }], isError: true };
  }
  try {
    appendJromEvent({
      tool_id: toolName,
      verb: 'tools/call',
      target: 'mcp://tools/call',
      args: { name: toolName, arguments: safeArgs },
      phase: 'Sek',
      lane: 'TOOL',
      status,
      result: result ? { isError: !!result.isError, content: result.content } : null,
      error: status === 'error' ? (result?.content?.[0]?.text || 'dispatch failed') : undefined,
      model: 'kuhul-server',
      micronaut: 'µ-tool',
      jrom_program: 'atomic_mcp'
    });
  } catch (e) { writeTrace('jrom.append.error', { source: 'dispatchMcpTool', error: e.message }); }
  return result;
}

async function handleMcpJsonRpc(msg) {
  const { id, method, params } = msg;

  if (method === 'initialize') {
    return mcpResult(id, {
      protocolVersion: MCP_PROTOCOL_VERSION,
      serverInfo: { name: 'kuhul-server', version: pkg.version },
      capabilities: {
        tools:     { listChanged: false },
        resources: { listChanged: false },
        prompts:   { listChanged: false }
      }
    });
  }

  if (method === 'tools/list') {
    return mcpResult(id, { tools: MCP_TOOLS });
  }

  if (method === 'tools/call') {
    const toolName = params?.name;
    const toolArgs = params?.arguments || {};
    try {
      const result = await dispatchMcpTool(toolName, toolArgs);
      return mcpResult(id, result);
    } catch (e) {
      return mcpError(id, -32603, String(e));
    }
  }

  if (method === 'ping') {
    return mcpResult(id, {});
  }

  if (method === 'notifications/initialized') {
    return null; // notification, no response needed
  }

  return mcpError(id, -32601, 'Method not found: ' + method);
}

// MCP SSE notification broadcast
function broadcastMcpNotification(notification) {
  const msg = 'data: ' + JSON.stringify(notification) + '\n\n';
  for (const res of mcpSseClients) {
    try { res.write(msg); } catch (_) { mcpSseClients.delete(res); }
  }
}

// =============================================================================
// CORS
// =============================================================================
function setCors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Accept, Mcp-Session-Id, mcp-protocol-version, Authorization, X-Requested-With');
  res.setHeader('Access-Control-Max-Age', '86400');
}

// =============================================================================
// Read body helper
// =============================================================================
function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', c => { body += c; });
    req.on('end', () => { try { resolve(JSON.parse(body)); } catch (_) { resolve({}); } });
    req.on('error', reject);
  });
}

// ── Studio persistence (projects / files / messages) ────────────────────────
// JSON-file backed store so the fabric chat survives gateway restarts.
const STUDIO_STATE = path.join(PROJECT_ROOT, 'studio-state.json');

function studioLoad() {
  try {
    if (fs.existsSync(STUDIO_STATE)) {
      const j = JSON.parse(fs.readFileSync(STUDIO_STATE, 'utf8'));
      return {
        projects: Array.isArray(j.projects) ? j.projects : [],
        files:    (j.files    && typeof j.files    === 'object') ? j.files    : {},
        messages: (j.messages && typeof j.messages === 'object') ? j.messages : {}
      };
    }
  } catch (e) { console.error('[studio] load error:', e.message); }
  return { projects: [], files: {}, messages: {} };
}
let studio = studioLoad();
function studioSave() {
  try { fs.writeFileSync(STUDIO_STATE, JSON.stringify(studio, null, 2)); }
  catch (e) { console.error('[studio] save error:', e.message); }
}
function studioTouchProject(id) {
  const pr = studio.projects.find(x => x.id === id);
  if (pr) pr.updated = Date.now();
}
function sendJson(res, code, obj) {
  res.writeHead(code, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(obj));
}

// ── Micronaut forge tools — real executable escalations ─────────────────────
// These are NOT model-hallucinated calls. Each entry maps a tool name to a
// real binary/script that the gateway spawns against an on-disk file/dir and
// returns the actual output (deterministic, exit-code verified).
const MICRONAUT_DIR = path.join(PROJECT_ROOT, 'dist');
const CODER_EXE  = path.join(MICRONAUT_DIR, 'micronaut-coder', 'Release', 'micronaut_coder.exe');
const REVIEW_EXE = path.join(MICRONAUT_DIR, 'micronaut-coder', 'Release', 'micronaut_code_reviewer.exe');
const V6_SCRIPT    = path.join(MICRONAUT_DIR, 'micronaut-coder', 'scripts', 'compute-v6-hashes.js');
const TODOS_SCRIPT = path.join(MICRONAUT_DIR, 'micronaut-coder', 'scripts', 'extract-todos-universal.js');

const FORGE_TOOLS = {
  review:          { exe: REVIEW_EXE, args: ['review', '{file}'],                          params: ['file'],          desc: 'Deterministic code review of a file' },
  'review-dir':    { exe: REVIEW_EXE, args: ['review-dir', '{dir}'],                       params: ['dir'],           desc: 'Review all files in a directory' },
  diff:            { exe: REVIEW_EXE, args: ['diff', '{old}', '{new}'],                    params: ['old', 'new'],    desc: 'Review a diff between two file versions' },
  refactor:        { exe: REVIEW_EXE, args: ['refactor', '{file}', '{goal}'],              params: ['file', 'goal'],  desc: 'Suggest refactoring improvements for a file' },
  optimize:        { exe: REVIEW_EXE, args: ['optimize', '{file}', '{metric}'],            params: ['file', 'metric'],desc: 'Suggest optimizations (speed/memory)' },
  todos:           { exe: REVIEW_EXE, args: ['todos', '{file}'],                           params: ['file'],          desc: 'Extract actionable todos from a file' },
  document:        { exe: REVIEW_EXE, args: ['document', '{file}'],                        params: ['file'],          desc: 'Generate documentation for a file' },
  test:            { exe: REVIEW_EXE, args: ['test', '{file}'],                            params: ['file'],          desc: 'Generate test cases for a file' },
  explain:         { exe: REVIEW_EXE, args: ['explain', '{file}'],                         params: ['file'],          desc: 'Explain code functionality' },
  'github-review': { exe: REVIEW_EXE, args: ['github-review', '{file}'],                   params: ['file'],          desc: 'Format a review for a GitHub PR' },
  'v6-hashes':     { node: true, exe: V6_SCRIPT,    args: ['{dir}', '--out', '{tmp}'],     params: ['dir'],           desc: 'Deterministic SHA256 hashes of a directory (replay fingerprint)' },
  'todos-universal':{ node: true, exe: TODOS_SCRIPT, args: ['{dir}', '--out', '{tmp}'],     params: ['dir'],           desc: 'Scan a directory for TODO/FIXME/HACK markers' },
};

// ── Verbs — SVO action vocabulary mapped to forge executables ──────────────
// The KXML `verb` tool (name/subject/object, glyph_token: null) is now REAL:
// each verb name maps to a forge tool and a subject/object → forge-arg slot.
// Model picks a verb + subject (+ optional object); the gateway executes it.
const VERBS = {
  review:           { tool: 'review',           args: { file: 'subject' } },
  todos:            { tool: 'todos',            args: { file: 'subject' } },
  explain:          { tool: 'explain',          args: { file: 'subject' } },
  document:         { tool: 'document',         args: { file: 'subject' } },
  test:             { tool: 'test',             args: { file: 'subject' } },
  'github-review':  { tool: 'github-review',    args: { file: 'subject' } },
  optimize:         { tool: 'optimize',         args: { file: 'subject', metric: 'object' } },
  refactor:         { tool: 'refactor',         args: { file: 'subject', goal: 'object' } },
  'v6-hashes':      { tool: 'v6-hashes',        args: { dir: 'subject' } },
  'todos-universal':{ tool: 'todos-universal',  args: { dir: 'subject' } },
};

function runVerb(name, subject, object, timeoutMs) {
  const v = VERBS[name];
  if (!v) return Promise.resolve({ ok: false, error: 'unknown_verb', verb: name });
  const args = {};
  for (const [forgeArg, slot] of Object.entries(v.args)) {
    args[forgeArg] = slot === 'subject' ? subject : (slot === 'object' ? object : undefined);
  }
  return runForgeTool(v.tool, args, timeoutMs);
}

// ── Tool/verb interceptor loop ─────────────────────────────────────────────
// Closes the gap from RESULT.md: model output → parse tool_call → dispatch →
// inject result → model continuation. Supports BOTH structured `tool_calls`
// (gemma3 + chatml) and raw `<tool_call>...</tool_call>` text (GPT-2 SLERP).
function activeModelChatUrl() {
  try {
    const am = JSON.parse(fs.readFileSync(path.join(PROJECT_ROOT, 'active-model.json'), 'utf8'));
    if (am.controller) return am.controller;   // explicit tool-call controller (3B dolphin)
    if (am.endpoint) return am.endpoint;
    if (am.port) return `http://127.0.0.1:${am.port}/v1/chat/completions`;
  } catch (_) {}
  return 'http://127.0.0.1:9003/v1/chat/completions'; // 3B dolphin tool-call controller
}

function modelChat(url, messages, extra, timeoutMs) {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const mod = u.protocol === 'https:' ? https : http;
    const payload = JSON.stringify({ messages, stream: false, ...extra });
    const req = mod.request(u, { method: 'POST', headers: {
      'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload), 'Connection': 'close',
    } }, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('error', e => { res.destroy(); reject(e); });
      res.on('end', () => { res.destroy(); try { resolve({ status: res.statusCode, json: JSON.parse(data) }); } catch (_) { reject(new Error('bad json from model')); } });
    });
    req.on('error', reject);
    if (timeoutMs) req.setTimeout(timeoutMs, () => req.destroy(new Error('timeout after ' + timeoutMs + 'ms')));
    req.write(payload); req.end();
  });
}

// Parse tool/verb calls from a model message (structured OR raw <tool_call>).
function parseCalls(msg) {
  const out = [];
  if (Array.isArray(msg.tool_calls)) {
    for (const tc of msg.tool_calls) {
      const f = tc.function || {};
      let args = {}; try { args = JSON.parse(f.arguments || '{}'); } catch (_) {}
      out.push({ name: f.name, args, id: tc.id });
    }
    return out;
  }
  const text = msg.content || '';
  const re = /<tool_call>([\s\S]*?)<\/tool_call>/gi;
  let m;
  while ((m = re.exec(text))) {
    const inner = m[1].trim();
    try { const j = JSON.parse(inner); out.push({ name: j.name || j.tool, args: j.arguments || j.args || {}, id: null }); }
    catch (_) {
      const p = inner.split('|').map(s => s.trim());
      out.push({ name: p[0], args: { subject: p[1], object: p[2] }, id: null });
    }
  }
  return out;
}

async function dispatchCall(c) {
  const name = c.name || ''; const args = c.args || {};
  const isSVO = args.subject !== undefined || args.object !== undefined;
  // Prefer the forge-tool interpretation when args carry its declared params
  // (e.g. {file:...} for `review`), else fall back to the SVO verb.
  if (FORGE_TOOLS[name]) {
    const params = FORGE_TOOLS[name].params || [];
    if (!isSVO || params.some(p => args[p] !== undefined)) {
      const t = await runForgeTool(name, args);
      return { kind: 'tool', name, ...t };
    }
  }
  if (VERBS[name]) { const v = await runVerb(name, args.subject, args.object); return { kind: 'verb', name, ...v }; }
  return { kind: 'unknown', name, error: 'no_dispatcher' };
}

// Render OpenAI messages to the raw USER:/ASSISTANT:/TOOL: text the fine-tuned
// mm-toolcall GPT-2 learned (matches the training render + <tool_call> parser).
function renderRaw(messages) {
  let s = '';
  for (const m of messages || []) {
    const role = m.role;
    const tc = Array.isArray(m.tool_calls) && m.tool_calls.length ? m.tool_calls[0] : null;
    if (role === 'system') s += 'SYSTEM: ' + (m.content || '') + '\n';
    else if (role === 'user') s += 'USER: ' + (m.content || '') + '\n';
    else if (role === 'tool') s += 'TOOL: ' + (typeof m.content === 'string' ? m.content : JSON.stringify(m.content)) + '\n';
    else if (role === 'assistant') {
      if (tc && tc.function) {
        s += 'ASSISTANT: <tool_call>' + JSON.stringify({ name: tc.function.name, arguments: tc.function.arguments }) + '</tool_call>\n';
      } else {
        s += 'ASSISTANT: ' + (m.content || '') + '\n';
      }
    }
  }
  return s;
}

// Raw /v1/completions call (for text-only GPT-2 toolcall models with no chat
// handler). Returns choices[0].text.
function rawCompletion(url, prompt, extra, timeoutMs) {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const mod = u.protocol === 'https:' ? https : http;
    const payload = JSON.stringify({ prompt, ...extra });
    const req = mod.request(u, { method: 'POST', headers: {
      'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload), 'Connection': 'close',
    } }, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('error', e => { res.destroy(); reject(e); });
      res.on('end', () => { res.destroy(); try { resolve({ status: res.statusCode, json: JSON.parse(data) }); } catch (_) { reject(new Error('bad json from model')); } });
    });
    req.on('error', reject);
    if (timeoutMs) req.setTimeout(timeoutMs, () => req.destroy(new Error('timeout after ' + timeoutMs + 'ms')));
    req.write(payload); req.end();
  });
}




// Path sandbox: file/dir args must resolve under the project root.
function forgeResolvePath(p) {
  if (!p) return null;
  const abs = path.isAbsolute(p) ? p : path.resolve(PROJECT_ROOT, p);
  const rel = path.relative(PROJECT_ROOT, abs);
  if (rel.startsWith('..') || path.isAbsolute(rel)) return null; // escape
  return abs;
}

function runForgeTool(tool, args, timeoutMs) {
  return new Promise((resolve) => {
    const t = FORGE_TOOLS[tool];
    if (!t) return resolve({ ok: false, error: 'unknown_tool', tool });

    let tmpOut = null;
    let missing = null;
    const argv = t.args.map(a => a.replace(/\{(\w+)\}/g, (m, key) => {
      if (key === 'tmp') { tmpOut = path.join(os.tmpdir(), 'forge-' + tool + '-' + Date.now() + '.json'); return tmpOut; }
      if (key === 'dir')  { const r = forgeResolvePath(args[key]); if (!r) { missing = key; } return r || ''; }
      if (key === 'old' || key === 'new') { const r = forgeResolvePath(args[key]); if (!r) { missing = key; } return r || ''; }
      if (key === 'file'){ const r = forgeResolvePath(args[key]); if (!r) { missing = key; } return r || ''; }
      return (args[key] !== undefined && args[key] !== null) ? String(args[key]) : '';
    }));
    if (missing) return resolve({ ok: false, error: 'arg_outside_sandbox_or_missing', arg: missing });
    if (argv.some(a => a.includes('{'))) return resolve({ ok: false, error: 'missing_arg', argv });

    const opts = { cwd: PROJECT_ROOT, windowsHide: true };
    const child = t.node
      ? spawn(process.execPath, [t.exe, ...argv], opts)
      : spawn(t.exe, argv, opts);
    let stdout = '', stderr = '';
    const timer = setTimeout(() => { child.kill('SIGKILL'); }, timeoutMs || 30000);
    child.stdout.on('data', c => stdout += c);
    child.stderr.on('data', c => stderr += c);
    child.on('error', e => { clearTimeout(timer); resolve({ ok: false, error: 'spawn_error', message: e.message }); });
    child.on('close', code => {
      clearTimeout(timer);
      let report = null;
      if (tmpOut && fs.existsSync(tmpOut)) {
        try { report = JSON.parse(fs.readFileSync(tmpOut, 'utf8')); } catch (_) {}
        try { fs.unlinkSync(tmpOut); } catch (_) {}
      }
      resolve({ ok: code === 0, tool, exitCode: code, stdout, stderr, report });
    });
  });
}

// =============================================================================
// HTTP Server
// =============================================================================
const server = http.createServer(async (req, res) => {
  setCors(res);
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  const url = new URL(req.url, 'http://localhost');
  const p   = url.pathname;

  // ------------------------------------------------------------------
  // Health
  // ------------------------------------------------------------------
  if (req.method === 'GET' && p === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true, pid: state.pid, enginePid: state.enginePid, engineUp: state.engineUp, uptime: process.uptime() }));
    return;
  }

  if (req.method === 'POST' && p === '/jrom/sync-v4') {
    const result = syncV4JromToCentral();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(result));
    return;
  }

  // ------------------------------------------------------------------
  // PRIMEOS — engine + stack status
  // ------------------------------------------------------------------
  if (req.method === 'GET' && p === '/kuhul/engine/status') {
    const engineUp = await checkEngineHealth();
    const jrExists  = fs.existsSync(JSON_RT_EXE);
    const wwaExists = fs.existsSync(WWAHOSTEXE);
    const boundPort = server.address()?.port || REGISTRY_PORT;
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      engine:  { pid: state.enginePid, port: state.enginePort, up: engineUp, exe: ENGINE_EXE },
      gateway: { pid: state.pid, port: boundPort, tools: MCP_TOOLS.length, drivers: glyphPhaseCount() + ' phases, ' + glyphLaneCount() + ' lanes' },
      json_runtime: { exe: JSON_RT_EXE, exists: jrExists, workDir: JSON_RT_DIR },
      wwa_host: { exe: WWAHOSTEXE, exists: wwaExists },
      shm: { name: 'Local\\KuhulGeometricState', available: true },
      uptime: process.uptime()
    }));
    return;
  }

  if (req.method === 'GET' && p === '/kuhul/stack/status') {
    const engineUp = await checkEngineHealth();
    const boundPort = server.address()?.port || REGISTRY_PORT;
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      services: [
        { name: 'kuhul-server',   port: boundPort, up: true },
        { name: 'kuhul_engine',   port: state.enginePort || ENGINE_PORT_DEFAULT, up: engineUp },
        { name: 'json_runtime',   port: 8787, up: fs.existsSync(JSON_RT_EXE) },
        { name: 'llama-server',   port: 9000, up: null },  // probe on demand
      ],
      mcp_tools: MCP_TOOLS.length,
      micronauts: loadRegistry().length + autoCreated.length,
      timestamp: new Date().toISOString()
    }));
    return;
  }

  // ------------------------------------------------------------------
  // Micronaut Registry
  // ------------------------------------------------------------------
  if (req.method === 'GET' && p === '/micronauts/events') {
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' });
    res.write('data: ' + JSON.stringify({ type: 'connected', autoCreated }) + '\n\n');
    sseClients.add(res);
    req.on('close', () => sseClients.delete(res));
    return;
  }

  if (req.method === 'GET' && p === '/micronauts') {
    const registry = loadRegistry();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ micronauts: registry, autoCreated }));
    return;
  }

  const singleMatch = p.match(/^\/micronauts\/([a-zA-Z0-9_-]+)$/);
  if (req.method === 'GET' && singleMatch && singleMatch[1] !== 'events' && singleMatch[1] !== 'factory' && singleMatch[1] !== 'select') {
    const name = singleMatch[1];
    const registry = loadRegistry();
    const mn = registry.find(m => m.name === name) || autoCreated.find(m => m.name === name);
    if (mn) { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(mn)); }
    else     { res.writeHead(404, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ error: 'not found', name })); }
    return;
  }

  if (req.method === 'POST' && p === '/micronauts/factory') {
    const opts = await readBody(req);
    const name  = opts.name || ('auto_' + Date.now());
    const entry = factoryCreate(name, opts);
    if (entry) { res.writeHead(201, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(entry)); }
    else        { res.writeHead(400, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ error: 'invalid name' })); }
    return;
  }

  if ((req.method === 'GET' || req.method === 'POST') && p === '/micronauts/select') {
    const body = req.method === 'POST' ? await readBody(req) : {};
    const prompt = body.prompt || url.searchParams.get('prompt') || '';
    if (!prompt) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'prompt is required' }));
      return;
    }
    const selected = selectMicronaut(prompt);
    const registry = loadRegistry();
    const full = registry.find(m => m.name === selected.name) || selected;

    // Read raw micronaut file to include system context — loadRegistry() strips it
    let systemContext = null;
    const mnFile = path.join(PROJECT_ROOT, 'micronauts', selected.name + '.json');
    if (fs.existsSync(mnFile)) {
      try { systemContext = JSON.parse(fs.readFileSync(mnFile, 'utf8')).system || null; } catch (_) {}
    }

    writeTrace('micronaut.select', { prompt: prompt.slice(0, 120), selected });
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      prompt,
      selected: { ...selected, full },
      system_context: systemContext,
      atomic_blocks: selected.blocks,
      available: registry.map(m => ({ name: m.name, category: m.category, fold: m.fold, confidence: m.confidence }))
    }, null, 2));
    return;
  }

  // ------------------------------------------------------------------
  // Port Discovery (so clients can find the abstract port)
  // ------------------------------------------------------------------
  if (req.method === 'GET' && p === '/.well-known/kuhul-server') {
    const actualPort = server.address()?.port || state.actualPort;
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ port: actualPort, version: pkg.version, engine: { port: ENGINE_PORT, up: state.engineUp } }));
    return;
  }

  // ------------------------------------------------------------------
  // Grammar REST
  // ------------------------------------------------------------------
  if (req.method === 'GET' && p === '/kuhul/grammar') {
    const mode = url.searchParams.get('mode') || 'excerpt';
    const gramPath = mode === 'full' ? KUHUL_EBNF_FULL : KUHUL_EBNF_EXCERPT;
    if (!fs.existsSync(gramPath)) {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'grammar file not found', path: gramPath }));
      return;
    }
    res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end(fs.readFileSync(gramPath, 'utf8'));
    return;
  }

  // ------------------------------------------------------------------
  // kuhul_engine Status + Proxy
  // ------------------------------------------------------------------
  if (req.method === 'GET' && p === '/kuhul/engine/status') {
    const up = await checkEngineHealth();
    const boundPort = server.address()?.port || REGISTRY_PORT;
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      engine:  { pid: state.enginePid, port: ENGINE_PORT, up, exe: ENGINE_EXE },
      runtime: { exe: JSON_RT_EXE, exists: fs.existsSync(JSON_RT_EXE) },
      wwa:     { exe: WWAHOSTEXE,  exists: fs.existsSync(WWAHOSTEXE) },
      sdk:     { ps1: SDK_PS1,     exists: fs.existsSync(SDK_PS1) },
      csharp:  { root: CSHARP_SDK, exists: fs.existsSync(CSHARP_SDK) },
      mcp:     { endpoint: `http://127.0.0.1:${boundPort}/mcp`, ssePath: `http://127.0.0.1:${boundPort}/mcp/sse` }
    }));
    return;
  }

  // POST /kuhul/engine/forge — forge a memory micronaut via kuhul_engine --forge
  if (req.method === 'POST' && p === '/kuhul/engine/forge') {
    const body = await readBody(req);
    const text = body.text || body.prompt || '';
    const name = (body.name || ('memory_' + Date.now())).toLowerCase().replace(/[^a-z0-9_-]/g, '_');
    const cat  = body.category || 'memory';
    if (!text) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'text is required' }));
      return;
    }
    const entry = factoryCreate(name, {
      category: cat,
      description: text.slice(0, 200),
      curiosity: 0.7, concern: 0.3, valence: 0.2, confidence: 0.8, attachment: 0.6
    });
    execFile(ENGINE_EXE, ['--forge', text], { timeout: 15000, maxBuffer: 2 * 1024 * 1024 },
      (err, stdout, stderr) => {
        writeTrace('engine.forge', { name, text: text.slice(0, 80) });
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ micronaut: entry, forge_output: (stdout || '').trim(), stderr: (stderr || '').trim() }));
      }
    );
    return;
  }

  if ((req.method === 'POST' || req.method === 'GET') && p.startsWith('/kuhul/engine/')) {
    const up = await checkEngineHealth();
    if (!up) {
      res.writeHead(503, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'kuhul_engine not available on port ' + ENGINE_PORT }));
      return;
    }
    const enginePath = '/' + p.replace(/^\/kuhul\/engine\//, 'v1/');
    const body = req.method === 'POST' ? await readBody(req) : null;
    proxyToEngine(req.method, enginePath, body, (err, data, status) => {
      if (err) { res.writeHead(502, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ error: err.message })); return; }
      res.writeHead(status || 200, { 'Content-Type': 'application/json' });
      res.end(typeof data === 'string' ? data : JSON.stringify(data));
    });
    return;
  }

  // ------------------------------------------------------------------
  // MCP Server — Streamable HTTP transport (POST /mcp)
  // ------------------------------------------------------------------
  if (req.method === 'POST' && p === '/mcp') {
    const msg    = await readBody(req);
    const accept = req.headers['accept'] || '';

    // Handle batch arrays
    if (Array.isArray(msg)) {
      const replies = (await Promise.all(msg.map(handleMcpJsonRpc))).filter(Boolean);
      if (accept.includes('text/event-stream')) {
        res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' });
        for (const r of replies) res.write('data: ' + JSON.stringify(r) + '\n\n');
        res.end();
      } else {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(replies));
      }
      return;
    }

    const reply = await handleMcpJsonRpc(msg);

    // Notifications return 202 with no body
    if (reply === null) { res.writeHead(202); res.end(); return; }

    // For single-response RPCs respond with application/json regardless of Accept.
    // Cloudflare and most reverse proxies cannot handle an SSE response that closes
    // immediately — they treat it as a broken connection and return 502.
    // SSE is only needed for server-initiated streaming (not supported here yet).
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(reply));
    return;
  }

  // ------------------------------------------------------------------
  // MCP SSE transport — GET /mcp/sse  (legacy) + GET /sse  (studio alias)
  // ------------------------------------------------------------------
  if (req.method === 'GET' && (p === '/mcp/sse' || p === '/sse')) {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive'
    });
    // Use the actual bound port (server.address().port), not the pre-bind REGISTRY_PORT constant
    const boundPort = server.address()?.port || REGISTRY_PORT;
    const msgPath   = p === '/sse' ? '/messages' : '/mcp/messages';
    const endpoint  = `http://127.0.0.1:${boundPort}${msgPath}`;
    res.write(`event: endpoint\ndata: ${endpoint}\n\n`);
    mcpSseClients.add(res);
    req.on('close', () => mcpSseClients.delete(res));
    return;
  }

  // MCP SSE message channel (POST /mcp/messages + POST /messages alias)
  if (req.method === 'POST' && (p === '/mcp/messages' || p === '/messages')) {
    const msg   = await readBody(req);
    const reply = await handleMcpJsonRpc(msg);
    if (reply) broadcastMcpNotification(reply);
    res.writeHead(202); res.end();
    return;
  }

  // ------------------------------------------------------------------
  // OpenAI-compatible /v1/models — exposes the gateway as a virtual model.
  // Tools like call_planner.ps1 or any OpenAI client pointed at :8764 will
  // see "kuhul-planner" and route build/create intents through the gateway.
  // ------------------------------------------------------------------
  if (req.method === 'GET' && p === '/v1/models') {
    const engineBase = `http://127.0.0.1:${state.enginePort}`;
    const models = [
      {
        id: 'kuhul-planner',
        object: 'model',
        created: Math.floor(Date.now() / 1000),
        owned_by: 'kuhul-server',
        description: 'Intent-routing gateway: build/create → MicrosoftSDK.ps1 planner; chat → active inference model',
        permission: [],
        root: 'kuhul-planner',
        parent: null
      }
    ];
    // Also surface the engine model if it's up
    if (state.engineUp) {
      models.push({
        id: ENGINE_MODEL,
        object: 'model',
        created: Math.floor(Date.now() / 1000),
        owned_by: 'kuhul_engine',
        description: `GPT-OSS 20B planner on ${engineBase}`,
        permission: [],
        root: ENGINE_MODEL,
        parent: null
      });
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ object: 'list', data: models }));
    return;
  }

  // ------------------------------------------------------------------
  // MCP tool discovery (plain REST, for non-MCP clients)
  // ------------------------------------------------------------------
  if (req.method === 'GET' && p === '/mcp/tools') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ tools: MCP_TOOLS, protocol: MCP_PROTOCOL_VERSION, server: 'kuhul-server', version: pkg.version }));
    return;
  }

  // ------------------------------------------------------------------
  // Intent-routing chat completions proxy
  // "build/create/make + target" → kuhul_tasklist (MicrosoftSDK.ps1 → kuhul_engine:17480)
  // everything else             → khanary-server inference model (reads active-model.json)
  // ------------------------------------------------------------------
  if (req.method === 'POST' && p === '/v1/chat/completions') {
    // Read raw body so we can both classify intent AND forward if needed
    let rawBody = '';
    await new Promise((ok, fail) => {
      req.on('data', c => { rawBody += c; });
      req.on('end', ok);
      req.on('error', fail);
    });
    let body;
    try { body = JSON.parse(rawBody); } catch (_) { body = {}; }

    // Total Recall: pull relevant persistent memory from /micronaut-v4 NNC-K
    const msgs = body.messages || [];
    const lastUserMsg = [...msgs].reverse().find(m => m.role === 'user')?.content || '';
    body.messages = await injectNncKMemory(msgs, lastUserMsg, 5);

    const lc = lastUserMsg.toLowerCase();

    // Explicit "build/create/make <target>" phrases that belong to the planner.
    // Regular chat ("what is X", "explain Y") flows through to inference.
    const BUILD_PHRASES = [
      'build a game', 'create a game', 'make a game',
      'build an app', 'create an app', 'make an app', 'build a program', 'create a program', 'make a program',
      'build a website', 'create a website', 'make a website', 'build a site', 'create a site',
      'build a micronaut', 'create a micronaut', 'new micronaut', 'create an agent', 'build an agent',
      'plan for', 'roadmap for', 'task list for', 'steps to build', 'steps to create'
    ];
    const isPlanIntent = BUILD_PHRASES.some(phrase => lc.includes(phrase));

    if (isPlanIntent) {
      // Route to planner — MicrosoftSDK.ps1 tasklist → kuhul_engine:17480
      let planResult;
      try { planResult = await dispatchMcpTool('kuhul_tasklist', { prompt: lastUserMsg, tokens: 512 }); }
      catch (e) { planResult = { content: [{ type: 'text', text: 'Planner error: ' + e.message }], isError: true }; }
      const planText = planResult.content?.[0]?.text || 'Task plan generated.';
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        id: 'kuhul-plan-' + Date.now(),
        object: 'chat.completion',
        model: 'kuhul_planner',
        choices: [{ index: 0, message: { role: 'assistant', content: planText }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      }));
      return;
    }

    // Pass-through to active inference model (Ollama/GC-1); reads active-model.json
    let inferPort = 25110;
    let inferModel = null;
    try {
      const amPath = path.join(PROJECT_ROOT, 'active-model.json');
      if (fs.existsSync(amPath)) {
        const am = JSON.parse(fs.readFileSync(amPath, 'utf8'));
        if (am.port && am.port !== REGISTRY_PORT) inferPort = am.port;
        if (am.model) inferModel = am.model;
      }
    } catch (_) {}

    // Override model field so any inference backend (Ollama, llama.cpp, etc.) gets the right name
    if (inferModel) {
      try { body.model = inferModel; } catch (_) {}
    }
    const forwardBody = JSON.stringify(body);

    const inferOpts = {
      hostname: '127.0.0.1', port: inferPort,
      path: '/v1/chat/completions', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(forwardBody) }
    };
    const proxyReq = http.request(inferOpts, proxyRes => {
      const h = { ...proxyRes.headers, 'access-control-allow-origin': '*' };
      res.writeHead(proxyRes.statusCode, h);
      proxyRes.pipe(res);
    });
    proxyReq.on('error', e => {
      if (!res.headersSent) {
        res.writeHead(502, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'inference proxy error: ' + e.message, port: inferPort }));
      }
    });
    proxyReq.write(forwardBody);
    proxyReq.end();
    return;
  }

  // ------------------------------------------------------------------
  // POST /v1/tool-chat — raw <tool_call> interceptor loop
  // For GPT-2 BPE models without chatml: send plain messages, scan raw
  // text output for <tool_call>{...}</tool_call>, execute locally or via
  // MCP, inject <tool_result>...</tool_result>, re-send up to max_loops.
  // Also handles structured tool_calls from chatml-capable models.
  //
  // Body: {
  //   messages, tools, model_url?, max_loops?, max_tokens?, temperature?,
  //   enable_fallback?, fallback_profile?, fallback_model_url?, fallback_timeout_ms?,
  //   allow_forced_tool_bootstrap?
  // }
  // model_url defaults to active-model.json endpoint.
  // fallback_profile defaults to mm-toolcall legacy endpoint (http://127.0.0.1:25502/chat).
  // ------------------------------------------------------------------
  if (req.method === 'POST' && p === '/v1/tool-chat') {
    const body       = await readBody(req);
    const messages   = body.messages   || [];
    const toolDefs   = body.tools      || [];
    const maxLoops   = Math.min(body.max_loops   || 3, 8);
    const maxTokens  = body.max_tokens  || 256;
    const temperature = body.temperature ?? 0.1;
    const allowForcedToolBootstrap = body.allow_forced_tool_bootstrap !== false;
    const enableFallback = body.enable_fallback !== false;
    const fallbackProfile = String(body.fallback_profile || 'mm-toolcall').toLowerCase();
    const fallbackTimeoutMs = Math.max(3000, Math.min(60000, Number(body.fallback_timeout_ms || 20000)));
    const rawToolCallStop = (typeof body.tool_call_stop === 'string' && body.tool_call_stop.trim())
      ? body.tool_call_stop.trim()
      : '</tool_call>';

    function normalizeChatEndpoint(rawUrl) {
      if (!rawUrl || typeof rawUrl !== 'string') return null;
      let u;
      try { u = new URL(rawUrl); } catch (_) { return null; }
      if (!u.pathname || u.pathname === '/' || u.pathname === '') {
        u.pathname = '/v1/chat/completions';
      }
      return u.toString();
    }

    function resolvePrimaryModelUrl() {
      if (body.model_url) return body.model_url;
      try {
        const am = JSON.parse(fs.readFileSync(path.join(PROJECT_ROOT, 'active-model.json'), 'utf8'));
        return am.endpoint || `http://127.0.0.1:${am.port || 9000}/v1/chat/completions`;
      } catch (_) {
        return 'http://127.0.0.1:9000/v1/chat/completions';
      }
    }

    function resolveFallbackModelUrl() {
      if (body.fallback_model_url) return body.fallback_model_url;
      if (process.env.KUHUL_TOOL_FALLBACK_URL) return process.env.KUHUL_TOOL_FALLBACK_URL;
      if (fallbackProfile === 'mm-toolcall' || fallbackProfile === 'tool_call' || fallbackProfile === 'tool-call') {
        return 'http://127.0.0.1:25502/chat';
      }
      return null;
    }

    // Resolve model URL: explicit body.model_url OR active-model.json
    const modelUrl = normalizeChatEndpoint(resolvePrimaryModelUrl()) || 'http://127.0.0.1:9000/v1/chat/completions';
    let fallbackModelUrl = normalizeChatEndpoint(resolveFallbackModelUrl());
    if (fallbackModelUrl === modelUrl) fallbackModelUrl = null;

    const stopTokens = Array.isArray(body.stop)
      ? body.stop.filter(s => typeof s === 'string' && s.length > 0)
      : [];
    for (const t of ['<|endoftext|>', '</tool_result>', rawToolCallStop]) {
      if (!stopTokens.includes(t)) stopTokens.push(t);
    }

    // Detect <tool_call>JSON</tool_call> OR bare {"name":…,"arguments":…} block
    const TC_RE = /<tool_call>([\s\S]*?)<\/tool_call>|(\{[\s\S]*?"name"\s*:\s*"[^"]+"\s*,[\s\S]*?"arguments"\s*:[\s\S]*?\})/;

    function execBuiltinTool(name, args) {
      if (name === 'calculate' || name === 'math') {
        const expr = String(args.expression || args.expr || '');
        try {
          // eslint-disable-next-line no-new-func
          return String(Function('"use strict";const sqrt=Math.sqrt,log=Math.log,pi=Math.PI;return(' + expr.replace(/[^0-9+\-*/().\s%sqrtlogpi]/g,'') + ')')());
        } catch (e) { return 'Error: ' + e.message; }
      }
      if (name === 'count_letters') return String((args.text || '').length);
      return null; // not a built-in — caller will dispatch via MCP
    }

    function hasToolDef(name) {
      if (!Array.isArray(toolDefs) || toolDefs.length === 0) return false;
      const target = String(name || '').toLowerCase();
      return toolDefs.some((t) => String(t?.function?.name || t?.name || '').toLowerCase() === target);
    }

    function extractMathExpression(text) {
      const src = String(text || '');
      const fenced = src.match(/`([^`]+)`/);
      if (fenced && fenced[1]) return fenced[1].trim();
      const mathSeq = src.match(/((?:sqrt\([^)]+\)|log\([^)]+\)|pi|[0-9]+(?:\.[0-9]+)?)(?:\s*[\+\-\*\/%]\s*(?:sqrt\([^)]+\)|log\([^)]+\)|pi|[0-9]+(?:\.[0-9]+)?))+)/i);
      if (mathSeq && mathSeq[1]) return mathSeq[1].trim();
      const afterVerb = src.match(/(?:what is|compute|calculate|evaluate|eval|solve)\s+([^?.!]+)/i);
      if (!afterVerb || !afterVerb[1]) return null;
      const cleaned = afterVerb[1]
        .replace(/\b(use|with)\s+calculate\b/ig, '')
        .replace(/\s+/g, ' ')
        .trim();
      return cleaned || null;
    }

    function extractCountLettersText(text) {
      const src = String(text || '');
      const quoted = src.match(/count(?:\s+the)?\s+letters(?:\s+in)?\s*['"]([^'"]+)['"]/i);
      if (quoted && quoted[1]) return quoted[1];
      const bare = src.match(/count[_\s]?letters\s*[:=]\s*(.+)$/i);
      if (bare && bare[1]) return bare[1].trim();
      return null;
    }

    function inferForcedToolCall(msgs) {
      const lastUserMsg = [...msgs].reverse().find((m) => m.role === 'user')?.content || '';
      const text = String(lastUserMsg);
      const lower = text.toLowerCase();

      if (hasToolDef('calculate') || hasToolDef('math')) {
        const asksMath = /\b(use|call)\s+(the\s+)?(calculate|math)\b/.test(lower)
          || /\bwhat is\b/.test(lower)
          || /\bcompute\b/.test(lower)
          || /\bcalculate\b/.test(lower)
          || /\bsolve\b/.test(lower);
        if (asksMath) {
          const expression = extractMathExpression(text);
          if (expression) {
            return { name: 'calculate', arguments: { expression }, source: 'heuristic_user_prompt' };
          }
        }
      }

      if (hasToolDef('count_letters')) {
        const asksCount = /\bcount(?:\s+the)?\s+letters\b/.test(lower) || /\bcount_letters\b/.test(lower);
        if (asksCount) {
          const countText = extractCountLettersText(text);
          if (countText) {
            return { name: 'count_letters', arguments: { text: countText }, source: 'heuristic_user_prompt' };
          }
        }
      }
      return null;
    }

    function assessTextAdmissibility(text) {
      const raw = typeof text === 'string' ? text : '';
      const trimmed = raw.trim();
      if (!trimmed) {
        return {
          admissible: false,
          reason: 'empty_output',
          metrics: { printableRatio: 0, punctuationRatio: 0, spaceRatio: 0, length: 0 },
        };
      }
      const chars = Array.from(trimmed);
      let printable = 0;
      let punctuation = 0;
      let spaces = 0;
      for (const ch of chars) {
        const code = ch.charCodeAt(0);
        if (code === 32) spaces += 1;
        if ((code >= 32 && code <= 126) || code === 9 || code === 10 || code === 13) printable += 1;
        if (/[\[\]{}()<>:;,.!?'"`~@#$%^&*+=\\/|_-]/.test(ch)) punctuation += 1;
      }
      const length = chars.length;
      const printableRatio = printable / length;
      const punctuationRatio = punctuation / length;
      const spaceRatio = spaces / length;
      const hasToolMarkup =
        trimmed.includes('<tool_call>') ||
        (trimmed.includes('"name"') && trimmed.includes('"arguments"'));
      const hasWord = /\b[a-zA-Z]{3,}\b/.test(trimmed);
      const looksCorrupt = printableRatio >= 0.85 && punctuationRatio > 0.45 && spaceRatio < 0.01 && !hasToolMarkup;
      const admissible = hasToolMarkup || !looksCorrupt || hasWord;
      return {
        admissible,
        reason: admissible ? 'ok' : 'non_admissible_glyph_noise',
        metrics: {
          printableRatio: Number(printableRatio.toFixed(3)),
          punctuationRatio: Number(punctuationRatio.toFixed(3)),
          spaceRatio: Number(spaceRatio.toFixed(3)),
          length,
        },
      };
    }

    function buildPromptFromMessages(msgs) {
      const lines = [];
      for (const m of msgs || []) {
        if (m.role === 'tool') {
          lines.push(`<tool_result>${m.content || ''}</tool_result>`);
          continue;
        }
        if (m.role === 'assistant' && m.content == null && Array.isArray(m.tool_calls) && m.tool_calls.length) {
          const tc = m.tool_calls[0];
          const fnName = tc?.function?.name || tc?.name || '';
          let fnArgs = tc?.function?.arguments || tc?.arguments || '{}';
          if (typeof fnArgs !== 'string') fnArgs = JSON.stringify(fnArgs);
          lines.push(`<tool_call>{"name":"${fnName}","arguments":${fnArgs}}</tool_call>`);
          continue;
        }
        lines.push(`${m.role || 'user'}: ${typeof m.content === 'string' ? m.content : JSON.stringify(m.content || '')}`);
      }
      return lines.join('\n');
    }

    function normalizeLegacyToolCalls(toolCalls) {
      return (toolCalls || []).map((tc, idx) => {
        const fnName = tc?.function?.name || tc?.name || '';
        let fnArgs = tc?.function?.arguments || tc?.arguments || tc?.args || {};
        if (typeof fnArgs !== 'string') fnArgs = JSON.stringify(fnArgs);
        return {
          id: tc?.id || `legacy_${idx}`,
          type: 'function',
          function: { name: fnName, arguments: fnArgs },
        };
      });
    }

    async function callModel(targetUrl, msgs, timeoutMs = 90000) {
      return new Promise((resolve, reject) => {
        const mUrl  = new URL(targetUrl);
        const isLegacyChatEndpoint = mUrl.pathname === '/chat';
        const payload = isLegacyChatEndpoint
          ? {
              prompt: buildPromptFromMessages(msgs),
              domain: body.domain || body.fallback_domain || 'D5',
              max_tokens: maxTokens,
              temperature,
              stop: stopTokens,
            }
          : {
              messages: msgs,
              max_tokens: maxTokens,
              temperature,
              stream: false,
              stop: stopTokens,
            };
        const rb = JSON.stringify(payload);
        const mOpts = {
          hostname: mUrl.hostname, port: parseInt(mUrl.port || '80', 10),
          path: mUrl.pathname, method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(rb) },
          timeout: timeoutMs
        };
        const mReq = http.request(mOpts, mRes => {
          let data = '';
          mRes.on('data', c => { data += c; });
          mRes.on('end', () => {
            let parsed;
            try {
              parsed = JSON.parse(data);
            } catch (e) {
              reject(new Error('model JSON parse: ' + data.slice(0, 120)));
              return;
            }
            if ((mRes.statusCode || 500) >= 400) {
              const msg = parsed?.error?.message || parsed?.error || JSON.stringify(parsed).slice(0, 120);
              reject(new Error(`model ${mRes.statusCode}: ${msg}`));
              return;
            }
            if (isLegacyChatEndpoint) {
              const text = parsed?.response || parsed?.text || parsed?.output || '';
              const legacyToolCalls = normalizeLegacyToolCalls(parsed?.tool_calls);
              resolve({
                choices: [{
                  index: 0,
                  message: {
                    role: 'assistant',
                    content: text,
                    ...(legacyToolCalls.length ? { tool_calls: legacyToolCalls } : {}),
                  },
                  finish_reason: legacyToolCalls.length ? 'tool_calls' : 'stop',
                }],
                _legacy_response: parsed,
              });
              return;
            }
            resolve(parsed);
          });
        });
        mReq.on('error', reject);
        mReq.on('timeout', () => { mReq.destroy(); reject(new Error(`model timeout after ${Math.floor(timeoutMs / 1000)}s`)); });
        mReq.write(rb); mReq.end();
      });
    }

    const trace = [];
    const initialMessages = await injectNncKMemory(messages, lastUserContent(messages), 5);
    let currentMessages = [...initialMessages];
    let finalText = '';
    let loopsDone = 0;
    let modelCalls = 0;
    let activeModelUrl = modelUrl;
    let fallbackUsed = false;
    let fallbackAttempted = false;
    let lastAdmissibility = null;
    let forcedToolBootstrapUsed = false;
    let lastForcedToolName = null;
    let lastForcedToolArgs = null;
    let lastForcedToolResult = null;

    try {
      for (let loop = 0; loop < maxLoops; loop++) {
        loopsDone = loop + 1;
        modelCalls += 1;
        const resp   = await callModel(activeModelUrl, currentMessages, activeModelUrl === modelUrl ? 90000 : fallbackTimeoutMs);
        const choice = resp?.choices?.[0] || {};
        const text   = choice.message?.content || choice.text || '';

        // Record model output to JROM (backend-agnostic bus)
        try {
          appendJromEvent({
            tool_id: 'model.toolchat',
            verb: 'model.generate',
            target: activeModelUrl,
            args: { loop, model_calls: modelCalls, text_sample: text.slice(0, 2000) },
            phase: 'Wo',
            lane: 'CHATML',
            status: 'ok',
            result: { text: text.slice(0, 4000), tool_calls: choice.message?.tool_calls || null },
            model: activeModelUrl,
            micronaut: 'µ-tool',
            jrom_program: 'atomic_mcp'
          });
        } catch (e) { writeTrace('jrom.append.error', { source: 'toolchat.model_output', error: e.message }); }

        // Path A: structured tool_calls (chatml-capable models — Gemma etc.)
        if (choice.message?.tool_calls?.length) {
          const tc     = choice.message.tool_calls[0];
          const fnName = tc.function?.name || '';
          const fnArgs = (() => { try { return JSON.parse(tc.function?.arguments || '{}'); } catch(_) { return {}; } })();
          const result = execBuiltinTool(fnName, fnArgs)
                      ?? (await dispatchMcpTool(fnName, fnArgs).then(r => r?.content?.[0]?.text || '').catch(e => 'dispatch error: ' + e.message));
          trace.push({ loop, type: 'structured', tool: fnName, args: fnArgs, result });
          currentMessages = [
            ...currentMessages,
            { role: 'assistant', content: null, tool_calls: choice.message.tool_calls },
            { role: 'tool', tool_call_id: tc.id || ('tc_' + loop), content: result }
          ];
          continue;
        }

        // Path B: raw <tool_call>...</tool_call> markup (GPT-2 raw-text models)
        const match = TC_RE.exec(text);
        if (match) {
          const rawJson = (match[1] || match[2] || '').trim();
          let tcObj; try { tcObj = JSON.parse(rawJson); } catch(_) { tcObj = null; }
          if (tcObj) {
            const fnName = tcObj.name || tcObj.function || '';
            const fnArgs = tcObj.arguments || tcObj.args || {};
            const result = execBuiltinTool(fnName, fnArgs)
                        ?? (await dispatchMcpTool(fnName, fnArgs).then(r => r?.content?.[0]?.text || '').catch(e => 'dispatch error: ' + e.message));
            trace.push({ loop, type: 'raw_markup', tool: fnName, args: fnArgs, result });
            const prefix   = text.slice(0, match.index + match[0].length);
            const injected = prefix + '\n<tool_result>' + result + '</tool_result>';
            currentMessages = [
              ...currentMessages,
              { role: 'assistant', content: injected }
            ];
            continue;
          }
        }

        const admissibility = assessTextAdmissibility(text);
        lastAdmissibility = admissibility;

        // If the primary model emits non-admissible output, switch once to a known tool-call profile.
        if (enableFallback && fallbackModelUrl && !fallbackAttempted && activeModelUrl !== fallbackModelUrl && !admissibility.admissible) {
          fallbackAttempted = true;
          fallbackUsed = true;
          trace.push({
            loop,
            type: 'fallback_switch',
            from: activeModelUrl,
            to: fallbackModelUrl,
            reason: admissibility.reason,
            metrics: admissibility.metrics,
            sample: text.slice(0, 160),
          });
          activeModelUrl = fallbackModelUrl;
          loop -= 1; // retry same conversational turn on fallback model
          continue;
        }

        if (!admissibility.admissible) {
          trace.push({
            loop,
            type: 'non_admissible_output',
            reason: admissibility.reason,
            metrics: admissibility.metrics,
            sample: text.slice(0, 160),
          });
        }

        // Last-resort bridge: if model did not emit a tool call but user intent clearly asks for a known tool,
        // execute one forced tool round-trip once and retry the model with injected result.
        if (allowForcedToolBootstrap && !forcedToolBootstrapUsed) {
          const forcedCall = inferForcedToolCall(currentMessages);
          if (forcedCall) {
            const forcedResult = execBuiltinTool(forcedCall.name, forcedCall.arguments)
              ?? (await dispatchMcpTool(forcedCall.name, forcedCall.arguments)
                .then(r => r?.content?.[0]?.text || '')
                .catch(e => 'dispatch error: ' + e.message));
            forcedToolBootstrapUsed = true;
            lastForcedToolName = forcedCall.name;
            lastForcedToolArgs = forcedCall.arguments;
            lastForcedToolResult = forcedResult;
            trace.push({
              loop,
              type: 'forced_tool_bootstrap',
              tool: forcedCall.name,
              args: forcedCall.arguments,
              result: forcedResult,
              source: forcedCall.source,
              reason: 'model_no_tool_call',
            });
            currentMessages = [
              ...currentMessages,
              {
                role: 'assistant',
                content:
                  `<tool_call>${JSON.stringify({ name: forcedCall.name, arguments: forcedCall.arguments })}</tool_call>\n` +
                  `<tool_result>${String(forcedResult)}</tool_result>`
              }
            ];
            continue;
          }
        }

        // No tool call found — final answer
        finalText = text;
        break;
      }
    } catch (e) {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        error: e.message,
        trace,
        _active_model_url: activeModelUrl,
        _fallback_model_url: fallbackModelUrl,
        _fallback_used: fallbackUsed
      }));
      return;
    }

    if (forcedToolBootstrapUsed) {
      const finalAdmissibility = assessTextAdmissibility(finalText);
      if (!finalAdmissibility.admissible || !finalText || !String(finalText).trim()) {
        finalText = `${lastForcedToolName || 'tool'} result: ${String(lastForcedToolResult ?? '')}`;
        trace.push({
          loop: loopsDone,
          type: 'forced_tool_finalize',
          tool: lastForcedToolName,
          args: lastForcedToolArgs,
          result: lastForcedToolResult,
          reason: 'non_admissible_or_empty_final_text',
        });
      }
    }

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      id: 'toolchat-' + Date.now(),
      object: 'chat.completion',
      model: 'kuhul-tool-interceptor',
      choices: [{ index: 0, message: { role: 'assistant', content: finalText }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      _tool_dispatch_trace: trace,
      _loops: loopsDone,
      _model_url: modelUrl,
      _active_model_url: activeModelUrl,
      _fallback_model_url: fallbackModelUrl,
      _fallback_used: fallbackUsed,
      _fallback_attempted: fallbackAttempted,
      _model_calls: modelCalls,
      _last_admissibility: lastAdmissibility,
      _forced_tool_bootstrap_used: forcedToolBootstrapUsed,
      _forced_tool_name: lastForcedToolName,
      _tool_defs_count: Array.isArray(toolDefs) ? toolDefs.length : 0
    }));
    return;
  }

  // ── /v1/chat  — conversational mode (no code-gen instructions) ──────
  // ── /v1/code  — code-gen mode (injects forge system prompt) ─────────
  if (req.method === 'POST' && (p === '/v1/chat' || p === '/v1/code')) {
    const CHAT_SYS = "You are K'UHUL, a helpful assistant. Answer questions clearly and concisely. Only generate code blocks when explicitly asked.";
    const CODE_SYS = "You are K'UHUL, a vibe-coding intelligence. Write each file as a fenced code block with the filename after the language tag: ```html index.html\n...```. For web apps produce a self-contained index.html with inline CSS+JS when small, and separate style.css / script.js when the app warrants it. Files appear as tabs in the Monaco canvas. Keep prose minimal — ship working code.";

    let rawBody = '';
    await new Promise((ok, fail) => {
      req.on('data', c => { rawBody += c; });
      req.on('end', ok);
      req.on('error', fail);
    });
    let body;
    try { body = JSON.parse(rawBody); } catch (_) { body = {}; }

    // Total Recall: pull relevant persistent memory from /micronaut-v4 NNC-K
    const memoryMessages = await injectNncKMemory(body.messages || [], lastUserContent(body.messages || []), 5);

    // Inject or replace the system message with the route-appropriate prompt
    const sysContent = p === '/v1/code' ? CODE_SYS : CHAT_SYS;
    const existing = memoryMessages.filter(m => m.role !== 'system');
    const memoryContext = memoryMessages.find(m => m.role === 'system')?.content || '';
    const fullSystem = memoryContext ? `${sysContent}\n\n${memoryContext}` : sysContent;
    body.messages = [{ role: 'system', content: fullSystem }, ...existing];

    let inferPort = 25110;
    let inferModel = null;
    try {
      const amPath = path.join(PROJECT_ROOT, 'active-model.json');
      if (fs.existsSync(amPath)) {
        const am = JSON.parse(fs.readFileSync(amPath, 'utf8'));
        if (am.port && am.port !== REGISTRY_PORT) inferPort = am.port;
        if (am.model) inferModel = am.model;
      }
    } catch (_) {}

    if (inferModel) body.model = inferModel;
    const patched = JSON.stringify(body);

    // Record model request to JROM before proxying (backend-agnostic bus)
    const jromRequestId = 'jrom-' + crypto.randomUUID();
    try {
      appendJromEvent({
        tool_id: body.model || 'active-model',
        verb: 'chat.completions',
        target: `http://127.0.0.1:${inferPort}/v1/chat/completions`,
        args: { endpoint: p, messages: body.messages },
        phase: 'Wo',
        lane: 'CHATML',
        status: 'pending',
        replay_id: jromRequestId,
        model: body.model || 'active-model',
        micronaut: 'µ-chat',
        jrom_program: 'atomic_mcp'
      });
    } catch (e) { writeTrace('jrom.append.error', { source: '/v1/chat.request', error: e.message }); }

    const inferOpts = {
      hostname: '127.0.0.1', port: inferPort,
      path: '/v1/chat/completions', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(patched) }
    };
    let responseBuf = '';
    const proxyReq = http.request(inferOpts, proxyRes => {
      const h = { ...proxyRes.headers, 'access-control-allow-origin': '*' };
      res.writeHead(proxyRes.statusCode, h);
      proxyRes.on('data', chunk => { responseBuf += chunk; res.write(chunk); });
      proxyRes.on('end', () => {
        res.end();
        try {
          let status = 'ok', result = { raw: responseBuf.slice(0, 4000) };
          try { result.parsed = JSON.parse(responseBuf); } catch (_) {}
          if (proxyRes.statusCode >= 400) status = 'error';
          appendJromEvent({
            tool_id: body.model || 'active-model',
            verb: 'chat.completions',
            target: `http://127.0.0.1:${inferPort}/v1/chat/completions`,
            args: { endpoint: p, messages: body.messages },
            phase: 'Xul',
            lane: 'CHATML',
            status,
            result,
            error: status === 'error' ? responseBuf.slice(0, 1000) : undefined,
            replay_id: jromRequestId,
            model: body.model || 'active-model',
            micronaut: 'µ-chat',
            jrom_program: 'atomic_mcp'
          });
        } catch (e) { writeTrace('jrom.append.error', { source: '/v1/chat.response', error: e.message }); }
      });
      // Propagate client abort (halt button) up to the inference server so it
      // stops generating tokens instead of running to the end.
      res.on('close', () => proxyReq.destroy());
    });
    proxyReq.on('error', e => {
      try {
        appendJromEvent({
          tool_id: body.model || 'active-model',
          verb: 'chat.completions',
          target: `http://127.0.0.1:${inferPort}/v1/chat/completions`,
          args: { endpoint: p, messages: body.messages },
          phase: 'Xul',
          lane: 'CHATML',
          status: 'error',
          error: e.message,
          replay_id: jromRequestId,
          model: body.model || 'active-model',
          micronaut: 'µ-chat',
          jrom_program: 'atomic_mcp'
        });
      } catch (jerr) { writeTrace('jrom.append.error', { source: '/v1/chat.error', error: jerr.message }); }
      if (!res.headersSent) {
        res.writeHead(502, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'inference proxy error: ' + e.message, port: inferPort }));
      }
    });
    proxyReq.write(patched);
    proxyReq.end();
    return;
  }

  // ── Distillation provider routes ────────────────────────────────────
  if (req.method === 'POST' && p === '/kuhul/distil/run') {
    const opts = await readBody(req);
    if (_distilJob && !_distilJob.done) {
      res.writeHead(409, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'distil_job_already_running', step: _distilJob.step }));
      return;
    }
    _startDistilJob(opts);
    res.writeHead(202, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'started', poll: '/kuhul/distil/status' }));
    return;
  }

  if (req.method === 'GET' && p === '/kuhul/distil/status') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(_distilJob || { step: 0, done: false, error: null, ready: false }));
    return;
  }

  // ── UI rebuild — SSE stream of npm run build output ──────────────────────────
  // Called by the "Build & Reload" button in Settings > Developer.
  // After the stream ends with { type:'done', success:true }, the client clears
  // all SW/Workbox caches and reloads so the new build is served fresh.
  if (req.method === 'GET' && p === '/kuhul/ui/rebuild') {
    const sse = (obj) => `data: ${JSON.stringify(obj)}\n\n`;
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-store',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*',
      'X-Accel-Buffering': 'no',
    });

    const batPath = path.join(PROJECT_ROOT, 'update-ui.bat');
    if (!fs.existsSync(batPath)) {
      res.write(sse({ type: 'done', success: false, msg: 'update-ui.bat not found at ' + batPath }));
      res.end();
      return;
    }

    res.write(sse({ type: 'start', msg: 'Starting UI rebuild...' }));

    const proc = spawn('cmd.exe', ['/c', batPath], { cwd: PROJECT_ROOT });

    const fwd = (type) => (chunk) => {
      for (const line of chunk.toString().split('\n')) {
        const t = line.trim();
        if (t) res.write(sse({ type, msg: t }));
      }
    };
    proc.stdout.on('data', fwd('log'));
    proc.stderr.on('data', fwd('err'));

    proc.on('close', (code) => {
      res.write(sse({ type: 'done', success: code === 0, msg: code === 0 ? 'Build complete.' : `Build failed (exit ${code}).` }));
      res.end();
    });

    req.on('close', () => { try { proc.kill(); } catch (_) {} });
    return;
  }

  // ── Model Cache routes ────────────────────────────────────────────────────
  // GET  /model/cache          — index of local model file presence (C:\ vs E:\)
  // POST /model/cache/copy     — trigger background robocopy of a model entry to C:\
  // POST /model/cache/copy-dds — trigger background robocopy of all DDS shards to C:\

  if (req.method === 'GET' && p === '/model/cache') {
    const cache = _loadModelCache();
    if (!cache) {
      res.writeHead(503, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'model cache index not found', index: MODEL_CACHE_JSON }));
      return;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(cache));
    return;
  }

  if (req.method === 'POST' && p === '/model/cache/copy') {
    const body = await readBody(req);
    const { id } = body;
    const cache = _loadModelCache();
    const entry = cache && (cache.entries || []).find(e => e.id === id);
    if (!entry) {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'entry not found', id }));
      return;
    }
    if (entry.cached) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'already_cached', id, path: entry.path }));
      return;
    }
    if (!entry.e_path || !entry.c_path) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'entry has no e_path/c_path staging paths', id }));
      return;
    }
    _spawnCacheFile(entry.e_path, entry.c_path);
    res.writeHead(202, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'copying', id, from: entry.e_path, to: entry.c_path }));
    return;
  }

  if (req.method === 'POST' && p === '/model/cache/copy-dds') {
    const cache = _loadModelCache();
    const dds = cache && cache.dds_folds;
    if (!dds || !dds.e_shard_dir) {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'dds_folds not in cache index' }));
      return;
    }
    _spawnCacheDdsShards(dds.e_shard_dir, dds.c_shard_dir);
    res.writeHead(202, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'copying', shards: 91, from: dds.e_shard_dir, to: dds.c_shard_dir }));
    return;
  }

  // ── Quantization tier: query or swap active llama-server model ──────────────
  //   GET  /model/quant          → { active_model, active_tier, available }
  //   POST /model/quant          → { tier?, model_hint?, micronaut? } → swap + report
  if (p === '/model/quant') {
    if (req.method === 'GET') {
      const registry = loadRegistry();
      const available = {};
      for (const mn of registry) {
        if (!mn.quant_tier) continue;
        const resolved = resolveQuantModel(mn);
        available[mn.name] = {
          quant_tier: mn.quant_tier,
          model_hint: mn.model_hint || null,
          resolved_path: resolved.path,
          exists: fs.existsSync(resolved.path),
        };
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        active_model: _activeLlamaModel,
        server_ready: _llamaServerReady,
        quant_model_dir: QUANT_MODEL_DIR,
        available,
      }, null, 2));
      return;
    }

    if (req.method === 'POST') {
      const body = await readBody(req);
      let targetModel = null;

      if (body.model_path) {
        targetModel = body.model_path;
      } else if (body.micronaut) {
        const registry = loadRegistry();
        const mn = registry.find(m => m.name === body.micronaut);
        if (!mn) {
          res.writeHead(404, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'micronaut not found', name: body.micronaut }));
          return;
        }
        targetModel = resolveQuantModel(mn).path;
      } else if (body.tier) {
        const fb = QUANT_TIER_FALLBACK[body.tier];
        if (!fb) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'unknown tier', valid: Object.keys(QUANT_TIER_FALLBACK) }));
          return;
        }
        targetModel = fb;
      }

      if (!targetModel) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'provide tier, micronaut, or model_path' }));
        return;
      }

      if (!fs.existsSync(targetModel)) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'model file not found', path: targetModel }));
        return;
      }

      writeTrace('model.quant.swap', { from: _activeLlamaModel, to: targetModel });
      const ok = await _swapLlamaModel(targetModel);
      res.writeHead(ok ? 200 : 503, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok, active_model: _activeLlamaModel, server_ready: _llamaServerReady }));
      return;
    }
  }

  // ── Studio fabric API — projects / files / messages ──────────────────────
  if (req.method === 'GET' && p === '/api/projects') {
    const list = studio.projects
      .map(pr => ({ ...pr, messages: studio.messages[pr.id] || [] }))
      .sort((a, b) => (b.updated || 0) - (a.updated || 0));
    return sendJson(res, 200, list);
  }

  if (req.method === 'POST' && p === '/api/projects') {
    const body = await readBody(req);
    const id = 'p' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    const now = Date.now();
    studio.projects.push({ id, title: body.title || 'untitled fabric', created: now, updated: now });
    studio.files[id]    = studio.files[id]    || [];
    studio.messages[id] = studio.messages[id] || [];
    studioSave();
    return sendJson(res, 200, { id });
  }

  if (req.method === 'DELETE' && p.startsWith('/api/projects/')) {
    const id = p.slice('/api/projects/'.length);
    studio.projects = studio.projects.filter(x => x.id !== id);
    delete studio.files[id];
    delete studio.messages[id];
    studioSave();
    return sendJson(res, 200, { success: true });
  }

  if (req.method === 'GET' && p.startsWith('/api/files/')) {
    const id = p.slice('/api/files/'.length);
    return sendJson(res, 200, studio.files[id] || []);
  }

  if (req.method === 'POST' && p === '/api/files') {
    const body = await readBody(req);
    const projectId = body.projectId;
    const arr = studio.files[projectId] || (studio.files[projectId] = []);
    const name = body.name;
    const lang = body.lang || 'plaintext';
    const content = body.content || '';
    const ex = arr.find(f => f.name === name);
    const now = Date.now();
    let id;
    if (ex) { id = ex.id; ex.lang = lang; ex.content = content; ex.updated = now; }
    else {
      id = 'f' + Date.now().toString(36) + Math.random().toString(36).slice(2, 5);
      arr.push({ id, name, lang, content, updated: now });
    }
    studioTouchProject(projectId);
    studioSave();
    return sendJson(res, 200, { id });
  }

  if (req.method === 'GET' && p.startsWith('/api/messages/')) {
    const id = p.slice('/api/messages/'.length);
    return sendJson(res, 200, studio.messages[id] || []);
  }

  if (req.method === 'POST' && p === '/api/messages') {
    const body = await readBody(req);
    const projectId = body.projectId;
    const arr = studio.messages[projectId] || (studio.messages[projectId] = []);
    arr.push({ role: body.role || 'user', content: body.content || '', ts: Date.now() });
    studioTouchProject(projectId);
    studioSave();
    return sendJson(res, 200, { ok: true });
  }

  // ── /v1/forge — micronaut tool escalation ──────────────────────────────────
  // Executes the real coder binaries/scripts (review/todos/refactor/v6-hashes/
  // ...) against on-disk files. A tool-calling model may invoke these; the
  // gateway does the actual subprocess work and returns real output.
  if (req.method === 'GET' && p === '/v1/forge/tools') {
    const list = Object.entries(FORGE_TOOLS).map(([name, t]) => ({
      name, description: t.desc, params: t.params,
      kind: t.node ? 'node-script' : 'native-exe',
    }));
    return sendJson(res, 200, { count: list.length, tools: list });
  }

  if (req.method === 'POST' && p === '/v1/forge') {
    const body = await readBody(req);
    if (!body || !body.tool) return sendJson(res, 400, { ok: false, error: 'tool required' });
    if (!FORGE_TOOLS[body.tool]) return sendJson(res, 400, { ok: false, error: 'unknown_tool', tool: body.tool });
    const result = await runForgeTool(body.tool, body.args || {}, body.timeout_ms);
    return sendJson(res, result.ok ? 200 : 500, result);
  }

  // ── /v1/verbs — SVO action vocabulary (KXML `verb` tool made real) ─────────
  if (req.method === 'GET' && p === '/v1/verbs') {
    const list = Object.entries(VERBS).map(([name, v]) => ({
      verb: name,
      tool: v.tool,
      slots: Object.entries(v.args).map(([arg, slot]) => ({ arg, slot })),
      description: (FORGE_TOOLS[v.tool] || {}).desc || '',
    }));
    return sendJson(res, 200, { count: list.length, verbs: list });
  }

  if (req.method === 'POST' && p === '/v1/forge/verb') {
    const body = await readBody(req);
    const name = body && body.name;
    if (!name) return sendJson(res, 400, { ok: false, error: 'verb name required' });
    if (!VERBS[name]) return sendJson(res, 400, { ok: false, error: 'unknown_verb', verb: name });
    const result = await runVerb(name, body.subject, body.object, body.timeout_ms);
    return sendJson(res, result.ok ? 200 : 500, result);
  }

  // ── /v1/forge/chat — tool interceptor LOOP ────────────────────────────────
  // model emits a tool/verb call → gateway executes it → result is injected
  // back into the conversation → model continues. Repeats until no more calls
  // or max_iters. Closes the "calls the tool but doesn't write it back" gap.
  if (req.method === 'POST' && p === '/v1/forge/chat') {
    const body = await readBody(req);
    const modelUrl = (body && body.model_url) || activeModelChatUrl();
    let messages = Array.isArray(body && body.messages) ? body.messages : [];
    const maxIters = (body && body.max_iters) || 5;
    const extra = { ...((body && body.sampling) || {}) };
    if (body && Array.isArray(body.tools)) extra.tools = body.tools;
    if (body && body.tool_choice) extra.tool_choice = body.tool_choice;
    if (body && body.max_tokens) extra.max_tokens = body.max_tokens;
    const trace = [];
    try {
      // Raw-text completion path — for the fine-tuned mm-toolcall GPT-2, which
      // has no chat handler and emits <tool_call> in free-form completion text.
      if (body && (body.mode === 'completion' || body.raw)) {
        let prompt = (body && body.prompt) || renderRaw(messages);
        for (let i = 0; i < maxIters; i++) {
          const r = await rawCompletion(modelUrl, prompt, extra, body && body.timeout_ms);
          if (r.status !== 200) {
            return sendJson(res, 502, { ok: false, error: 'model_error', status: r.status, detail: r.json, trace });
          }
          const text = (r.json.choices && r.json.choices[0] && r.json.choices[0].text) || '';
          const calls = parseCalls({ content: text });
          if (!calls.length) {
            return sendJson(res, 200, { ok: true, final: text.trim(), iterations: i + 1, trace });
          }
          for (const c of calls) {
            const result = await dispatchCall(c);
            trace.push({ call: c, result });
            prompt += '\nTOOL: ' + JSON.stringify(result) + '\nASSISTANT:';
          }
        }
        return sendJson(res, 200, { ok: true, final: '(max iterations reached)', iterations: maxIters, trace });
      }

      // Chat/completions path (structured tool_calls or raw <tool_call> text).
      // Force tool_choice on the FIRST turn so the model emits a call; after the
      // result is injected, drop tools/tool_choice so it can answer directly.
      for (let i = 0; i < maxIters; i++) {
        const callExtra = { ...extra };
        if (i > 0) { delete callExtra.tool_choice; delete callExtra.tools; }
        const r = await modelChat(modelUrl, messages, callExtra, body && body.timeout_ms);
        if (r.status !== 200) {
          return sendJson(res, 502, { ok: false, error: 'model_error', status: r.status, detail: r.json, trace });
        }
        const msg = r.json.choices?.[0]?.message || {};
        const calls = parseCalls(msg);
        if (!calls.length) {
          return sendJson(res, 200, { ok: true, final: msg.content || '', iterations: i + 1, trace });
        }
        for (const c of calls) {
          const result = await dispatchCall(c);
          trace.push({ call: c, result });
          // Inject a CLEAN assistant message for continuation: strip the
          // deprecated function_call field; content must be a string (some
          // backends reject content:null alongside tool_calls).
          const assistantMsg = { role: 'assistant', content: typeof msg.content === 'string' ? msg.content : '' };
          if (Array.isArray(msg.tool_calls) && msg.tool_calls.length) {
            assistantMsg.tool_calls = msg.tool_calls.map(tc => ({ id: tc.id, type: 'function', function: tc.function }));
          }
          messages.push(assistantMsg);
          messages.push(c.id
            ? { role: 'tool', tool_call_id: c.id, content: JSON.stringify(result) }
            : { role: 'tool', content: JSON.stringify(result) });
        }
      }
      return sendJson(res, 200, { ok: true, final: '(max iterations reached)', iterations: maxIters, trace });
    } catch (e) {
      return sendJson(res, 500, { ok: false, error: 'interceptor_error', message: e.message, trace });
    }
  }

  // ── Static UI — host-routed multi-project serving ───────────────────────────
  // Resolves dist folder from Host header via UI_BY_HOST.
  // Adding a new UI project: build it into any folder, add one line to
  // UI_BY_HOST above, add a Cloudflare tunnel route subdomain → localhost:8764.
  //
  // Cache strategy (llama.cpp --path is NOT used — it bakes stale headers):
  //   /_app/**  → immutable  (SvelteKit content-hashes these filenames)
  //   everything else → no-store  (index.html must always be fresh)
  const host = (req.headers['host'] || '').split(':')[0].toLowerCase();
  const uiDist = UI_BY_HOST[host] || UI_DEFAULT;

  if (req.method === 'GET' && fs.existsSync(path.join(uiDist, 'index.html'))) {
    const UI_MIME = {
      '.html': 'text/html; charset=utf-8',
      '.js':   'application/javascript; charset=utf-8',
      '.mjs':  'application/javascript; charset=utf-8',
      '.css':  'text/css; charset=utf-8',
      '.json': 'application/json',
      '.webmanifest': 'application/manifest+json',
      '.png':  'image/png',
      '.jpg':  'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.gif':  'image/gif',
      '.svg':  'image/svg+xml',
      '.ico':  'image/x-icon',
      '.woff': 'font/woff',
      '.woff2': 'font/woff2',
      '.txt':  'text/plain',
    };

    // Resolve to a real file; fall back to index.html for SPA routing.
    let filePath = path.join(uiDist, p === '/' ? 'index.html' : p.split('?')[0]);
    if (!filePath.startsWith(uiDist)) filePath = path.join(uiDist, 'index.html');
    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      filePath = path.join(uiDist, 'index.html');
    }

    const ext = path.extname(filePath).toLowerCase();
    const mime = UI_MIME[ext] || 'application/octet-stream';
    const cacheControl = p.startsWith('/_app/')
      ? 'public, max-age=31536000, immutable'
      : 'no-store, no-cache, must-revalidate';

    try {
      const data = fs.readFileSync(filePath);
      res.writeHead(200, {
        'Content-Type':   mime,
        'Cache-Control':  cacheControl,
        'Content-Length': data.length,
      });
      res.end(data);
    } catch (e) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'ui_serve_error', detail: e.message }));
    }
    return;
  }

  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'not found', path: p }));
});

// =============================================================================
// Distillation — TaskEngine-driven provider loop
// =============================================================================

let _distilJob = null; // { step, best_loss, done, error, saved }

function _httpPost(url, body) {
  return new Promise((resolve) => {
    const payload = JSON.stringify(body);
    const u = new URL(url);
    const opts = {
      hostname: u.hostname, port: parseInt(u.port) || 80,
      path: u.pathname, method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
    };
    const mod = u.protocol === 'https:' ? https : http;
    const req = mod.request(opts, (res) => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => { try { resolve(JSON.parse(data)); } catch { resolve({ raw: data }); } });
    });
    req.on('error', e => resolve({ error: e.message }));
    req.setTimeout(300000, () => { req.destroy(); resolve({ error: 'timeout' }); });
    req.write(payload);
    req.end();
  });
}

const DISTIL_PROVIDER = 'http://127.0.0.1:1236';

const DISTIL_DEFAULT_PROMPTS = [
  "Explain the K'UHUL fold system and how it organizes semantic execution phases.",
  "Describe the role of the BOSS orchestrator in the WebX compute engine.",
  "How does the DirectML GEMM bridge accelerate matrix operations on Intel HD 4600?",
  "What is SCXQ2 IR and how does it represent the K'UHUL execution graph?",
  "How do semantic micronauts influence llama-server sampling parameters?",
  "What is the difference between the TaskEngine planner and the MCP executor?",
  "Describe the LoRA distillation pipeline from GPT-OSS to from_zero.",
  "How does the micronaut factory create and evolve micronaut profiles?",
  "What are atomic blocks in the K'UHUL semantic field system?",
  "How does the PRIMEOS shell communicate with kuhul_engine via shared memory?",
];

// Hive route: micronaut name → MM-* specialist URL (POST /chat {prompt, domain})
// Entries with null use _superChat (POST /orchestrate/execute on port 5774) instead of _hiveChat
const HIVE_ROUTE = {
  khanary:            null,                       // → supernaut orchestrator (5774) via _superChat
  stack_doc:          'http://127.0.0.1:25503',  // mm-kuhul — K'UHUL/fold/semantic
  distillation_guide: 'http://127.0.0.1:25502',  // mm-toolcall — reasoning
  scx_guide:          'http://127.0.0.1:25503',  // mm-kuhul — scxq2/xcfe
  asx_guide:          'http://127.0.0.1:25502',  // mm-toolcall — physics/routing
  primeos_guide:      'http://127.0.0.1:25507',  // mm-design — UI/app shell
  coder:              'http://127.0.0.1:25500',  // mm-coder
  ui:                 'http://127.0.0.1:25507',  // mm-design
  tool_call:          'http://127.0.0.1:25502',  // mm-toolcall
  memory:             'http://127.0.0.1:25502',  // mm-toolcall
  librarian:          'http://127.0.0.1:25504',  // mm-research
  chat:               null,                       // → supernaut orchestrator (5774) via _superChat
  factory:            'http://127.0.0.1:25505',  // mm-agent
  evolution:          'http://127.0.0.1:25505',  // mm-agent
  compiled_model:     'http://127.0.0.1:25503',  // mm-kuhul — compiled model Q&A
};

// Domain codes passed to MM-* /chat endpoint
const MICRONAUT_DOMAIN = {
  khanary: 'D0', stack_doc: 'D0', distillation_guide: 'D0',
  scx_guide: 'D0', asx_guide: 'D0', primeos_guide: 'D8',
  coder: 'D1', ui: 'D8', tool_call: 'D5', memory: 'D0',
  librarian: 'D2', chat: 'D0', factory: 'D6', evolution: 'D6',
  compiled_model: 'D0',
};

// Compiled micronaut service endpoints (coordinator/factory/executor from E:\models\micronaut\micronaut)
const COMPILED_MNAUT = {
  coordinator: 'http://127.0.0.1:25100',  // CO-1: GET /registry POST /route POST /register
  factory:     'http://127.0.0.1:25101',  // FG-1: POST /create POST /register GET /services
  executor:    'http://127.0.0.1:25103',  // EX-1: POST /execute GET /jobs
  worker:      'http://127.0.0.1:5010',   // DOTNET-WORKER-1: POST /run (math/SIMD/tensor)
};

// Supernaut orchestration stack (dist/supernaut-cpp/)
const SUPERNAUT_ORCHESTRATOR = 'http://127.0.0.1:5774';  // run.mjs — POST /orchestrate/execute
const SUPERNAUT_S7_API       = 'http://127.0.0.1:5775';  // supernaut_api.py — POST /generate
const SUPERNAUT_ASXR         = 'http://127.0.0.1:5776';  // supernaut_native.exe — ASXR substrate

// Call the supernaut Phase 7.5 orchestrator. Returns completion string or null.
async function _superChat(prompt, systemContext) {
  const query = systemContext ? `${systemContext}\n\n${prompt}` : prompt;
  const result = await _httpPost(SUPERNAUT_ORCHESTRATOR + '/orchestrate/execute', {
    query,
    topN: 3,
    strategy: 'weighted',
    timeout: 8000
  });
  if (result.error || !result.result) return null;
  return typeof result.result === 'string' ? result.result.trim() : JSON.stringify(result.result);
}

// Call a hive MM-* specialist. Returns completion string or null if unavailable.
async function _hiveChat(hiveUrl, prompt, domain, systemContext) {
  const body = systemContext ? `${systemContext}\n\n${prompt}` : prompt;
  const result = await _httpPost(hiveUrl + '/chat', { prompt: body, domain: domain || 'D0' });
  if (result.error || result.status === 'error' || !result.response) return null;
  return result.response;
}

// Call kuhul_engine local inference. Engine uses {model, prompt, max_tokens} format,
// not OpenAI messages format. Valid local teacher before cloud.
async function _engineChat(prompt, systemContext, maxTokens) {
  if (!state.engineUp) state.engineUp = await isPortOpen(state.enginePort);
  if (!state.engineUp) return null;
  const fullPrompt = systemContext ? `${systemContext}\n\n${prompt}` : prompt;
  const modelPath  = _resolveModelPath('gpt-oss-20b') || ENGINE_MODEL;
  const payload = JSON.stringify({
    model:      modelPath,
    prompt:     fullPrompt,
    max_tokens: maxTokens || 128,
  });
  return new Promise((resolve) => {
    const req = http.request({
      hostname: '127.0.0.1', port: state.enginePort,
      path: '/v1/chat/completions', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
    }, (res) => {
      let d = '';
      res.on('data', c => d += c);
      res.on('end', () => {
        try {
          const j = JSON.parse(d);
          // Engine returns either choices[].message.content or top-level text/content
          const text = j.choices?.[0]?.message?.content
                    || j.choices?.[0]?.text
                    || j.text
                    || j.content
                    || '';
          resolve(text.trim());
        } catch { resolve(''); }
      });
    });
    req.on('error', () => { state.engineUp = false; resolve(''); });
    req.setTimeout(30_000, () => { req.destroy(); state.engineUp = false; resolve(''); });
    req.write(payload);
    req.end();
  }).then(r => (r && r.trim()) ? r.trim() : null);
}

// Local llama-server teacher (OpenAI-compatible, port 17481, gemma-3-1B).
// kuhul_engine can't load GGUF (binary doesn't call ggml_backend_load_all()); llama-server does.
// modelPath: explicit GGUF path; omit to use the current active model or default.
async function _ensureLlamaServer(modelPath) {
  const mpath = modelPath || _activeLlamaModel || LLAMA_SERVER_MODEL;
  if (_llamaServerReady && _activeLlamaModel === mpath) return true;
  if (await isPortOpen(LLAMA_SERVER_PORT)) {
    _llamaServerReady = true;
    if (!_activeLlamaModel) _activeLlamaModel = mpath;
    return true;
  }
  if (!fs.existsSync(LLAMA_SERVER_EXE) || !fs.existsSync(mpath)) return false;
  llamaServerProcess = spawn(LLAMA_SERVER_EXE,
    ['-m', mpath, '--port', String(LLAMA_SERVER_PORT), '--host', '127.0.0.1', '-ngl', '99', '--log-disable'],
    { detached: false, stdio: 'ignore', windowsHide: true, cwd: LLAMA_SERVER_DIR });
  _activeLlamaModel = mpath;
  llamaServerProcess.on('exit', () => { _llamaServerReady = false; llamaServerProcess = null; _activeLlamaModel = null; });
  // Wait up to 30 s for it to bind
  for (let i = 0; i < 30; i++) {
    await new Promise(r => setTimeout(r, 1000));
    if (await isPortOpen(LLAMA_SERVER_PORT)) { _llamaServerReady = true; return true; }
  }
  return false;
}

async function _llamaChat(prompt, systemContext, maxTokens) {
  if (!await _ensureLlamaServer()) return null;
  const messages = systemContext
    ? [{ role: 'system', content: systemContext }, { role: 'user', content: prompt }]
    : [{ role: 'user', content: prompt }];
  const payload = JSON.stringify({ messages, max_tokens: maxTokens || 128 });
  return new Promise((resolve) => {
    const req = http.request({
      hostname: '127.0.0.1', port: LLAMA_SERVER_PORT,
      path: '/v1/chat/completions', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
    }, (res) => {
      let d = '';
      res.on('data', c => d += c);
      res.on('end', () => {
        try {
          const j = JSON.parse(d);
          const text = j.choices?.[0]?.message?.content || j.choices?.[0]?.text || '';
          resolve(text.trim());
        } catch { resolve(''); }
      });
    });
    req.on('error', () => { _llamaServerReady = false; resolve(''); });
    req.setTimeout(60_000, () => { req.destroy(); _llamaServerReady = false; resolve(''); });
    req.write(payload);
    req.end();
  }).then(r => (r && r.trim()) ? r.trim() : null);
}

// ── Model Cache ───────────────────────────────────────────────────────────────
// Tracks which model weight files are resident on C:\ (fast SSD) vs E:\ (slow HDD).
// Cloud Ollama is only used when all local inference paths fail.
//
// DDS shards: xvm_d12_host.exe loads from whichever path is listed in the index.
// C:\ copies load significantly faster than E:\ on this machine.

const MODEL_CACHE_DIR  = 'C:\\model-cache';
const MODEL_CACHE_JSON = path.join(MODEL_CACHE_DIR, 'index.json');

let _modelCacheMtime = 0;
let _modelCacheData  = null;

function _loadModelCache() {
  try {
    const stat = fs.statSync(MODEL_CACHE_JSON);
    if (!_modelCacheData || stat.mtimeMs > _modelCacheMtime) {
      const raw = JSON.parse(fs.readFileSync(MODEL_CACHE_JSON, 'utf8'));
      // cached = file exists at primary path (wherever the model lives)
      for (const e of (raw.entries || [])) {
        e.cached = !!(e.path && fs.existsSync(e.path));
      }
      if (raw.dds_folds) {
        raw.dds_folds.manifests_cached = !!(raw.dds_folds.c_manifest && fs.existsSync(raw.dds_folds.c_manifest));
        const shardDir = raw.dds_folds.c_shard_dir;
        if (shardDir && fs.existsSync(shardDir)) {
          raw.dds_folds.shards_on_c = fs.readdirSync(shardDir).filter(f => f.endsWith('.dds')).length;
        } else {
          raw.dds_folds.shards_on_c = 0;
        }
      }
      _modelCacheData  = raw;
      _modelCacheMtime = stat.mtimeMs;
    }
  } catch (_) {}
  return _modelCacheData;
}

// Return the primary path for a model (wherever it lives — LM Studio, E:\, or staged C:\).
function _resolveModelPath(entryId) {
  const cache = _loadModelCache();
  if (!cache) return null;
  const e = (cache.entries || []).find(x => x.id === entryId);
  if (!e) return null;
  return e.path || e.c_path || e.e_path || null;
}

// Spawn a background robocopy to copy a single file to C:\model-cache\.
// Returns immediately; caller can poll /model/cache to see when cached=true.
function _spawnCacheFile(ePath, cPath) {
  const destDir = path.dirname(cPath);
  if (!fs.existsSync(destDir)) fs.mkdirSync(destDir, { recursive: true });
  const child = spawn('robocopy', [
    path.dirname(ePath),
    destDir,
    path.basename(ePath),
    '/NDL', '/NJH', '/NJS', '/NC', '/NS'
  ], { detached: true, stdio: 'ignore', windowsHide: true });
  child.unref();
  // Invalidate cache so next read re-probes file presence
  _modelCacheMtime = 0;
}

// Copy entire DDS shard directory to C:\model-cache\dds_shards\ via robocopy.
function _spawnCacheDdsShards(eShardDir, cShardDir) {
  if (!fs.existsSync(cShardDir)) fs.mkdirSync(cShardDir, { recursive: true });
  const child = spawn('robocopy', [
    eShardDir, cShardDir,
    '*.dds',
    '/MT:4', '/NDL', '/NJH', '/NJS', '/NC', '/NS'
  ], { detached: true, stdio: 'ignore', windowsHide: true });
  child.unref();
  _modelCacheMtime = 0;
}

function _startDistilJob(opts) {
  // Load DAG to extract loop params + validate via driver
  const dagPath = path.join(PROJECT_ROOT, 'drivers', 'DistillationDAG.TaskList.json');
  let dag = {};
  try { dag = JSON.parse(fs.readFileSync(dagPath, 'utf8')); } catch (_) {}
  const loop = dag.loop || {};

  if (driverDll) {
    const dagForDriver = JSON.stringify({ tasks: dag.tasks || [] });
    const h = getDriverHandle();
    if (h) {
      const errBuf = Buffer.alloc(512);
      const ok = driverDll.kd_load_tasks(h, dagForDriver, errBuf, errBuf.length);
      if (ok) {
        const plan = driverDll.kd_plan(h);
        writeTrace('distil.dag_plan', { plan: plan ? JSON.parse(plan) : null });
        if (plan) driverDll.kd_free_string(plan);
      }
    }
  }

  const student      = opts.student      || loop.student      || 'models/from_zero/from_zero_v0.6_merged.safetensors';
  const outLora      = opts.out_lora     || loop.out_lora     || 'models/from_zero/from_zero_v0.6_lora.safetensors';
  const steps        = opts.steps        || loop.steps        || 500;
  const rank         = opts.rank         || loop.rank         || 8;
  const lr           = opts.lr           || loop.lr           || 1e-4;
  const ollamaUrl    = opts.ollama_url   || loop.ollama_url   || 'http://127.0.0.1:11434';
  const ollamaModel  = opts.ollama_model || loop.ollama_model || 'gpt-oss:120b-cloud';
  const maxTokens    = opts.teacher_tokens || loop.teacher_tokens || 128;

  _distilJob = { step: 0, best_loss: null, done: false, error: null, saved: null };

  (async () => {
    try {
      // Init python-distil provider (POST /init)
      const initResult = await _httpPost(DISTIL_PROVIDER + '/init', { student, out_lora: outLora, rank, lr });
      if (initResult.error) throw new Error('distil_provider /init failed: ' + initResult.error);
      writeTrace('distil.init', initResult);

      for (let s = 0; s < steps; s++) {
        const prompt = DISTIL_DEFAULT_PROMPTS[s % DISTIL_DEFAULT_PROMPTS.length];

        // Task 1: micronaut_xquery (kuhul-server internal — selectMicronaut + read system)
        const selected = selectMicronaut(prompt);
        let systemContext = null;
        const mnFile = path.join(PROJECT_ROOT, 'micronauts', selected.name + '.json');
        if (fs.existsSync(mnFile)) {
          try { systemContext = JSON.parse(fs.readFileSync(mnFile, 'utf8')).system || null; } catch (_) {}
        }

        // Task 2: teacher_sample — local-first chain before cloud:
        //   supernaut:5774 → MM-* hive → kuhul_engine:17480 → cloud Ollama
        const hiveUrl    = HIVE_ROUTE[selected.name];
        const hiveDomain = MICRONAUT_DOMAIN[selected.name] || 'D0';

        let completion  = await _superChat(prompt, systemContext);
        let teacherMode = (completion && completion.trim()) ? 'supernaut:5774' : null;

        if (!teacherMode) {
          completion  = hiveUrl ? await _hiveChat(hiveUrl, prompt, hiveDomain, systemContext) : null;
          teacherMode = (completion && completion.trim()) ? ('hive:' + selected.name) : null;
        }

        if (!teacherMode) {
          completion  = await _engineChat(prompt, systemContext, maxTokens);
          teacherMode = (completion && completion.trim()) ? ('engine:' + state.enginePort) : null;
        }

        if (!teacherMode) {
          completion  = await _llamaChat(prompt, systemContext, maxTokens);
          teacherMode = (completion && completion.trim()) ? ('llama:' + LLAMA_SERVER_PORT) : null;
        }

        if (!teacherMode) teacherMode = 'ollama';

        if (!completion || !completion.trim()) {
          completion = await (async () => {
            const payload = JSON.stringify({
              model: ollamaModel,
              messages: [
                { role: 'system', content: systemContext || 'Answer briefly and directly.' },
                { role: 'user',   content: prompt },
              ],
              stream: false,
              options: { num_predict: maxTokens, temperature: 0.7, reasoning: false },
            });
            return new Promise((resolve) => {
              const u = new URL(ollamaUrl + '/api/chat');
              const req = http.request({
                hostname: u.hostname, port: parseInt(u.port) || 11434,
                path: u.pathname, method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
              }, (res2) => {
                let d = '';
                res2.on('data', c => d += c);
                res2.on('end', () => {
                  try { const j = JSON.parse(d); resolve(j.message?.content || j.response || ''); }
                  catch { resolve(''); }
                });
              });
              req.on('error', () => resolve(''));
              req.setTimeout(300000, () => { req.destroy(); resolve(''); });
              req.write(payload);
              req.end();
            });
          })();
        }

        if (!completion || !completion.trim()) {
          _distilJob.step = s + 1;
          continue;
        }

        // Task 3: distil_step (python-distil provider POST /step)
        const stepResult = await _httpPost(DISTIL_PROVIDER + '/step', { prompt, completion });
        if (stepResult.error) {
          writeTrace('distil.step_error', { step: s + 1, error: stepResult.error, teacher: teacherMode });
        } else {
          _distilJob.step      = stepResult.step      || s + 1;
          _distilJob.best_loss = stepResult.best_loss || null;
          writeTrace('distil.step', { step: s + 1, micronaut: selected.name, loss: stepResult.loss, teacher: teacherMode });
          if ((s + 1) % 10 === 0) {
            console.log(`[distil] step ${s+1}/${steps}  loss=${stepResult.loss?.toFixed(4)}  micronaut=${selected.name}  teacher=${teacherMode}`);
          }
        }

        // Task 4: factory_store (micronaut-factory — write Q&A to semantic dir)
        const semDir = path.join(PROJECT_ROOT, 'micronauts', 'semantic');
        if (!fs.existsSync(semDir)) fs.mkdirSync(semDir, { recursive: true });
        const qaFile = path.join(semDir, selected.name + '_distil.json');
        let qaData = { name: selected.name, qa_pairs: [] };
        if (fs.existsSync(qaFile)) {
          try { qaData = JSON.parse(fs.readFileSync(qaFile, 'utf8')); } catch (_) {}
        }
        if (!Array.isArray(qaData.qa_pairs)) qaData.qa_pairs = [];
        qaData.qa_pairs.push({ prompt, completion, step: s + 1, timestamp: new Date().toISOString() });
        fs.writeFileSync(qaFile, JSON.stringify(qaData, null, 2));

        writeTrace('distil.step', { step: s + 1, micronaut: selected.name, loss: stepResult.loss });
      }

      // Save LoRA
      const saved = await _httpPost(DISTIL_PROVIDER + '/save', { out_lora: outLora });
      _distilJob.done  = true;
      _distilJob.saved = saved;
      console.log(`[distil] done. steps=${steps}  best_loss=${_distilJob.best_loss?.toFixed(4)}  out=${outLora}`);
      writeTrace('distil.done', { steps, best_loss: _distilJob.best_loss, out_lora: outLora });

    } catch (e) {
      _distilJob.error = e.message;
      _distilJob.done  = true;
      console.error('[distil] job failed:', e.message);
    }
  })();
}

// =============================================================================
// Start sequence
// =============================================================================
server.listen(REGISTRY_PORT, '127.0.0.1', async () => {
  const actualPort = server.address().port;
  state.actualPort = actualPort;
  // Write port to file so external processes can discover it
  try { fs.writeFileSync(PORT_FILE, String(actualPort)); } catch (_) {}
  console.log('[kuhul-server] listening on http://127.0.0.1:' + actualPort);
  console.log('[kuhul-server] port file   -> ' + PORT_FILE);
  console.log('[kuhul-server] micronauts  -> /micronauts');
  console.log('[kuhul-server] MCP server  -> POST /mcp  |  GET /mcp/sse');
  console.log('[kuhul-server] MCP tools   -> GET /mcp/tools');
  console.log('[kuhul-server] engine      -> GET /kuhul/engine/status');
  console.log('[kuhul-server] grammar     -> GET /kuhul/grammar');
  writeTrace('start', { registryPort: actualPort, portFile: PORT_FILE });
  recordMicronautMetadataToJrom('startup');
  syncV4JromToCentral();
  await ensureEngineRunning();
});

// =============================================================================
// Tick Loop
// =============================================================================
setInterval(async () => {
  await checkEngineHealth();
  syncV4JromToCentral();
  writeTrace('tick', { uptime: process.uptime(), engineUp: state.engineUp });
  // Auto-restart engine if it went down
  if (!state.engineUp && !engineProcess) {
    console.log('[kuhul-server] engine down — attempting restart...');
    await ensureEngineRunning();
  }
}, 60000);

// =============================================================================
// Shutdown
// =============================================================================
process.on('SIGINT', () => {
  writeTrace('shutdown', { reason: 'SIGINT' });
  console.log('\nShutting down...');
  if (engineProcess) {
    try { engineProcess.kill(); } catch (_) {}
  }
  process.exit(0);
});

process.on('SIGTERM', () => {
  if (engineProcess) try { engineProcess.kill(); } catch (_) {}
  process.exit(0);
});
