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
const NNC_K_WEBX = 'C:\\Users\\canna\\.NNC-K\\bin\\v3.5.0-WebX';
const NNC_K_BIN  = path.join(NNC_K_WEBX, 'build-llama', 'bin', 'Release');
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
    windowsHide: true
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
  stack_doc:        ['stack', 'kuhul', 'khanary', 'system', 'architecture', 'how does', 'what is', 'native', 'driver', 'dll', 'micronauts', 'registry'],
  primeos_guide:    ['primeos', 'desktop', 'wpf', 'app shell', 'window', 'model selector', 'webview2', 'winui'],
  scx_guide:        ['scx', 'scxq2', 'bytecode', 'compile', 'decompile', 'runtime', 'opcode', '@op', 'instruction', 'xcfe'],
  asx_guide:        ['asx', 'gravity', 'entropy', 'pressure', 'physics', 'routing', 'attention', 'state', 'shared memory'],
  distillation_guide: ['distill', 'lora', 'teacher', 'train', 'fine-tune', 'from_zero', 'safetensors', 'oss_distillation'],
  tool_call:        ['tool', 'function call', 'call', 'invoke', 'mcp', 'json-rpc', 'kuhul_task_boss'],
  chat:             ['chat', 'talk', 'hello', 'hi', 'hey', 'conversation'],
};

const ATOMIC_BLOCKS = ['HEADER', 'MENU', 'BODY', 'FEED', 'FOOTER'];

function selectMicronaut(prompt) {
  const text = (prompt || '').toLowerCase();
  const registry = loadRegistry();
  let best = { name: 'khanary', confidence: 0.5, reason: 'default', blocks: ['BODY'] };
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
          reason: 'keyword match',
          blocks: blockForMicronaut(mn)
        };
      }
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

async function dispatchMcpTool(toolName, toolArgs) {
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
        curiosity: 0.7, concern: 0.3, valence: 0.2, confidence: 0.8, attachment: 0.6
      });
      return new Promise(resolve => {
        execFile(ENGINE_EXE, ['--forge', forgeText], { timeout: 15000, maxBuffer: 2 * 1024 * 1024 },
          (err, stdout, stderr) => {
            const out = (stdout || '') + (stderr ? '\n[stderr] ' + stderr : '');
            writeTrace('engine.forge', { name: forgeName, text: forgeText.slice(0, 80) });
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
    writeTrace('micronaut.select', { prompt: prompt.slice(0, 120), selected });
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      prompt,
      selected: { ...selected, full },
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

    // SSE streaming response
    if (accept.includes('text/event-stream')) {
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' });
      const reply = await handleMcpJsonRpc(msg);
      if (reply) res.write('data: ' + JSON.stringify(reply) + '\n\n');
      res.end();
      return;
    }

    const reply = await handleMcpJsonRpc(msg);
    if (reply === null) { res.writeHead(202); res.end(); return; }
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

    const msgs = body.messages || [];
    const lastUserMsg = [...msgs].reverse().find(m => m.role === 'user')?.content || '';
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

    // Pass-through to active inference model (khanary-server, default :9000)
    // active-model.json.port = the chat model port; engine_endpoint = planner (17480)
    let inferPort = 9000;
    try {
      const amPath = path.join(PROJECT_ROOT, 'active-model.json');
      if (fs.existsSync(amPath)) {
        const am = JSON.parse(fs.readFileSync(amPath, 'utf8'));
        if (am.port && am.port !== REGISTRY_PORT) inferPort = am.port;
      }
    } catch (_) {}

    const inferOpts = {
      hostname: '127.0.0.1', port: inferPort,
      path: '/v1/chat/completions', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(rawBody) }
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
    proxyReq.write(rawBody);
    proxyReq.end();
    return;
  }

  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'not found', path: p }));
});

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
  await ensureEngineRunning();
});

// =============================================================================
// Tick Loop
// =============================================================================
setInterval(async () => {
  await checkEngineHealth();
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
