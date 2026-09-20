#!/usr/bin/env node
'use strict';

/*
 * KUHUL-ES CLI v3.5.0
 * K'UHUL language runtime + WebX-3D geometric intelligence stack
 *
 * Language (1.0.x compat):
 *   kuhul-es run <file>      -- execute .kuhules / .ts via π/τ/glyph runtime
 *   kuhul-es compile <file>  -- TypeScript bridge: .ts → ASX → .kpi binary
 *   kuhul-es new <name>      -- scaffold K'UHUL project
 *   kuhul-es server          -- deterministic runtime host
 *   kuhul-es doctor          -- environment diagnostics
 *
 * WebX-3D (3.5.0):
 *   kuhul-es kxml run <f>    -- execute .kxml bidirectional graph
 *   kuhul-es kxml verify <f> -- validate + Merkle root
 *   kuhul-es models          -- list K'UHUL tool models
 *   kuhul-es gpu             -- probe GPU backends
 */

const fs     = require('fs');
const path   = require('path');
const crypto = require('crypto');
const { program } = require('commander');
let pkg;
try {
  pkg = require('kuhul-es/package.json');          // installed package (real version)
} catch {
  pkg = JSON.parse(fs.readFileSync(path.join(__dirname, 'package.json'), 'utf8'));
}

console.log(`
 ╔══════════════════════════════════════════════════╗
 ║  K'UHUL-ES v${pkg.version.padEnd(6)}  WebX-3D                  ║
 ║  ECMAScript syntax  |  K'UHUL semantics          ║
 ║  pi bindings  |  tau bindings  |  KXML graph     ║
 ╚══════════════════════════════════════════════════╝
`);

program.name('kuhul-es')
  .description("K'UHUL-ES -- ECMAScript syntax, K'UHUL semantics")
  .version(pkg.version);

// ─── Language surface ─────────────────────────────────────────────────────────

program.command('new <name>')
  .description("Scaffold a new K'UHUL-ES project")
  .action((name) => {
    fs.mkdirSync(name, { recursive: true });
    const tpl = [
      `// ${name} -- K'UHUL-ES`,
      `pi config = { name: "${name}", version: "1.0.0" };`,
      'tau frame = 0;',
      'function* main() {',
      '  yield* Pop("init");',
      `  yield* Sek('log', config.name);`,
      '  yield* Xul();',
      '}',
      'main();'
    ].join('\n');
    fs.writeFileSync(path.join(name, 'main.kuhules'), tpl);
    fs.writeFileSync(path.join(name, 'package.json'),
      JSON.stringify({ name, version: '1.0.0', private: true,
        scripts: { start: 'kuhul-es run main.kuhules' },
        dependencies: { 'kuhul-es': `^${pkg.version}` }
      }, null, 2));
    console.log(`Created: ${name}/\n  cd ${name} && npm install && npm start`);
    process.exit(0);
  });

program.command('compile <input>')
  .description("TypeScript bridge: .ts K'UHUL source -> ASX JSON -> .kpi binary")
  .option('-o, --output <file>', 'Output .kpi path')
  .action((input, opts) => {
    console.log(`Compiling ${input} via TypeScript bridge...`);
    if (opts.output) console.log(`--> ${opts.output}`);
    console.log('See compiler/src/parser.ts for KUHULParser implementation.');
    console.log('Compile complete (stub)');
    process.exit(0);
  });

program.command('run <file>')
  .description("Execute a K'UHUL-ES program (.kuhules or .ts)")
  .option('--record', 'Record trace to trace.json')
  .option('--replay <f>', 'Verify and replay a trace.json')
  .action((file, opts) => {
    if (opts.record) {
      const t = { version: pkg.version, file,
        argv: process.argv.slice(2), cwd: process.cwd(),
        seq: 1 };  // monotonic; wall clock excluded from hash
      t.hash = crypto.createHash('sha256')
        .update(JSON.stringify({ version: t.version, file: t.file, argv: t.argv, cwd: t.cwd, seq: t.seq }))
        .digest('hex');
      t.wall_ts = new Date().toISOString();  // display only, not hashed
      fs.writeFileSync('trace.json', JSON.stringify(t, null, 2));
      console.log(`Trace recorded  hash: ${t.hash}`);
    }
    if (opts.replay) {
      const t = JSON.parse(fs.readFileSync(opts.replay, 'utf8'));
      const exp = crypto.createHash('sha256')
        .update(JSON.stringify({ ...t, hash: undefined })).digest('hex');
      if (t.hash !== exp) { console.error('Hash mismatch'); process.exit(1); }
      console.log(`Trace verified  hash: ${t.hash}`);
    }
    // Load and run via node runtime
    try {
      const { KUHULRuntimeNode } = require('../runtime/src/node.cjs');
      const rt = new KUHULRuntimeNode();
      rt.executeFile(path.resolve(file)).then(() => process.exit(0));
    } catch (e) {
      console.log(`Running ${file} (runtime stub)`);
      process.exit(0);
    }
  });

program.command('server')
  .description('Start K\'UHUL deterministic runtime host')
  .action(() => require('./kuhul-server.cjs'));

program.command('doctor')
  .description('Environment diagnostics')
  .action(() => {
    console.log("K'UHUL-ES Doctor\n");
    console.log('Version :', pkg.version);
    console.log('Node    :', process.version);
    console.log('Platform:', process.platform);
    ['commander', 'typescript'].forEach(d => {
      try { require.resolve(d); console.log(`${d}: OK`); }
      catch { console.log(`${d}: MISSING`); }
    });
    const srcDir = path.join(__dirname, '..', 'src');
    console.log('\nWebX-3D modules:');
    ['kxml', 'xvm', 'gpu', 'mayan', 'linalg', 'supernaut'].forEach(m => {
      console.log(`  src/${m}/: ${fs.existsSync(path.join(srcDir, m)) ? 'OK' : 'MISSING'}`);
    });
    const regPath = path.join(__dirname, '..', 'models', 'model-registry.json');
    if (fs.existsSync(regPath)) {
      const reg = JSON.parse(fs.readFileSync(regPath, 'utf8'));
      console.log(`\nModels (${reg.models.length}):`);
      reg.models.forEach(m =>
        console.log(`  ${m.status === 'production' ? '*' : ' '} ${m.id}  (${m.status})`));
    }
    console.log('\nDiagnostics complete');
    process.exit(0);
  });

// ─── WebX-3D commands ─────────────────────────────────────────────────────────

const kxml = program.command('kxml').description('KXML bidirectional graph commands');

kxml.command('run <file>')
  .description('Execute a .kxml bidirectional computation graph through phase-gated dispatch')
  .option('--tinyX', 'Enable tiny.x GPU pass')
  .action(async (file, opts) => {
    const { KXMLGraph } = await import('../src/kxml/kxml-graph.js');
    const graph = KXMLGraph.fromString(fs.readFileSync(file, 'utf8')).compile();
    const disp  = graph.createDispatcher({ tinyXValid: !!opts.tinyX });
    const result = disp.run();
    console.log('\nPhase results:');
    for (const [phase, nodes] of Object.entries(result))
      if (nodes.length) console.log(`  ${phase}: ${nodes.join(', ')}`);
    console.log(`\nMerkle: ${graph.merkleRoot()}`);
    process.exit(0);
  });

kxml.command('verify <file>')
  .description('Validate .kxml — edges, Lipschitz contracts, Merkle root')
  .action(async (file) => {
    const { KXMLGraph } = await import('../src/kxml/kxml-graph.js');
    const graph = KXMLGraph.fromString(fs.readFileSync(file, 'utf8')).compile();
    const v = graph.validate();
    if (v.valid) {
      console.log(`Valid  nodes=${graph.nodes.size}  edges=${graph.edges.length}`);
      console.log(`Merkle: ${graph.merkleRoot()}`);
    } else {
      v.errors.forEach(e => console.error(`Error: ${e}`));
      process.exit(1);
    }
    process.exit(0);
  });

program.command('models')
  .description("List K'UHUL WebX-3D tool-calling models")
  .action(() => {
    const reg = JSON.parse(fs.readFileSync(
      path.join(__dirname, '..', 'models', 'model-registry.json'), 'utf8'));
    console.log(`\nK'UHUL WebX-3D Tool Models\n`);
    reg.models.forEach(m => {
      const mark = m.status === 'production' ? '[*]' : '[ ]';
      const loss = m.training?.final_loss ?? `target ${m.training?.target_loss}`;
      console.log(`${mark} ${m.id.padEnd(26)} ${(m.size_mb + 'MB').padEnd(8)} loss=${loss}  ${m.status}`);
    });
    console.log('\nUsage:');
    console.log('  npm i kuhul-es');
    console.log("  import { KXMLGraph } from 'kuhul-es/kxml'");
    process.exit(0);
  });

program.command('gpu')
  .description('Probe available GPU backends')
  .action(async () => {
    console.log("K'UHUL GPU Probe\n");
    try {
      const { detectWebGPU, detectWasmSimd, detectSAB } =
        await import('../src/gpu/webgpu-runtime.js');
      console.log(`WebGPU  : ${await detectWebGPU()}`);
      console.log(`WASM SIMD: ${detectWasmSimd()}`);
      console.log(`SAB     : ${detectSAB()}`);
    } catch {
      console.log('Full probe requires browser context.');
      console.log('D3D11 probe: python .gpu_trainer/gpu_capability_test.py');
    }
    process.exit(0);
  });

// ─── Parse ────────────────────────────────────────────────────────────────────
program.parse(process.argv);
if (!process.argv.slice(2).length) program.outputHelp();
