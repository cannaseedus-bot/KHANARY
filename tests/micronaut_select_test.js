#!/usr/bin/env node
'use strict';
// Test /micronauts/select routing.
// Reads the kuhul-server port from dist/khanary-server/.kuhul-server.port.

const fs = require('fs');
const path = require('path');
const http = require('http');

const portFile = path.join(__dirname, '..', 'dist', 'khanary-server', '.kuhul-server.port');
const PORT = fs.existsSync(portFile) ? fs.readFileSync(portFile, 'utf8').trim() : '8764';
const HOST = '127.0.0.1';

const cases = [
  { prompt: 'how does kuhul route tasks', expect: 'stack_doc', blocks: ['MENU', 'BODY'] },
  { prompt: 'write a python function to sort', expect: 'coder', blocks: ['BODY'] },
  { prompt: 'save this memory', expect: 'memory', blocks: ['MENU', 'BODY'] },
  { prompt: 'what is the SCX bytecode format', expect: 'scx_guide', blocks: ['MENU', 'BODY'] },
  { prompt: 'how do I train a LoRA', expect: 'distillation_guide', blocks: ['MENU', 'BODY'] },
  { prompt: 'explain PRIMEOS', expect: 'primeos_guide', blocks: ['MENU', 'BODY'] },
  { prompt: 'hi chat', expect: 'chat', blocks: ['BODY'] },
  { prompt: 'pop phase observe', expect: 'pop', blocks: ['HEADER'] },
  { prompt: 'xul emit output', expect: 'xul', blocks: ['FOOTER'] },
];

function request(prompt) {
  return new Promise((resolve, reject) => {
    const opts = {
      hostname: HOST,
      port: PORT,
      path: '/micronauts/select?prompt=' + encodeURIComponent(prompt),
      method: 'GET',
    };
    const req = http.request(opts, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch (e) { reject(new Error('bad json: ' + data)); }
      });
    });
    req.on('error', reject);
    req.end();
  });
}

(async () => {
  let passed = 0;
  let failed = 0;
  for (const tc of cases) {
    const result = await request(tc.prompt);
    const got = result.selected?.name;
    const blocks = result.atomic_blocks || [];
    const ok = got === tc.expect && tc.blocks.every(b => blocks.includes(b));
    if (ok) {
      passed++;
      console.log(`PASS  "${tc.prompt}" -> ${got}  blocks=[${blocks.join(',')}]`);
    } else {
      failed++;
      console.log(`FAIL  "${tc.prompt}" -> ${got} (expected ${tc.expect})  blocks=[${blocks.join(',')}] (expected [${tc.blocks.join(',')}])`);
    }
  }
  console.log(`\n${passed}/${cases.length} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
})();
