#!/usr/bin/env node
// package_wwa.cjs — package a folder into a .wwa zip container via
// kuhul_engine_driver.dll ke_package_wwa (native, deterministic STORE).
//
// Usage:
//   node tools/package_wwa.cjs <app_root> <out.wwa>
//
// Falls back to PowerShell Compress-Archive (.zip → rename) if the DLL
// is not available. Exit code 0 = success, 1 = failure.
'use strict';

const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const [,, appRoot, outWwa] = process.argv;
if (!appRoot || !outWwa) {
  console.error('usage: node tools/package_wwa.cjs <app_root> <out.wwa>');
  process.exit(1);
}

const dllPath = path.join(__dirname, '..', 'drivers', 'kuhul_engine_driver.dll');

function nativePackage() {
  // koffi-backed ffi shim lives in dist/khanary-server (koffi is installed there)
  const shimPath = path.join(__dirname, '..', 'dist', 'khanary-server', 'ffi-shim.js');
  if (!fs.existsSync(shimPath)) throw new Error('ffi-shim not found at ' + shimPath);
  const ffi = require(shimPath);
  const ref = ffi.ref;
  const dll = ffi.Library(dllPath, {
    ke_package_wwa: ['int', ['string', 'string', 'pointer', 'int']],
  });
  const errBuf = Buffer.alloc(512);
  const ok = dll.ke_package_wwa(appRoot, outWwa, errBuf, errBuf.length);
  if (!ok) {
    const msg = ref.readCString(errBuf, 0);
    throw new Error('ke_package_wwa failed: ' + msg);
  }
  return 'native';
}

function psFallback() {
  const tmpZip = outWwa.replace(/\.wwa$/i, '') + '.tmp.zip';
  if (fs.existsSync(tmpZip)) fs.unlinkSync(tmpZip);
  if (fs.existsSync(outWwa)) fs.unlinkSync(outWwa);
  execFileSync('powershell', ['-NoProfile', '-Command',
    `Get-ChildItem -Path "${appRoot}" -File | Where-Object { $_.Extension -ne '.wwa' } | Compress-Archive -DestinationPath "${tmpZip}" -Force`],
    { timeout: 30000, stdio: 'pipe' });
  if (!fs.existsSync(tmpZip)) throw new Error('powershell compress produced no output');
  fs.renameSync(tmpZip, outWwa);
  return 'powershell';
}

async function main() {
  try {
    let backend;
    if (fs.existsSync(dllPath)) {
      try { backend = nativePackage(); }
      catch (e) {
        console.warn('[package] native failed (' + e.message + '), falling back to PowerShell');
        backend = psFallback();
      }
    } else {
      backend = psFallback();
    }
    const size = fs.statSync(outWwa).size;
    console.log(`[package] ${outWwa} (${size} bytes, backend: ${backend})`);
    process.exit(0);
  } catch (e) {
    console.error('[package] ERROR: ' + e.message);
    process.exit(1);
  }
}

main();
