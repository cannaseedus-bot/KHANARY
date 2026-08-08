// ffi-shim.js — koffi-backed compatibility layer for ffi-napi style DLL loading
//
// koffi is actively maintained and works on Node 18/20/22/24, whereas ffi-napi
// is deprecated and its native bindings fail to compile on newer Node versions.
// This shim exposes a minimal ffi-napi-like surface so kuhul-server.cjs can
// load the Khanary driver DLLs without changes.
//

const koffi = require('koffi');

const typeMap = {
  pointer: 'void *',
  string: 'const char *',
  int: 'int',
  uint32: 'uint32_t',
  void: 'void',
};

function ffiToKoffi(type) {
  return typeMap[type] || type;
}

function readCString(buffer, offset = 0) {
  if (!Buffer.isBuffer(buffer)) return '';
  const end = buffer.indexOf(0, offset);
  const len = end === -1 ? buffer.length - offset : end - offset;
  if (len <= 0) return '';
  return buffer.toString('utf8', offset, offset + len);
}

function Library(path, functions) {
  const lib = koffi.load(path);
  const wrapper = {};

  // Functions that free malloc'd C strings. With koffi, string-returning
  // functions decode to JS strings, so we no-op if passed a JS string.
  const isFreeString = (name) => /free_string$/.test(name);

  for (const [name, [retType, argTypes]] of Object.entries(functions)) {
    const koffiRet = ffiToKoffi(retType);
    const koffiArgs = argTypes.map(ffiToKoffi);
    const fn = lib.func(name, koffiRet, koffiArgs);

    wrapper[name] = (...args) => {
      // For free-string functions, skip native call when argument is a JS
      // string (koffi already decoded it). Passing a JS string to a char*
      // arg would otherwise crash because koffi would create a temporary
      // buffer that the C side tries to delete[].
      if (isFreeString(name)) {
        if (args.length === 0) return fn();
        if (typeof args[0] === 'string') {
          // Native pointer is leaked here; callers should declare
          // string-returning functions as 'pointer' if precise lifetime is
          // needed. No-op is safe for short-lived daemon workloads.
          return undefined;
        }
      }
      return fn(...args);
    };
  }

  return wrapper;
}

module.exports = { Library, ref: { readCString } };
