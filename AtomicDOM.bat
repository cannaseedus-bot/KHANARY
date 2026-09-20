@echo off
setlocal

:: AtomicDOM.bat — launch kuhul_engine (or moe_gguf_runtime) with a per-model Atomic Block DOM manifest
::
:: Usage:
::   AtomicDOM                          -- from_zero default (KUHUL chat)
::   AtomicDOM from_zero                -- from_zero v0.6 KUHUL chat + KXML tools
::   AtomicDOM kxml                     -- KXML tool-call agent (trained-in glyphs)
::   AtomicDOM gpt-oss                  -- GPT-OSS 20B teacher (CPU, 11.28 GB)
::   AtomicDOM gpt2-xl                  -- GPT-2 XL Q8 MCP-baked (GPU, 1668 MB)
::   AtomicDOM lfm2                     -- LFM2.5 1.2B 128K context (GPU, 1188 MB)
::   AtomicDOM gemma-1b                 -- Gemma 3 1B QAT (GPU, 687 MB) fastest
::   AtomicDOM gemma-1b-q8              -- Gemma 3 1B Q8 unsloth (GPU, 1020 MB) quality
::   AtomicDOM gemma-4b                 -- Gemma 3 4B (CPU, ~2.7 GB, still downloading)
::   AtomicDOM gemma-4-e2b              -- Gemma 4 E2B multimodal vision (CPU, 4.2 GB)
::   AtomicDOM phi3-mini                -- Phi-3 Mini 4K micronaut tool agent (CPU, 2282 MB)
::   AtomicDOM dolphin                  -- Dolphin 2.6 Phi-2 uncensored (CPU, 1844 MB)
::   AtomicDOM qwen-1b8                 -- Qwen 1.8B Chat F16 safetensors (convert req'd)
::   AtomicDOM qwen-story               -- Qwen 2.5 0.5B story mode (GPU, 644 MB)
::   AtomicDOM mgguf-gpt2               -- GPT-2 2-expert MoE ASX mgguf (GPU, 1408 MB)
::   AtomicDOM mgguf-qwen               -- Qwen 1-expert MoE ASX mgguf (CPU, 1862 MB)
::   AtomicDOM <path\to\manifest.json>  -- explicit manifest

set "ROOT=%~dp0"
set "ENGINE=C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\build\bin\Release\kuhul_engine.exe"
if not exist "%ENGINE%" set "ENGINE=C:\Users\canna\_khanary_inspect\khanary-llama-build\llama.cpp\build\bin\Release\llama-server.exe"
set "MGGUF_RUNTIME=%ROOT%bin\moe_gguf_runtime.exe"

:: Resolve manifest from first arg (or default to from_zero)
set "ARG=%~1"
set "BACKEND=kuhul_engine"
if "%ARG%"==""             set "MANIFEST=%ROOT%models\from_zero\atomic.manifest.json"
if "%ARG%"=="from_zero"    set "MANIFEST=%ROOT%models\from_zero\atomic.manifest.json"
if "%ARG%"=="kxml"         set "MANIFEST=%ROOT%models\khanary-kxml-v0.5.0\atomic.manifest.json"
if "%ARG%"=="gpt-oss"      set "MANIFEST=%ROOT%models\gpt-oss\atomic.manifest.json"
if "%ARG%"=="gpt2-xl"      set "MANIFEST=%ROOT%models\gpt2-xl\atomic.manifest.json"
if "%ARG%"=="lfm2"         set "MANIFEST=%ROOT%models\lfm2-1b\atomic.manifest.json"
if "%ARG%"=="gemma-1b"     set "MANIFEST=%ROOT%models\gemma-3-1b\atomic.manifest.json"
if "%ARG%"=="gemma-1b-q8"  set "MANIFEST=%ROOT%models\gemma-3-1b-q8\atomic.manifest.json"
if "%ARG%"=="gemma-4b"     set "MANIFEST=%ROOT%models\gemma-3-4b\atomic.manifest.json"
if "%ARG%"=="gemma-4-e2b"  set "MANIFEST=%ROOT%models\gemma-4-e2b\atomic.manifest.json"
if "%ARG%"=="phi3-mini"    set "MANIFEST=%ROOT%models\phi3-mini-4k\atomic.manifest.json"
if "%ARG%"=="dolphin"      set "MANIFEST=%ROOT%models\dolphin-phi2\atomic.manifest.json"
if "%ARG%"=="qwen-1b8"     set "MANIFEST=%ROOT%models\qwen-1b8-chat\atomic.manifest.json"
if "%ARG%"=="qwen-story"   set "MANIFEST=%ROOT%models\qwen25-05b-story\atomic.manifest.json"
if "%ARG%"=="mgguf-gpt2"   set "MANIFEST=%ROOT%models\mgguf-gpt2-2expert\atomic.manifest.json" & set "BACKEND=mgguf_runtime"
if "%ARG%"=="mgguf-qwen"   set "MANIFEST=%ROOT%models\mgguf-qwen-1expert\atomic.manifest.json" & set "BACKEND=mgguf_runtime"

:: If arg looks like a path, use it directly
if exist "%ARG%" set "MANIFEST=%ARG%"

if not exist "%MANIFEST%" (
  echo AtomicDOM: manifest not found: %MANIFEST%
  exit /b 1
)

echo [AtomicDOM] Backend : %BACKEND%
echo [AtomicDOM] Manifest: %MANIFEST%
echo.

if "%BACKEND%"=="mgguf_runtime" (
  if exist "%MGGUF_RUNTIME%" (
    "%MGGUF_RUNTIME%" --Atomic.DOM "%MANIFEST%" --chat
  ) else (
    echo [AtomicDOM] moe_gguf_runtime.exe not found: %MGGUF_RUNTIME%
    exit /b 1
  )
) else (
  if exist "%ENGINE%" (
    "%ENGINE%" --Atomic.DOM "%MANIFEST%" --chat
  ) else (
    echo [AtomicDOM] kuhul_engine.exe not found at %ENGINE%
    echo             Start kuhul-server.cjs instead: node dist/khanary-server/kuhul-server.cjs
    exit /b 1
  )
)
endlocal
