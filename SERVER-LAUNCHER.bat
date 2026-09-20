@echo off  
setlocal  
  
:: ============================================================================  
:: SERVER-LAUNCHER.bat — DEPRECATED 2026-08-06 — use START-SERVERS.bat instead  
:: START-SERVERS.bat: same launch logic + active-model.json + MCP hookup.  
:: Build pipeline: llama-build deploy | START-SERVERS  
:: ============================================================================  
  
echo [SERVER-LAUNCHER] DEPRECATED — use START-SERVERS.bat instead  
echo [SERVER-LAUNCHER] Build: llama-build deploy | Launch: START-SERVERS  
echo [SERVER-LAUNCHER] Forwarding to START-SERVERS.bat...  
call "%%~dp0START-SERVERS.bat" %%*  
