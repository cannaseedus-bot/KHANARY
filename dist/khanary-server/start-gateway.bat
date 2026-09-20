@echo off
:: KUHUL-SERVER GATEWAY WRAPPER
:: Sets KUHUL_REGISTRY_PORT=8764 and starts kuhul-server.cjs from the project root.
:: Called by START-SERVERS.bat via: start "kuhul-server" <this file>
::
:: Running node via this wrapper (rather than cmd /c "set VAR && node") avoids
:: Windows shell quoting issues and guarantees the env var is set before node starts.
setlocal
set "KUHUL_REGISTRY_PORT=8764"
cd /d "C:\Users\canna\_khanary_inspect"
node dist\khanary-server\kuhul-server.cjs
endlocal
