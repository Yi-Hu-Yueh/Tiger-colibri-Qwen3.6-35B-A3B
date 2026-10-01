$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = if ($env:COLIBRI_PYTHON) { $env:COLIBRI_PYTHON } elseif (Test-Path -LiteralPath 'D:\python3.11.3\python.exe') { 'D:\python3.11.3\python.exe' } else { throw 'Configure COLIBRI_PYTHON with a Python 3.11.x interpreter.' }
$version = & $python -c "import platform; print(platform.python_version())"
if ("$version" -notmatch '^3\.11\.') { throw "Python 3.11.x required; found $version" }
& $python -c "import fastapi, uvicorn, jinja2; print('Python runtime validated'); print('FastAPI', fastapi.__version__); print('Uvicorn', uvicorn.__version__); print('Jinja2', jinja2.__version__)"
