$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$launcher = $null
$launcherArgs = @()
function Test-Python311([string]$command, [string[]]$arguments = @()) {
    try {
        $version = & $command @arguments -c "import platform; print(platform.python_version())" 2>$null
        return ($LASTEXITCODE -eq 0 -and "$version" -match '^3\.11\.')
    } catch { return $false }
}

if ($env:COLIBRI_PYTHON) {
    if (-not (Test-Python311 $env:COLIBRI_PYTHON)) { throw "COLIBRI_PYTHON must point to a working Python 3.11.x interpreter." }
    $launcher = $env:COLIBRI_PYTHON
} elseif ((Test-Path -LiteralPath 'D:\python3.11.3\python.exe') -and (Test-Python311 'D:\python3.11.3\python.exe')) {
    $launcher = 'D:\python3.11.3\python.exe'
} else {
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py -and (Test-Python311 $py.Source @('-3.11'))) {
        $launcher = $py.Source
        $launcherArgs = @('-3.11')
    } else {
        $python311 = Get-Command python3.11.exe -ErrorAction SilentlyContinue
        if ($python311 -and (Test-Python311 $python311.Source)) { $launcher = $python311.Source }
    }
}

if (-not $launcher) { throw "Python 3.11.x was not found. Refusing to fall back to the system-default Python." }
$resolvedVersion = & $launcher @launcherArgs -c "import platform; print(platform.python_version())"
Write-Host "Starting with Python $resolvedVersion via $launcher $launcherArgs"
& $launcher @launcherArgs -m uvicorn app.main:app --host 127.0.0.1 --port 18081
