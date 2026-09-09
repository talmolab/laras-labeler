<#
.SYNOPSIS
  One command to get hidra-in-the-loop running on Windows.

.DESCRIPTION
  Sets up everything the HiDRA loop needs and then starts the app:

    1. stops any labeler already running (it locks its own files, so an install fails otherwise)
    2. clones or updates the HiDRA checkout, on the branch that fixes the hardcoded /dev/shm
       (that path does not exist on Windows, so inference cannot run without it)
    3. builds an interpreter with JAX -- CPU wheels, no CUDA and no WSL required
    4. downloads the model weights (~664 MB) if they are not already there
    5. installs the labeler from the branch carrying the loop fixes
    6. starts it with HIDRA_HOME / HIDRA_PYTHON already pointed at the checkout

  Safe to re-run: every step is skipped if it is already done. Nothing is deleted.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup_hidra_windows.ps1

.EXAMPLE
  # keep everything under D:\hidra instead of the default, and do not launch at the end
  .\setup_hidra_windows.ps1 -Root D:\hidra -NoStart
#>
[CmdletBinding()]
param(
    # Where the HiDRA checkout and its interpreter live.
    [string]$Root = "$HOME\code",
    # The projects folder the labeler opens (created on first run).
    [string]$Projects = "$HOME\laras-projects",
    # Branch of laras-labeler to install.
    [string]$LabelerRef = "claude/annotation-event-log-48xvu9",
    # Branch of HiDRA to check out.
    [string]$HidraRef = "claude/scratch-dir-portability",
    # Set up only; do not start the app.
    [switch]$NoStart
)

$ErrorActionPreference = "Stop"
$step = 0
function Step($msg) { $script:step++; Write-Host "`n[$script:step] $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Note($msg) { Write-Host "    $msg" -ForegroundColor DarkGray }
function Die($msg)  { Write-Host "`nSTOPPED: $msg" -ForegroundColor Red; exit 1 }

$hidraHome = Join-Path $Root "HiDRA"
$venv      = Join-Path $hidraHome ".venv"
$py        = Join-Path $venv "Scripts\python.exe"

Write-Host "hidra-in-the-loop setup" -ForegroundColor White
Note "checkout    $hidraHome"
Note "interpreter $py"
Note "projects    $Projects"

# --- prerequisites -----------------------------------------------------------------------------
Step "checking prerequisites"
foreach ($tool in @("git", "uv")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        if ($tool -eq "uv") {
            Die "uv is not installed. Install it, reopen PowerShell, and re-run:`n    irm https://astral.sh/uv/install.ps1 | iex"
        }
        Die "$tool is not on PATH. Install it and re-run."
    }
    Ok "$tool found"
}

# --- 1. stop a running labeler -----------------------------------------------------------------
# Windows will not let an install overwrite a running .exe, and the failure is an opaque
# "Access is denied" on the tool's Scripts directory.
Step "stopping any running labeler"
$running = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessName -match '^(laras-labeler|hidra-in-the-loop)$' }
if ($running) {
    $running | Stop-Process -Force
    Start-Sleep -Seconds 2
    Ok "stopped $($running.Count) process(es)"
} else {
    Ok "none running"
}

# --- 2. the HiDRA checkout ----------------------------------------------------------------------
Step "HiDRA checkout"
if (-not (Test-Path $Root)) { New-Item -ItemType Directory -Path $Root -Force | Out-Null }
if (Test-Path (Join-Path $hidraHome ".git")) {
    Ok "already cloned"
    git -C $hidraHome fetch origin $HidraRef --quiet
    git -C $hidraHome checkout $HidraRef --quiet
    git -C $hidraHome pull --ff-only origin $HidraRef --quiet
    Ok "updated to $HidraRef"
} else {
    Note "cloning (private repo -- a GitHub login may be requested)"
    git clone --quiet https://github.com/talmolab/HiDRA.git $hidraHome
    if (-not (Test-Path (Join-Path $hidraHome "predict.py"))) { Die "clone did not produce predict.py in $hidraHome" }
    git -C $hidraHome fetch origin $HidraRef --quiet
    git -C $hidraHome checkout $HidraRef --quiet
    Ok "cloned and on $HidraRef"
}

# --- 3. an interpreter with JAX -----------------------------------------------------------------
Step "interpreter with JAX"
$haveJax = $false
if (Test-Path $py) {
    & $py -c "import jax" 2>$null
    if ($LASTEXITCODE -eq 0) { $haveJax = $true }
}
if ($haveJax) {
    Ok "already present"
} else {
    if (-not (Test-Path $py)) {
        uv venv --python 3.12 $venv
        if ($LASTEXITCODE -ne 0) { Die "could not create a venv at $venv" }
    }
    Note "installing jax 0.9.2 (CPU) -- a few minutes"
    uv pip install --python $py --quiet "jax==0.9.2" numpy pandas pyarrow huggingface_hub
    if ($LASTEXITCODE -ne 0) { Die "installing JAX failed" }
    Ok "installed"
}
$backend = (& $py -c "import jax; print(jax.default_backend())" 2>$null)
if ($LASTEXITCODE -ne 0) { Die "JAX is installed but will not import in $py" }
Ok "JAX backend: $backend"

# --- 4. model weights ---------------------------------------------------------------------------
Step "model weights"
$models = Join-Path $hidraHome "models"
$have = 0
if (Test-Path $models) { $have = (Get-ChildItem $models -Filter *.pkl -ErrorAction SilentlyContinue).Count }
if ($have -ge 11) {
    Ok "$have weight files already present"
} else {
    Note "downloading ~664 MB from Hugging Face -- this is the slow step"
    Push-Location $hidraHome
    try { & $py download_models.py } finally { Pop-Location }
    $have = (Get-ChildItem $models -Filter *.pkl -ErrorAction SilentlyContinue).Count
    if ($have -lt 11) { Die "only $have of 11 weight files arrived; re-run to resume" }
    Ok "$have weight files"
}

# --- 5. the labeler -----------------------------------------------------------------------------
Step "installing the labeler"
uv tool install --force "git+https://github.com/talmolab/laras-labeler@$LabelerRef"
if ($LASTEXITCODE -ne 0) {
    Die "install failed. If it says Access is denied, a labeler is still running -- close that window and re-run."
}
Ok "installed from $LabelerRef"
# uv puts console scripts here; it is not on PATH by default and the warning is easy to miss.
$uvBin = Join-Path $HOME ".local\bin"
if (Test-Path $uvBin) { $env:PATH = "$uvBin;$env:PATH"; Note "added $uvBin to PATH for this session" }

# --- 6. run -------------------------------------------------------------------------------------
Step "pointing the labeler at the checkout"
$env:HIDRA_HOME = $hidraHome
$env:HIDRA_PYTHON = $py
Ok "HIDRA_HOME   = $env:HIDRA_HOME"
Ok "HIDRA_PYTHON = $env:HIDRA_PYTHON"
if (-not (Test-Path $Projects)) { New-Item -ItemType Directory -Path $Projects -Force | Out-Null }

if ($NoStart) {
    Write-Host "`nSetup complete. Start it with:" -ForegroundColor White
    Write-Host "    `$env:HIDRA_HOME='$hidraHome'; `$env:HIDRA_PYTHON='$py'; hidra-in-the-loop '$Projects'"
    exit 0
}

Write-Host "`nStarting. It opens your browser; the banner below should say 82 classifiers." -ForegroundColor White
Write-Host "Leave this window open -- it IS the server. Ctrl+C stops it.`n" -ForegroundColor DarkGray
hidra-in-the-loop $Projects
