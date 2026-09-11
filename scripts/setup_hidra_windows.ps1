<#
.SYNOPSIS
  One command to get hidra-in-the-loop running on Windows, on the PyTorch HiDRA.

.DESCRIPTION
  HiDRA was rewritten from JAX to PyTorch: it now uses the Windows GPU (JAX could not),
  runs several times faster, and is an installable package. This sets everything up and
  starts the app:

    1. stops any labeler already running (it locks its own files, so an install fails otherwise)
    2. clones or updates the HiDRA checkout to the current main (the PyTorch code)
    3. builds a torch interpreter and installs HiDRA into it (GPU wheels auto-detected)
    4. makes sure the model weights are present (your existing .pkl files are reused as-is)
    5. installs the labeler from the branch carrying the loop + threshold-slider work
    6. starts it with HIDRA_HOME / HIDRA_PYTHON pointed at the torch checkout

  Safe to re-run: every step is skipped if it is already done. Nothing is deleted.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup_hidra_windows.ps1

.EXAMPLE
  # CPU-only machine (no NVIDIA GPU): use the CPU torch wheels
  .\setup_hidra_windows.ps1 -TorchBackend cpu
#>
[CmdletBinding()]
param(
    # Where the HiDRA checkout and its interpreter live.
    [string]$Root = "$HOME\code",
    # The projects folder the labeler opens (created on first run).
    [string]$Projects = "$HOME\laras-projects",
    # Branch of laras-labeler to install.
    [string]$LabelerRef = "claude/annotation-event-log-48xvu9",
    # Branch/ref of HiDRA to check out. main is the PyTorch rewrite.
    [string]$HidraRef = "main",
    # Torch wheels: 'auto' lets uv detect your GPU (CUDA if present, else CPU); 'cpu' forces CPU.
    [ValidateSet("auto", "cpu", "cu124", "cu121", "cu118")]
    [string]$TorchBackend = "auto",
    # Set up only; do not start the app.
    [switch]$NoStart
)

$ErrorActionPreference = "Stop"
$step = 0
function Step($m) { $script:step++; Write-Host "`n[$script:step] $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "    $m" -ForegroundColor Green }
function Note($m) { Write-Host "    $m" -ForegroundColor DarkGray }
function Die($m)  { Write-Host "`nSTOPPED: $m" -ForegroundColor Red; exit 1 }

$hidraHome = Join-Path $Root "HiDRA"
$venv      = Join-Path $hidraHome ".venv-torch"     # separate from any old JAX .venv
$py        = Join-Path $venv "Scripts\python.exe"

Write-Host "hidra-in-the-loop setup (PyTorch HiDRA)" -ForegroundColor White
Note "checkout    $hidraHome"
Note "interpreter $py"
Note "projects    $Projects"
Note "torch       $TorchBackend"

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
Step "stopping any running labeler"
$running = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessName -match '^(laras-labeler|hidra-in-the-loop)$' }
if ($running) { $running | Stop-Process -Force; Start-Sleep -Seconds 2; Ok "stopped $($running.Count)" }
else { Ok "none running" }

# --- 2. the HiDRA checkout (PyTorch main) ------------------------------------------------------
Step "HiDRA checkout on '$HidraRef'"
if (-not (Test-Path $Root)) { New-Item -ItemType Directory -Path $Root -Force | Out-Null }
if (Test-Path (Join-Path $hidraHome ".git")) {
    git -C $hidraHome fetch origin $HidraRef --quiet
    git -C $hidraHome checkout $HidraRef --quiet
    git -C $hidraHome pull --ff-only origin $HidraRef --quiet
    Ok "updated to $HidraRef"
} else {
    Note "cloning (private repo -- a GitHub login may be requested)"
    git clone --quiet https://github.com/talmolab/HiDRA.git $hidraHome
    git -C $hidraHome checkout $HidraRef --quiet
    Ok "cloned on $HidraRef"
}
if (-not (Test-Path (Join-Path $hidraHome "src\hidra\cli.py"))) {
    Die "this HiDRA checkout has no src\hidra (is '$HidraRef' the PyTorch main?)"
}

# --- 3. a torch interpreter with HiDRA installed ----------------------------------------------
Step "torch interpreter with HiDRA"
$haveTorch = $false
if (Test-Path $py) { & $py -c "import torch, hidra" 2>$null; if ($LASTEXITCODE -eq 0) { $haveTorch = $true } }
if ($haveTorch) {
    Ok "already present"
} else {
    if (-not (Test-Path $py)) {
        uv venv --python 3.12 $venv
        if ($LASTEXITCODE -ne 0) { Die "could not create a venv at $venv" }
    }
    Note "installing HiDRA + PyTorch (torch backend '$TorchBackend') -- a few minutes"
    if ($TorchBackend -eq "cpu") {
        uv pip install --python $py --index-url https://download.pytorch.org/whl/cpu "torch>=2.6"
        if ($LASTEXITCODE -ne 0) { Die "installing CPU torch failed" }
        uv pip install --python $py -e "$hidraHome"
    } else {
        # uv picks the right CUDA (or CPU) wheels for this machine when TorchBackend is 'auto'.
        uv pip install --python $py --torch-backend=$TorchBackend -e "$hidraHome[torch]"
    }
    if ($LASTEXITCODE -ne 0) {
        Die "installing HiDRA failed. If it was the GPU wheel, re-run with -TorchBackend cpu."
    }
    Ok "installed"
}
$backend = (& $py -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>$null)
if ($LASTEXITCODE -ne 0) { Die "torch is installed but will not import in $py" }
Ok "torch backend: $backend$(if ($backend -eq 'cpu') { '  (no GPU detected -- inference will be slower)' })"

# --- 4. model weights --------------------------------------------------------------------------
Step "model weights"
$models = Join-Path $hidraHome "models"
$have = 0
if (Test-Path $models) {
    $have = (Get-ChildItem $models -Include *.pkl, *.safetensors -Recurse -ErrorAction SilentlyContinue).Count
}
if ($have -ge 10) {
    Ok "$have weight files already present (reused)"
} else {
    Note "downloading weights (~660 MB) from Hugging Face -- the slow step"
    & $py -m hidra.download_models
    $have = (Get-ChildItem $models -Include *.pkl, *.safetensors -Recurse -ErrorAction SilentlyContinue).Count
    if ($have -lt 10) { Die "only $have weight files arrived; re-run to resume" }
    Ok "$have weight files"
}

# --- 5. the labeler ----------------------------------------------------------------------------
Step "installing the labeler"
uv tool install --force "git+https://github.com/talmolab/laras-labeler@$LabelerRef"
if ($LASTEXITCODE -ne 0) {
    Die "install failed. If it says Access is denied, a labeler is still running -- close that window and re-run."
}
Ok "installed from $LabelerRef"
$uvBin = Join-Path $HOME ".local\bin"
if (Test-Path $uvBin) { $env:PATH = "$uvBin;$env:PATH"; Note "added $uvBin to PATH for this session" }

# --- 6. run ------------------------------------------------------------------------------------
Step "pointing the labeler at the torch checkout"
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

Write-Host "`nStarting. The banner should read: 82 classifiers available - backend: torch:$backend." -ForegroundColor White
Write-Host "Leave this window open -- it IS the server. Ctrl+C stops it.`n" -ForegroundColor DarkGray
hidra-in-the-loop $Projects
