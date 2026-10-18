# Start ComfyUI with flags tuned for this machine (6GB VRAM / 15.8GB RAM).
#
#   powershell -ExecutionPolicy Bypass -File start_comfyui.ps1
#   powershell -ExecutionPolicy Bypass -File start_comfyui.ps1 -ExtraArgs "--fast-disk"
#
# Why these flags (all measured on this machine, see comfyui/README.md):
#   --cache-none   Peak commit memory dropped from 39.8 GB to 28.8 GB. ComfyUI's
#                  default RAM cache may retain up to 100% of system RAM; without
#                  this the pagefile ballooned by ~15 GB and the process was once
#                  killed by the OS.
#   --use-sage-attention / --cuda-malloc   Already used by the aki launcher.
#
# Note: --lowvram is deliberately NOT set. It is a no-op here because dynamic VRAM
# is enabled by default, and it only affects text-encoder placement, not the
# system-memory pressure that actually caused the crash.

param(
    [string]$ExtraArgs = "",
    [int]$TimeoutSeconds = 180,
    # Use when ComfyUI-Manager's network self-check hangs the boot.
    # Disables Manager ONLY, leaving every other custom node loaded.
    [switch]$NoManager,
    [switch]$RestoreManager
)

$ErrorActionPreference = "Stop"
$root = "D:\Al\ComfyUI-aki-v3.2"
$exe = Join-Path $root "python\python.exe"
$main = Join-Path $root "ComfyUI\main.py"
$work = Join-Path $root "ComfyUI"
$mgr = Join-Path $work "custom_nodes\ComfyUI-Manager"
$mgrOff = Join-Path $work "custom_nodes\ComfyUI-Manager.disabled"

if (-not (Test-Path $exe)) { throw "python not found: $exe" }
if (-not (Test-Path $main)) { throw "main.py not found: $main" }

# NEVER use --disable-all-custom-nodes to fix a hung boot: it also kills
# ComfyUI-GGUF, and the Qwen-Image-Edit workflow needs UnetLoaderGGUF /
# CLIPLoaderGGUF, so the whole edit chain becomes unknown node classes.
# (Same for IPAdapter and UltimateSDUpscale.) Rename only the Manager dir.
#
# NOTE: keep this file pure ASCII. Windows PowerShell 5.1 reads BOM-less
# UTF-8 as GBK, and a Chinese character's trailing byte swallows the newline
# and merges the next line into the previous one (breaks param() blocks).
if ($RestoreManager) {
    if (Test-Path $mgrOff) {
        Rename-Item $mgrOff "ComfyUI-Manager"
        Write-Host "Manager re-enabled (takes effect on next start)"
    } else {
        Write-Host "Manager was already enabled"
    }
    exit 0
}
if ($NoManager) {
    if ((Test-Path $mgr) -and -not (Test-Path $mgrOff)) {
        Rename-Item $mgr "ComfyUI-Manager.disabled"
        Write-Host "Manager temporarily disabled (run -RestoreManager when done)"
    } else {
        Write-Host "Nothing to disable"
    }
}

# Already running?
# Custom nodes that fetch weights at RUN time (comfyui_controlnet_aux's pose/depth
# preprocessors use huggingface_hub) fail with LocalEntryNotFoundError on this machine,
# because huggingface.co is not reachable directly and no proxy is running. Pointing the
# whole process at the domestic mirror fixes it without touching any node's code.
# Must be set before Start-Process so the child inherits it.
$env:HF_ENDPOINT = "https://hf-mirror.com"

# ComfyUI-Manager needs GitPython. GitPython rejects .bat shims, and this
# machine's PATH git is D:\Al\bin\git.bat -> "Bad git executable" -> Manager
# IMPORT FAILED on every boot. The Aki bundle ships a real git.exe; point at it.
# Verified 2026-11: with this set, GitPython 3.1.46 reads the ComfyUI repo fine.
$gitExe = Join-Path $root "git\cmd\git.exe"
if (Test-Path $gitExe) { $env:GIT_PYTHON_GIT_EXECUTABLE = $gitExe }

$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*ComfyUI*main.py*" }
if ($existing) {
    Write-Host "ComfyUI already running (PID $($existing.ProcessId -join ', '))"
} else {
    $args = @($main, "--preview-method", "auto", "--use-sage-attention", "--cuda-malloc", "--cache-none")
    if ($ExtraArgs) { $args += $ExtraArgs.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries) }
    Write-Host "launching: python $($args -join ' ')"
    Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory $work -WindowStyle Minimized
}

# Wait for the API to answer.
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:8188/system_stats" -TimeoutSec 5
        $vram = "{0:N2} GB free of {1:N2} GB" -f ($r.devices[0].vram_free / 1GB), ($r.devices[0].vram_total / 1GB)
        Write-Host "ComfyUI is up (v$($r.system.comfyui_version), $($r.devices[0].name))"
        Write-Host "VRAM: $vram"
        Write-Host "args: $($r.system.argv -join ' ')"
        exit 0
    } catch {
        Start-Sleep -Seconds 2
    }
}
Write-Error "ComfyUI did not respond within $TimeoutSeconds seconds"
exit 1
