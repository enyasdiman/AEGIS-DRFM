# ============================================================================
# AEGIS-DRFM Phase 8: C++20 Real-Time Engine & HIL Build Helper (Windows)
# Checks for CMake & C++20 compiler, installs via winget if requested,
# builds drfm_rt_benchmark.exe, and runs the MSPS / latency benchmark.
# ============================================================================
param(
    [switch]$InstallToolchain,
    [switch]$EnableSoapySDR
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$CppSrc   = Join-Path $RepoRoot "packages\rt-cpp-engine\cpp"
$BuildDir = Join-Path $RepoRoot "build-rt"

Write-Host "[AEGIS-RT] Checking C++20 build tools..." -ForegroundColor Cyan
$hasCmake = [bool](Get-Command cmake -ErrorAction SilentlyContinue)
$hasGpp   = [bool](Get-Command g++ -ErrorAction SilentlyContinue)
$hasCl    = [bool](Get-Command cl -ErrorAction SilentlyContinue)

if (-not $hasCmake -or (-not $hasGpp -and -not $hasCl)) {
    Write-Host "  Missing toolchain: cmake=$hasCmake, g++=$hasGpp, msvc=$hasCl" -ForegroundColor Yellow
    if ($InstallToolchain) {
        Write-Host "[AEGIS-RT] Installing CMake + WinLibs MinGW-w64 (GCC C++20) via winget..." -ForegroundColor Cyan
        winget install -e --id Kitware.CMake --accept-package-agreements --accept-source-agreements
        winget install -e --id BrechtSanders.WinLibs.POSIX.UCRT --accept-package-agreements --accept-source-agreements
        Write-Host "[AEGIS-RT] Installed! Restart PowerShell and re-run .\scripts\setup_cpp_hil.ps1" -ForegroundColor Green
        exit 0
    } else {
        Write-Host "  Run with -InstallToolchain to install CMake & MinGW-w64 automatically via winget:" -ForegroundColor Yellow
        Write-Host "    .\scripts\setup_cpp_hil.ps1 -InstallToolchain" -ForegroundColor White
        exit 1
    }
}

$soapyFlag = if ($EnableSoapySDR) { "-DAEGIS_ENABLE_SOAPYSDR=ON" } else { "-DAEGIS_ENABLE_SOAPYSDR=OFF" }
$genArgs = @()
if ($hasGpp -and -not $hasCl) {
    if (Get-Command ninja -ErrorAction SilentlyContinue) {
        $genArgs = @("-G", "Ninja")
    } else {
        $genArgs = @("-G", "MinGW Makefiles")
    }
}
if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
Write-Host "[AEGIS-RT] Configuring CMake ($soapyFlag)..." -ForegroundColor Cyan
cmake -S $CppSrc -B $BuildDir @genArgs -DCMAKE_BUILD_TYPE=Release $soapyFlag
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cmake --build $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ExePath = Join-Path $BuildDir "drfm_rt_benchmark.exe"
if (-not (Test-Path $ExePath)) {
    $ExePath = Join-Path $BuildDir "Release\drfm_rt_benchmark.exe"
}
Write-Host "[AEGIS-RT] Running C++20 Streaming Latency Benchmark..." -ForegroundColor Green
& $ExePath
