@echo off
rem Windows build of the DeepSeek-V4.1 engine (MSVC + CUDA, native; no WSL). See docs/WINDOWS.md.
rem   build-win.bat configure      - CMake configure into build-win
rem   build-win.bat <target...>    - build the given targets (default ds41_generate)
rem Optional: CUDA_ARCH (default 86 = RTX 30xx; 89 = RTX 40xx, 120 = RTX 50xx), JOBS (default 6), VCVARS (path to
rem vcvars64.bat), CUDA_PATH (the CUDA toolkit; the installer sets it).
setlocal
cd /d "%~dp0"
if "%VCVARS%"=="" (
  for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
)
if not exist "%VCVARS%" (
  echo build-win: Visual Studio 2022 C++ tools not found. Install "Desktop development with C++" or set VCVARS.
  exit /b 1
)
call "%VCVARS%" >nul || exit /b 1
if "%CUDA_PATH%"=="" (
  echo build-win: CUDA_PATH is not set. Install the CUDA toolkit 12.8+ or set CUDA_PATH.
  exit /b 1
)
if "%CUDA_ARCH%"=="" set "CUDA_ARCH=86"
rem a local venv with cmake + ninja (python -m venv .venv-build && .venv-build\Scripts\pip install cmake ninja)
if exist "%~dp0.venv-build\Scripts" set "PATH=%~dp0.venv-build\Scripts;%PATH%"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "NVCC=%CUDA_PATH:\=/%/bin/nvcc.exe"
if /i "%~1"=="configure" (
  cmake -S . -B build-win -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON -DSTRATA_MMQ_KQUANTS=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH% "-DCMAKE_CUDA_COMPILER=%NVCC%" ^
    -DSTRATA_GGML_DIR=%~dp0third_party/llama.cpp ^
    -DSTRATA_DSV41_FQ_OP=ON -DSTRATA_DSV41_HC_MIX=ON
  exit /b %errorlevel%
)
set "TARGETS=%*"
if "%TARGETS%"=="" set "TARGETS=ds41_generate"
rem Cap the parallel jobs: each nvcc job can take several GB of RAM, and the default (cores + 2) froze a 128 GB PC.
if "%JOBS%"=="" set "JOBS=6"
ninja -C build-win -k 0 -j %JOBS% %TARGETS%
