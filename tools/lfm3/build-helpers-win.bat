@echo off
rem tools\lfm3\build-helpers-win.bat - Windows build of xolotl-lfm3's helper programs, the counterpart of
rem build-helpers.sh: separate processes, built against a llama.cpp checkout with LFM3 support (upstream f3586e6 or
rem later), so libmtmd's ggml never meets the engine's own vendored ggml.
rem   build-helpers-win.bat clone [dir]   git clone ggml-org/llama.cpp (depth 1), then build its mtmd and llama with
rem                                        CUDA.  The default dir is third_party\llama.cpp-upstream.
rem   build-helpers-win.bat [dir]          build the helpers against that checkout (the default dir when none is given).
rem     -> build-win\xolotl-lfm3-vision.exe   (images, audio and video in)
rem     -> build-win\xolotl-lfm3-speech.exe   (speech out)
rem     -> build-win\lfm3_ref_dump.exe
rem The runtime DLLs of the checkout are copied beside them.  Re-run after changing the helper sources: the llama.cpp
rem build is cached by CMake and Ninja.  Optional, as for build-win.bat: CUDA_ARCH (default 86), JOBS (default 6), VCVARS
rem (the path to vcvars64.bat), CUDA_PATH (the toolkit), OUT (the engine's build directory, default build-win).
setlocal
cd /d "%~dp0..\.."
if "%VCVARS%"=="" (
  for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
)
if not exist "%VCVARS%" (
  echo build-helpers-win: Visual Studio 2022 C++ tools not found. Install "Desktop development with C++" or set VCVARS.
  exit /b 1
)
call "%VCVARS%" >nul || exit /b 1
if "%CUDA_PATH%"=="" (
  echo build-helpers-win: CUDA_PATH is not set. Install the CUDA toolkit 12.8+ or set CUDA_PATH.
  exit /b 1
)
rem a local venv with cmake + ninja (python -m venv .venv-build && .venv-build\Scripts\pip install cmake ninja)
if exist "%~dp0..\..\.venv-build\Scripts" set "PATH=%~dp0..\..\.venv-build\Scripts;%PATH%"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "NVCC=%CUDA_PATH:\=/%/bin/nvcc.exe"
if "%CUDA_ARCH%"=="" set "CUDA_ARCH=86"
if "%JOBS%"=="" set "JOBS=6"
if "%OUT%"=="" set "OUT=build-win"
set "L=third_party\llama.cpp-upstream"
set "DOCLONE="
if not "%~1"=="" set "L=%~1"
if /i "%~1"=="clone" (
  set "DOCLONE=1"
  set "L=third_party\llama.cpp-upstream"
  if not "%~2"=="" set "L=%~2"
)
for /f "delims=" %%i in ("%L%") do set "L=%%~fi"
if defined DOCLONE if not exist "%L%\" git clone --depth 1 https://github.com/ggml-org/llama.cpp "%L%" || exit /b 1
if not exist "%L%\include\llama.h" (
  echo build-helpers-win: "%L%" is not a llama.cpp checkout ^(no include\llama.h^). "build-helpers-win.bat clone" makes one.
  exit /b 1
)
rem cl.exe prints its /showIncludes notes in the console code page, and CMake writes the localized prefix that Ninja
rem matches them against into rules.ninja in the code page it generates under (build-win.bat says more): configure and
rem build under UTF-8, then restore the caller's code page.
for /f "tokens=2 delims=:." %%c in ('chcp') do set "OLDCP=%%c"
chcp 65001 >nul
set "RCODE=0"
rem The checkout's own build, with Ninja as build-win.bat does (one build directory, %L%\build).
rem mtmd's LFM3 support postdates the vendored snapshot, which is why the helpers cannot use the engine's ggml.
if not exist "%L%\build\build.ninja" (
  cmake -S "%L%" -B "%L%\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH% "-DCMAKE_CUDA_COMPILER=%NVCC%" || set "RCODE=1"
)
if "%RCODE%"=="0" ninja -C "%L%\build" -j %JOBS% mtmd llama || set "RCODE=1"
rem The import libraries of the checkout's DLLs.  Where CMake puts them moves between commits of llama.cpp (in one
rem day, ggml-cuda.lib went from build\ggml\src to build\ggml\src\ggml-cuda), so each is looked up by its name.
set "LIBS="
if "%RCODE%"=="0" for %%n in (mtmd llama ggml ggml-base ggml-cpu ggml-cuda) do call :addlib %%n || set "RCODE=1"
set "LIBS=%LIBS% "%CUDA_PATH%\lib\x64\cudart.lib""
rem The flags mirror build-helpers.sh.  Absolute paths throughout: cl /link looks its inputs up from the working
rem directory of the process, which must not be assumed.
rem The object files go to %OUT% too: cl writes them into the working directory unless /Fo says where.
set "INC=-Itools\vision -I"%L%\include" -I"%L%\ggml\include" -I"%L%\tools\mtmd""
if not exist "%OUT%" mkdir "%OUT%"
if "%RCODE%"=="0" cl /nologo /O2 /std:c++17 /EHsc /utf-8 /W3 %INC% tools\vision\strata_vision.cpp /Fo"%OUT%\strata_vision.obj" /Fe"%OUT%\xolotl-lfm3-vision.exe" /link %LIBS% || set "RCODE=1"
if "%RCODE%"=="0" cl /nologo /O2 /std:c++17 /EHsc /utf-8 /W3 %INC% tools\lfm3\lfm3_speech.cpp /Fo"%OUT%\lfm3_speech.obj" /Fe"%OUT%\xolotl-lfm3-speech.exe" /link %LIBS% || set "RCODE=1"
if "%RCODE%"=="0" cl /nologo /O2 /std:c++17 /EHsc /utf-8 /W3 -I"%L%\include" -I"%L%\ggml\include" tools\lfm3\lfm3_ref_dump.cpp /Fo"%OUT%\lfm3_ref_dump.obj" /Fe"%OUT%\lfm3_ref_dump.exe" /link %LIBS% || set "RCODE=1"
rem The helpers load the ggml, llama and mtmd of the checkout (a standalone build), never the engine's: the DLLs go
rem beside the executables.
if "%RCODE%"=="0" copy /y "%L%\build\bin\ggml*.dll" "%OUT%\" >nul
if "%RCODE%"=="0" copy /y "%L%\build\bin\llama.dll" "%OUT%\" >nul
if "%RCODE%"=="0" copy /y "%L%\build\bin\mtmd.dll" "%OUT%\" >nul
if defined OLDCP chcp %OLDCP: =% >nul
if "%RCODE%"=="0" echo built %OUT%\xolotl-lfm3-vision.exe, xolotl-lfm3-speech.exe and lfm3_ref_dump.exe
exit /b %RCODE%

:addlib
rem %1 = the name of a library of the checkout.  Puts its one import library into LIBS; errorlevel 1 when there is none.
set "_found="
for /f "delims=" %%f in ('dir /s /b "%L%\build\%1.lib" 2^>nul') do set "_found=%%f"
if not defined _found (
  echo build-helpers-win: %1.lib is not in the build of "%L%".  Is the checkout built with Ninja and GGML_CUDA=ON^? "clone" builds it that way.
  exit /b 1
)
set "LIBS=%LIBS% "%_found%""
exit /b 0
