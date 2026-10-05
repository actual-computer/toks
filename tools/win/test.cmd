@echo off
rem tools/win/test.cmd [CASES.bin ...]: runs what tools\win\build.cmd built, from the source root (make test's set,
rem test_e2e against toks.dll, the K3 tier tests direct and through the abi canary harness), then the hf 0.23.2
rem differential (tests/k3/check.c) on each case file of tests/k3/gen.py, every K3 tier built.
rem Tokenizer files: %TOKS_TOKENIZER_CACHE%, default %USERPROFILE%\.cache\toks\tokenizers (gpt2 llama3 glm53 ...).
setlocal enabledelayedexpansion
if "%TOKS_TOKENIZER_CACHE%"=="" set TOKS_TOKENIZER_CACHE=%USERPROFILE%\.cache\toks\tokenizers
set B=build\windows-x86_64\tests
set FAIL=0
echo TOKS_TOKENIZER_CACHE=%TOKS_TOKENIZER_CACHE%
for %%t in (tests\c\*.c) do call :run %%~nt
call :run test_e2e_dll
for %%k in (c avx2 avx512) do (
  if exist %B%\test_k3_%%k.exe call :run test_k3_%%k
  if exist %B%\test_k3_%%k_abi.exe call :run test_k3_%%k_abi
)
:cases
if "%~1"=="" goto done
for %%k in (c avx2 avx512) do (
  if exist %B%\k3check_%%k.exe call :run k3check_%%k %1
  if exist %B%\k3check_%%k_abi.exe call :run k3check_%%k_abi %1
)
shift
goto cases
:done
echo test.cmd: FAIL=%FAIL%
exit /b %FAIL%

:run
echo == %1 %2
%B%\%1.exe %2
rem any nonzero exit fails: a crash exits negative (0xC0000005 = -1073741819), which "if errorlevel 1" misses
if %errorlevel% neq 0 (echo -- FAIL %1 exit %errorlevel% & set FAIL=1)
goto :eof
