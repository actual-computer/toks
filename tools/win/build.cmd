@echo off
rem tools/win/build.cmd: the Makefile's lib + test rules on windows x86-64 (no make there), from the source root.
rem   tools\win\build.cmd          -> build\windows-x86_64\{libtoks.lib, toks.dll + toks.lib, tests\*.exe}
rem Compiler: %TOKS_LLVM_BIN%\clang.exe, default %USERPROFILE%\.cache\toks-llvm\21.1.8\bin (docs/machines.md, aimax395).
rem Same flags as the Makefile (OS=windows): CSTRICT for the library, CTEST for the tests, and its kernel rule:
rem every src\asm\x86_64\k*.S defines TOKS_HAVE_<NAME>=1 (the name upper-cased) for the library and the tests,
rem so the dispatch runs what is assembled, and the tests get TOKS_ASM_SOURCES (test_tier checks the wiring).
rem Also builds test_e2e against toks.dll, and per K3 asm tier present in src\asm\x86_64: test_k3 and
rem tests/k3/check.c with -DK3_SCAN=<tier>, both directly and through the abi canary harness (tests/k3/abi_k3.c).
setlocal enabledelayedexpansion
if "%TOKS_LLVM_BIN%"=="" set TOKS_LLVM_BIN=%USERPROFILE%\.cache\toks-llvm\21.1.8\bin
set CC="%TOKS_LLVM_BIN%\clang.exe"
set B=build\windows-x86_64
set CSTRICT=-std=c17 -O3 -fno-strict-aliasing -fwrapv -Wall -Wextra -Wconversion -Wsign-conversion -Werror -fno-builtin-strlen -fno-builtin-bcmp
set CTEST=-std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror
set CPP=-Iinclude -Isrc\core -Isrc\platform -D_CRT_SECURE_NO_WARNINGS
for %%d in (obj shobj tests) do if not exist %B%\%%d mkdir %B%\%%d
set HAVE=
set ASMS=
for %%f in (src\asm\x86_64\*.S) do (
  set "ASMS=!ASMS! %%~nxf"
  set "N=%%~nf"
  if /i "!N:~0,1!"=="k" (
    for %%u in (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z) do set "N=!N:%%u=%%u!"
    set "HAVE=!HAVE! -DTOKS_HAVE_!N!=1"
  )
)
if defined ASMS set "ASMS=!ASMS:~1!"
echo kernels: !HAVE!
set FAIL=0
set OBJS=
set SHOBJS=
for %%f in (src\core\*.c src\platform\*.c src\gen\*.c src\par\*.c) do (
  %CC% %CSTRICT% -fvisibility=hidden %CPP% %HAVE% -c %%f -o %B%\obj\%%~nf.o
  if errorlevel 1 (echo FAIL compile %%f & set FAIL=1)
  %CC% %CSTRICT% -DTOKS_BUILD_SHARED %CPP% %HAVE% -c %%f -o %B%\shobj\%%~nf.o
  if errorlevel 1 (echo FAIL compile shared %%f & set FAIL=1)
  set OBJS=!OBJS! %B%\obj\%%~nf.o
  set SHOBJS=!SHOBJS! %B%\shobj\%%~nf.o
)
for %%f in (src\asm\x86_64\*.S) do (
  %CC% %CPP% -Isrc\asm\x86_64 -c %%f -o %B%\obj\%%~nf.S.o
  if errorlevel 1 (echo FAIL assemble %%f & set FAIL=1)
  set OBJS=!OBJS! %B%\obj\%%~nf.S.o
  set SHOBJS=!SHOBJS! %B%\obj\%%~nf.S.o
)
if exist %B%\libtoks.lib del %B%\libtoks.lib
"%TOKS_LLVM_BIN%\llvm-lib.exe" /nologo /out:%B%\libtoks.lib %OBJS%
if errorlevel 1 (echo FAIL libtoks.lib & set FAIL=1)
%CC% -shared -o %B%\toks.dll %SHOBJS%
if errorlevel 1 (echo FAIL toks.dll & set FAIL=1)
for %%t in (tests\c\*.c) do (
  %CC% %CTEST% %CPP% %HAVE% "-DTOKS_ASM_SOURCES=\"%ASMS%\"" -Itests\common -o %B%\tests\%%~nt.exe %%t tests\common\guard.c tests\common\abicheck_x86_64.S %B%\libtoks.lib
  if errorlevel 1 (echo FAIL build %%t & set FAIL=1)
)
%CC% %CTEST% %CPP% -o %B%\tests\test_e2e_dll.exe tests\c\test_e2e.c %B%\toks.lib
if errorlevel 1 (echo FAIL build test_e2e_dll & set FAIL=1)
copy /y %B%\toks.dll %B%\tests\toks.dll >nul
for %%k in (c avx2 avx512) do (
  set HAVE=0
  if "%%k"=="c" set HAVE=1
  if exist src\asm\x86_64\k3_cl100k_%%k.S set HAVE=1
  if "!HAVE!"=="1" (
    %CC% %CTEST% %CPP% -Itests\common -DK3_SCAN=toks_k3_scan_cl100k_%%k -o %B%\tests\test_k3_%%k.exe tests\c\test_k3.c tests\common\guard.c %B%\libtoks.lib
    if errorlevel 1 (echo FAIL build test_k3_%%k & set FAIL=1)
    %CC% %CTEST% %CPP% -Itests\common -DK3_SCAN=toks_k3_scan_abi -DK3_TIER=toks_k3_scan_cl100k_%%k -o %B%\tests\test_k3_%%k_abi.exe tests\c\test_k3.c tests\k3\abi_k3.c tests\common\guard.c tests\common\abicheck_x86_64.S %B%\libtoks.lib
    if errorlevel 1 (echo FAIL build test_k3_%%k_abi & set FAIL=1)
    %CC% %CTEST% %CPP% -DK3_SCAN=toks_k3_scan_cl100k_%%k -o %B%\tests\k3check_%%k.exe tests\k3\check.c %B%\libtoks.lib
    if errorlevel 1 (echo FAIL build k3check_%%k & set FAIL=1)
    %CC% %CTEST% %CPP% -Itests\common -DK3_SCAN=toks_k3_scan_abi -DK3_TIER=toks_k3_scan_cl100k_%%k -o %B%\tests\k3check_%%k_abi.exe tests\k3\check.c tests\k3\abi_k3.c tests\common\abicheck_x86_64.S %B%\libtoks.lib
    if errorlevel 1 (echo FAIL build k3check_%%k_abi & set FAIL=1)
  )
)
echo build.cmd: FAIL=%FAIL%
exit /b %FAIL%
