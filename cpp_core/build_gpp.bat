@echo off
REM Build with plain g++ (no CMake needed). Works with the old MinGW g++ 6.3 too.
REM First run downloads Eigen 3.4.0 (header-only) into third_party\eigen using git.
setlocal
cd /d "%~dp0"
if not exist third_party\eigen\Eigen (
  echo Downloading Eigen 3.4.0 ...
  git clone --depth 1 --branch 3.4.0 https://github.com/eigen-mirror/eigen.git third_party\eigen || goto :fail
)
if not exist bin mkdir bin
if not exist results mkdir results
set FLAGS=-std=c++14 -O2 -Iinclude -Ithird_party\eigen
set CORE=src\mlp.cpp src\data.cpp src\neurons.cpp src\learning_rules.cpp
echo Building unit_tests ... & g++ %FLAGS% tests\unit_tests.cpp %CORE% -o bin\unit_tests.exe || goto :fail
echo Building neurons ...    & g++ %FLAGS% src\neurons_main.cpp %CORE% -o bin\neurons.exe || goto :fail
echo Building sandbox ...    & g++ %FLAGS% src\sandbox_main.cpp %CORE% -o bin\sandbox.exe || goto :fail
echo Building mlp_train ...  & g++ %FLAGS% src\mlp_train_main.cpp %CORE% -o bin\mlp_train.exe || goto :fail
echo Building learning_rules ... & g++ %FLAGS% src\learning_rules_main.cpp %CORE% -o bin\learning_rules.exe || goto :fail
echo.
echo Build OK. Running unit tests:
bin\unit_tests.exe
goto :eof
:fail
echo BUILD FAILED
exit /b 1
