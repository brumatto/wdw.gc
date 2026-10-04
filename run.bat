@echo off
setlocal

echo ========================================================
echo  Wheeler-DeWitt Auto-Runner
echo ========================================================

:: Define o arquivo INI padrao. O usuario pode mudar isso aqui.
set INI_FILE=examples\fast.ini

:: Se o usuario arrastar um arquivo para cima do script, ele usa o arquivo arrastado
if not "%~1"=="" set INI_FILE=%~1

echo [*] Target configuration: %INI_FILE%
echo.

:: 1. Configura o CMake se for a primeira vez rodando
if not exist build\ (
    echo [*] First time setup: Configuring CMake...
    mkdir build
    cd build
    cmake ..
    cd ..
)

:: 2. Compila o codigo (se houver alteracoes no .cpp, ele atualiza sozinho)
echo [*] Compiling (Release mode)...
cmake --build build --config Release -j
echo.

:: 3. Executa o programa buscando automaticamente onde o CMake guardou o .exe
echo [*] Starting Simulation...
echo --------------------------------------------------------
if exist build\Release\wdw.gc.exe (
    build\Release\wdw.gc.exe %INI_FILE%
) else if exist build\wdw.gc.exe (
    build\wdw.gc.exe %INI_FILE%
) else (
    echo [!] ERROR: Executable not found. Compilation may have failed.
)
echo --------------------------------------------------------

:: Pausa para o usuario conseguir ler o resultado antes da janela fechar
pause