#!/bin/bash

# Interrompe o script se ocorrer algum erro
set -e

echo "============================================="
echo "   Compiling and Running wdw.gc (Linux)"
echo "============================================="

# 1. Cria a pasta de build (se não existir) e entra nela
mkdir -p build
cd build

# 2. Configura o projeto com o CMake
echo ">>> Configuring CMake..."
cmake ..

# 3. Compila o código usando todos os núcleos disponíveis (-j)
echo ">>> Compiling..."
make -j$(nproc)

# 4. Executa o programa (ajuste o caminho do arquivo .ini se necessário)
echo ">>> Running wdw.gc..."
./wdw.gc ../fast.ini

echo "============================================="
echo "   Execution completed successfully!         "
echo "============================================="