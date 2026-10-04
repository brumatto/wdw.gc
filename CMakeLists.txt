# Define a versão mínima do CMake necessária
cmake_minimum_required(VERSION 3.10)

# Nome do projeto, versão e linguagem (CXX significa C++)
project(WheelerDeWittSolver VERSION 1.0 LANGUAGES CXX)

# Exige o padrão C++17 (conforme descrito no README)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Desabilita os avisos chatos da Microsoft sobre funções da linguagem C
add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

# Otimização de performance
# Se o usuário não especificar o tipo de build, usa "Release" com otimização máxima (-O3)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Busca pela biblioteca OpenMP (obrigatória para o código)
find_package(OpenMP REQUIRED)

# Define onde o compilador deve procurar os arquivos de cabeçalho (.hpp)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Lista de arquivos fonte
set(SOURCES
    src/wdw.gc.cpp
    src/wdw.gc.aux.cpp
    src/GCsolve.cpp
)

# Cria o executável final com o nome "wdw.gc" baseado nos arquivos fonte
add_executable(wdw.gc ${SOURCES})

# Conecta o OpenMP ao executável
target_link_libraries(wdw.gc PRIVATE OpenMP::OpenMP_CXX)

# Habilita o OpenMP moderno se o compilador for o Visual Studio (MSVC)
if(MSVC)
    target_compile_options(wdw.gc PRIVATE "-openmp:llvm")
endif()

# Mensagem amigável ao final do processo
message(STATUS "Configuração concluída! O projeto será compilado em modo: ${CMAKE_BUILD_TYPE}")