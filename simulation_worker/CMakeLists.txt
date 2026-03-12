cmake_minimum_required(VERSION 3.16)
project(simulation_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    cmake_policy(SET CMP0091 NEW)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

set(WASMTIME_DIR ${CMAKE_SOURCE_DIR}/wasmtime)
add_subdirectory(${WASMTIME_DIR}/crates/c-api ${CMAKE_BINARY_DIR}/wasmtime)

file(GLOB_RECURSE SOURCES src/*.cpp)
add_executable(simulation_worker ${SOURCES})

target_include_directories(simulation_worker PRIVATE ${WASMTIME_DIR}/crates/c-api/include)
target_link_libraries(simulation_worker PRIVATE wasmtime)
target_compile_features(simulation_worker PRIVATE cxx_std_17)
