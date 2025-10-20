cmake_minimum_required(VERSION 3.28)

project(polyai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_SCAN_FOR_MODULES ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(BUILD_MULTITHREADING OFF CACHE BOOL "" FORCE)
set(USE_GRAPHICAL_BENCHMARK OFF CACHE BOOL "" FORCE)
set(USE_GLUT OFF CACHE BOOL "" FORCE)
set(BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
set(BUILD_DEMOS OFF CACHE BOOL "" FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(ammo.js/bullet EXCLUDE_FROM_ALL)

file(GLOB_RECURSE MODULE_SOURCES
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.ixx
)

file(GLOB_RECURSE CPP_SOURCES
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(polyai
    ${CPP_SOURCES}
)

if(MODULE_SOURCES)
    target_sources(polyai
        PUBLIC
            FILE_SET CXX_MODULES
                FILES ${MODULE_SOURCES}
    )
endif()

target_compile_options(polyai PRIVATE
    -fno-exceptions
    -fno-rtti
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>
    $<$<CONFIG:Release>:-flto>
)

target_link_options(polyai PRIVATE
    $<$<CONFIG:Release>:-flto>
    $<$<CONFIG:Release>:-s>
)

target_link_libraries(polyai PRIVATE
    BulletDynamics
    BulletCollision
    LinearMath
)

target_include_directories(polyai PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ammo.js/bullet/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
