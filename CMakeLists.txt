cmake_minimum_required(VERSION 3.16)

project(bfloat16 VERSION 1.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 99)

option(BFLOAT16_BUILD_TESTS "Build the bfloat16 test suite" ON)
option(BFLOAT16_WERROR "Treat warnings as errors" OFF)

add_library(bfloat16 STATIC
    src/bfloat16_convo.cpp
    src/bfloat_arithemetics.cpp
    src/bfloat_comp.cpp
    src/bfloat16_bias.cpp
)
target_include_directories(bfloat16 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(bfloat16 PRIVATE -Wall -Wextra -Wpedantic)
if(BFLOAT16_WERROR)
    target_compile_options(bfloat16 PRIVATE -Werror)
endif()

# Math library (fma/ldexp) on platforms that need it.
find_library(MATH_LIB m)
if(MATH_LIB)
    target_link_libraries(bfloat16 PUBLIC ${MATH_LIB})
endif()

if(BFLOAT16_BUILD_TESTS)
    enable_testing()
    add_executable(bfloat_test bfloat_test.cpp)
    target_link_libraries(bfloat_test PRIVATE bfloat16)
    target_compile_options(bfloat_test PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME bfloat_test COMMAND bfloat_test)
endif()

include(GNUInstallDirs)
install(TARGETS bfloat16 ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/bfloat16.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
