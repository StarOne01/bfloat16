cmake_minimum_required(VERSION 3.16)

project(bfloat16 VERSION 1.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(BFLOAT16_BUILD_TESTS "Build the bfloat16 test suite" ON)
option(BFLOAT16_EXHAUSTIVE_TESTS "Register the slow exhaustive tests with CTest" OFF)
option(BFLOAT16_EXACT_FMA_TESTS
       "Run the sampled exact-rational fma test when Python3 is available" ON)
option(BFLOAT16_ML_DTYPES_TESTS
       "Run the sampled Python/ml_dtypes differential test when available" ON)
option(BFLOAT16_WERROR "Treat warnings as errors" OFF)

set(BFLOAT16_SOURCES
    src/bfloat16_convo.cpp
    src/bfloat_arithmetics.cpp
    src/bfloat_comp.cpp
    src/bfloat16_bias.cpp
)

add_library(bfloat16 STATIC ${BFLOAT16_SOURCES})
target_include_directories(bfloat16 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(bfloat16 PRIVATE -Wall -Wextra -Wpedantic)
if(BFLOAT16_WERROR)
    target_compile_options(bfloat16 PRIVATE -Werror)
endif()

# Math library (nextafter/ldexp) on platforms that need it.
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

    add_executable(bfloat_test_c test_c.c)
    target_link_libraries(bfloat_test_c PRIVATE bfloat16)
    target_compile_options(bfloat_test_c PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME bfloat_test_c COMMAND bfloat_test_c)

    if(BFLOAT16_EXACT_FMA_TESTS OR BFLOAT16_ML_DTYPES_TESTS)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_library(bfloat16_test_shared SHARED ${BFLOAT16_SOURCES})
            set_target_properties(bfloat16_test_shared PROPERTIES
                WINDOWS_EXPORT_ALL_SYMBOLS ON)
            target_include_directories(bfloat16_test_shared PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include)
            target_compile_options(bfloat16_test_shared PRIVATE
                -Wall -Wextra -Wpedantic)
            if(BFLOAT16_WERROR)
                target_compile_options(bfloat16_test_shared PRIVATE -Werror)
            endif()
            if(MATH_LIB)
                target_link_libraries(bfloat16_test_shared PUBLIC ${MATH_LIB})
            endif()

            if(BFLOAT16_EXACT_FMA_TESTS)
                add_test(NAME bfloat_test_fma_exact
                    COMMAND ${Python3_EXECUTABLE}
                            ${CMAKE_CURRENT_SOURCE_DIR}/test_fma_exact.py
                            $<TARGET_FILE:bfloat16_test_shared>)
            endif()

            if(BFLOAT16_ML_DTYPES_TESTS)
                execute_process(
                    COMMAND ${Python3_EXECUTABLE} -c "import ml_dtypes"
                    RESULT_VARIABLE BFLOAT16_ML_DTYPES_IMPORT
                    OUTPUT_QUIET ERROR_QUIET
                )
                if(BFLOAT16_ML_DTYPES_IMPORT EQUAL 0)
                    add_test(NAME bfloat_test_ml_dtypes
                        COMMAND ${Python3_EXECUTABLE}
                                ${CMAKE_CURRENT_SOURCE_DIR}/test_ml_dtypes.py
                                $<TARGET_FILE:bfloat16_test_shared>)
                else()
                    message(STATUS "ml_dtypes differential test skipped: module not found")
                endif()
            endif()
        else()
            message(STATUS "Python3 not found: exact fma and ml_dtypes tests skipped")
        endif()
    endif()

    if(BFLOAT16_EXHAUSTIVE_TESTS)
        add_test(NAME bfloat_test_exhaustive COMMAND bfloat_test --exhaustive)
        set_tests_properties(bfloat_test_exhaustive PROPERTIES LABELS "exhaustive")
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS bfloat16 ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/bfloat16.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
