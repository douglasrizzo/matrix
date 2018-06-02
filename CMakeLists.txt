cmake_minimum_required(VERSION 3.8)
project(matrix)

set(CMAKE_CXX_STANDARD 11)

# this OpenMP hack was found here https://stackoverflow.com/a/12404666
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

SET(GCC_ERROR_RETURN_TYPE "-Werror=return-type")
add_definitions(${GCC_ERROR_RETURN_TYPE})

#SET(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
#SET(GCC_COVERAGE_LINK_FLAGS    "-lgcov")
#
#add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
#add_definitions(${GCC_COVERAGE_LINK_FLAGS})

set(SOURCE_FILES test/main.cpp Matrix.hpp include/nr3/nr3.h include/nr3/eigen_unsym.h include/csv_reader/CSVReader.hpp include/mersenne_twister/MersenneTwister.hpp)
add_executable(matrix ${SOURCE_FILES})
