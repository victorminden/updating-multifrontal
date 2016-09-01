cmake_minimum_required(VERSION 2.8.12)

# Project name
project(MF_UPDATE_SPARSE)

# Version 0.1
set(MF_UPDATE_SOLVER_VERSION_MAJOR 0)
set(MF_UPDATE_SOLVER_VERSION_MINOR 1)

# Will use this to optimize or not
option(DEBUG "Debug mode." OFF)
option(VERBOSE "Print verbose information during development." ON)
option(HASCPP11TIMING "Use C++11 for timing linear algebra." ON)
option(GPROF "compile with -pg." OFF)
option(MKL "use mkl blas/lapack." OFF)

# Set CXX flags
if(DEBUG)
  add_compile_options(-g -Wall -DDEBUG)
else()
  add_compile_options(-O3 -g)
endif()
if(VERBOSE)
  add_compile_options(-DVERBOSE)
endif()
if(HASCPP11TIMING)
  add_compile_options(-DHASCPP11TIMING)
  add_compile_options(-std=c++0x)
endif()
if(GPROF)
  add_compile_options(-pg)
endif()

if(MKL)
set(MATH_LIBS mkl_intel_lp64 mkl_core mkl_sequential pthread m)
link_directories(${MKLROOT}/lib/intel64${MKLROOT}/lib/intel64)
#add_compile_options(-I${MKLROOT}/include)
add_compile_options(-mkl=sequential)
endif()

# subdirectories
subdirs(test_src)

# Find BLAS/LAPACK libraries
if(NOT MATH_LIBS)
  set(MATH_DESC "BLAS/LAPACK link flags")
  # Look for default BLAS/LAPACK
  set(REFERENCE_REQUIRED LAPACK BLAS)
  find_library(BLAS_LIB
               NAMES blas blas.a blas.lib
               PATHS ${REFERENCE_ROOT})
  find_library(LAPACK_LIB
               NAMES lapack lapack.a lapack.lib
                     reflapack reflapack.a reflapack.lib
               PATHS ${REFERENCE_ROOT})
  set(REFERENCE_FOUND ON)
  set(MATH_LIBS "")
  foreach(NAME ${REFERENCE_REQUIRED})
    if( ${NAME}_LIB )
      message(STATUS "Found ${NAME}_LIB: ${${NAME}_LIB}")
      list(APPEND MATH_LIBS ${${NAME}_LIB})
    else()
      message(STATUS "Could not find ${NAME}_LIB")
      set(REFERENCE_FOUND OFF)
    endif()
  endforeach()
  message(STATUS "REFERENCE_FOUND=${REFERENCE_FOUND}")
  if(REFERENCE_FOUND)
    message(STATUS "WARNING: Using reference BLAS/LAPACK.")
    message(STATUS "MATH_LIBS=${MATH_LIBS}")
  else()
    set(MATH_LIBS "" CACHE STRING ${MATH_DESC})
    message(FATAL_ERROR
      "Could not find BLAS/LAPACK libs. Please provide the root directory of MKL with -DMKL_ROOT, the directory of reference implementations with -DREFERENCE_ROOT, or manually specify all math libraries with -DMATH_LIBS. There are numerous idiosyncratic library dependencies for BLAS/LAPACK, so you will almost certainly need to manually specify -DMATH_LIBS.")
  endif()
  # Append the standard math libraries to the link list.
  list(APPEND MATH_LIBS m)
endif()


# Attempt to detect the BLAS and LAPACK underscore conventions.
# We currently only handle whether or not there is an underscore appended.
include(CheckFunctionExists)
set(CMAKE_REQUIRED_LIBRARIES ${MATH_LIBS})
check_function_exists(daxpy  HAVE_DAXPY)
check_function_exists(daxpy_ HAVE_DAXPY_POST)
if(HAVE_DAXPY)
  set(BLAS_POST FALSE)
  set(BLAS_DEFS "")
elseif(HAVE_DAXPY_POST)
  set(BLAS_POST TRUE)
  set(BLAS_DEFS "-DBLAS_POST")
else()
  message(FATAL_ERROR "Could not determine BLAS format.")
endif()
check_function_exists(dpotrf  HAVE_DPOTRF)
check_function_exists(dpotrf_ HAVE_DPOTRF_POST)
if(HAVE_DPOTRF)
  set(LAPACK_POST FALSE)
  set(LAPACK_DEFS "")
elseif(HAVE_DPOTRF_POST)
  set(LAPACK_POST TRUE)
  set(LAPACK_DEFS "-DLAPACK_POST")
else()
  message(FATAL_ERROR "Could not determine LAPACK format.")
endif()

add_compile_options(${LAPACK_DEFS})
add_compile_options(${BLAS_DEFS})


# Add project subdirectories
include_directories(${MF_UPDATE_SPARSE_SOURCE_DIR}/include/LinAlg  ${MF_UPDATE_SPARSE_SOURCE_DIR}/include/DE)
link_directories(${MF_UPDATE_SPARSE_SOURCE_DIR}/src)
