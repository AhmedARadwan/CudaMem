cmake_minimum_required(VERSION 3.10)

# Set the project name
project(CudaSmartPointers LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED QUIET)
include_directories(${CUDA_INCLUDE_DIRS})
if (CUDA_VERSION VERSION_LESS "11.0")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-gpu-targets)
else ()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 -Wno-deprecated-gpu-targets)
endif ()

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 11)

# Add the library
add_library(CudaSmartPointers STATIC
    src/CudaUniquePointer.cpp
    src/CudaSharedPointer.cpp
)

# Specify include directories for the library
target_include_directories(CudaSmartPointers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} include/)
target_link_libraries(CudaSmartPointers ${CUDA_LIBRARIES})