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
add_library(CudaSmartPointers SHARED
    src/CudaUniquePointer.cpp
    src/CudaSharedPointer.cpp
    src/CudaMemUtils.cpp
)

# Specify include directories for the library
target_include_directories(CudaSmartPointers PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(CudaSmartPointers ${CUDA_LIBRARIES})


# testing
add_subdirectory(test)


# install
set_target_properties(CudaSmartPointers PROPERTIES
    PUBLIC_HEADER "include/CudaMem/CudaUniquePointer.h;include/CudaMem/CudaSharedPointer.h;include/CudaMem/CudaMemUtils.h"
)

install(TARGETS CudaSmartPointers
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/CudaMem
)

if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY
    )

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
    )
endif()