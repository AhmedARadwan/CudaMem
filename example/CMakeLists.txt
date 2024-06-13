project(CudaSmartPointersExample)

# Add the executable for unique pointer tests
add_executable(example main.cpp)
target_link_libraries(example PRIVATE CudaSmartPointers)