# Set the project name
project(CudaSmartPointersTests)

# Add the executable for unique pointer tests
add_executable(unique_pointer_test test_unique_pointer.cpp)
target_link_libraries(unique_pointer_test PRIVATE CudaSmartPointers)

# Add the executable for shared pointer tests
add_executable(shared_pointer_test test_shared_pointer.cpp)
target_link_libraries(shared_pointer_test PRIVATE CudaSmartPointers)
