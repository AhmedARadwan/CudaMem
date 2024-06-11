#include <iostream>
#include <cassert>
#include "CudaMem/CudaUniquePointer.h"

int main() {
    constexpr size_t SIZE = 10;

    // Test unique pointer
    CudaMem::unique_ptr<int> cuda_unique_ptr(SIZE);

    // Allocate host memory
    int* host_ptr = new int[SIZE];
    for (size_t i = 0; i < SIZE; ++i) {
        host_ptr[i] = i * 2;
    }

    // Copy data to device
    cuda_unique_ptr.copyToDevice(host_ptr, SIZE);

    // Copy data back to host
    int* result_ptr = new int[SIZE];
    cuda_unique_ptr.copyToHost(result_ptr, SIZE);

    // Check the result
    for (size_t i = 0; i < SIZE; ++i) {
        assert(result_ptr[i] == host_ptr[i]); // Check host-to-device copy
        std::cout << "Result[" << i << "] = " << result_ptr[i] << std::endl;
    }

    // Free memory
    delete[] host_ptr;
    delete[] result_ptr;

    return 0;
}
