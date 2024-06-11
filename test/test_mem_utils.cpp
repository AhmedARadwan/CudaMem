#include <iostream>
#include <cassert>
#include <CudaMem/CudaMemUtils.h>
#include <CudaMem/CudaSharedPointer.h>


int main() {
    // Test make_unique
    {
        auto unique_ptr = CudaMem::make_unique<int>(10);

        // Allocate host memory and initialize
        int* host_ptr = new int[1];
        host_ptr[0] = 42;

        // Copy data to device
        unique_ptr.copyToDevice(host_ptr, 1);

        // Copy data back to host
        int* result_ptr = new int[1];
        unique_ptr.copyToHost(result_ptr, 1);

        // Check the result
        assert(result_ptr[0] == host_ptr[0]);
        std::cout << "make_unique test passed." << std::endl;

        // Free memory
        delete[] host_ptr;
        delete[] result_ptr;
    }

    // Test make_shared
    {
        auto shared_ptr = CudaMem::make_shared<int>(10);

        // Allocate host memory and initialize
        int* host_ptr = new int[1];
        host_ptr[0] = 42;

        // Copy data to device
        shared_ptr.copyToDevice(host_ptr, 1);

        // Copy data back to host
        int* result_ptr = new int[1];
        shared_ptr.copyToHost(result_ptr, 1);

        // Check the result
        assert(result_ptr[0] == host_ptr[0]);
        std::cout << "make_shared test passed." << std::endl;

        // Free memory
        delete[] host_ptr;
        delete[] result_ptr;
    }

    return 0;
}
