# CudaSmartPointers

CudaSmartPointers is a C++ library that provides smart pointers for managing memory on CUDA-enabled devices. It includes implementations of both unique and shared pointers, allowing you to efficiently allocate, deallocate, and transfer data between the host and device memory.

## Installation

To install CudaSmartPointers, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/CudaSmartPointers.git
    ```

2. Navigate to the project directory:

    ```bash
    cd CudaSmartPointers
    ```

3. Create a build directory:

    ```bash
    mkdir build
    cd build
    ```

4. Configure the project with CMake:

    ```bash
    cmake ..
    ```

5. Build the project:

    ```bash
    make
    ```

6. Optionally, install the library:

    ```bash
    sudo make install
    ```

## Example Usage

Here's a simple example demonstrating how to use the unique pointer from the CudaSmartPointers library:

```cpp
#include <iostream>
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
        std::cout << "Result[" << i << "] = " << result_ptr[i] << std::endl;
    }

    // Free memory
    delete[] host_ptr;
    delete[] result_ptr;

    return 0;
}

```