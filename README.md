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
#include "CudaMem.h"
#include <memory>
#include <vector>
#include <iostream>
#include <random>

int main(){

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<long> dis(1, 1000000);

    std::vector<long> host_indexes;
    const int num_elements = 100;

    for (int i = 0; i < num_elements; ++i) {
        host_indexes.push_back(dis(gen));
    }

    CudaMem::shared_ptr<long> device_indexes;

    device_indexes.copyToDevice(host_indexes);

    std::vector<long> host_indexes_2(device_indexes.getSize());
    device_indexes.copyToHost(host_indexes_2);


    for (const auto& i : host_indexes_2){
        std::cout << "Vector element: " << i << std::endl;
    }

    return 0;
}
```