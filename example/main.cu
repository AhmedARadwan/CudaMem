#include "CudaMem.h"
#include <memory>
#include <vector>
#include <iostream>
#include <random>
#include <cuda_runtime.h>

__global__ void sumBuffers(const long *buffer1, 
                           const long *buffer2, 
                           long *result, 
                           int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = buffer1[idx] + buffer2[idx];
    }
}

int main(){

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<long> dis(1, 1000000);

    // const int num_elements = 1e6;
    const int num_elements = 256;

    std::vector<long> host_indexes1;
    std::vector<long> host_indexes2;

    for (int i = 0; i < num_elements; ++i) {
        host_indexes1.push_back(dis(gen));
        host_indexes2.push_back(dis(gen));

    }

    CudaMem::shared_ptr<long> device_indexes1;
    CudaMem::shared_ptr<long> device_indexes2;
    CudaMem::shared_ptr<long> device_output(num_elements);

    device_indexes1.copyToDevice(host_indexes1);
    device_indexes2.copyToDevice(host_indexes2);

    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;
    sumBuffers<<<numBlocks, blockSize>>>(device_indexes1.get(),
                                         device_indexes2.get(),
                                         device_output.get(),
                                         device_output.getSize()
    );
    cudaDeviceSynchronize();
    
    std::vector<long> host_output(device_output.getSize());
    device_output.copyToHost(host_output);


    for (const auto& i : host_output){
        std::cout << "Vector element: " << i << std::endl;
    }

    return 0;
}