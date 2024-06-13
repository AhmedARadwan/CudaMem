#include "CudaMem.h"
#include <memory>
#include <vector>
#include <iostream>

int main(){

    std::vector<long> host_indexes = {3213213,321321,3456445,65657564543};
    CudaMem::shared_ptr<long> device_indexes;

    device_indexes.copyToDevice(host_indexes);

    std::vector<long> host_indexes_2(device_indexes.getSize());
    device_indexes.copyToHost(host_indexes_2);


    for (const auto& i : host_indexes_2){
        std::cout << "Vector element: " << i << std::endl;
    }

    return 0;
}