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