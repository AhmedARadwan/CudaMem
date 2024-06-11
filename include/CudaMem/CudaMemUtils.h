#ifndef CUDA_MEMORY_UTILS_H
#define CUDA_MEMORY_UTILS_H

#include "CudaUniquePointer.h"
#include "CudaSharedPointer.h"

namespace CudaMem {

    // Function to make a unique pointer
    template<typename T, typename... Args>
    unique_ptr<T> make_unique(Args&&... args);

    // Function to make a shared pointer
    template<typename T, typename... Args>
    shared_ptr<T> make_shared(Args&&... args);

} // namespace CudaMem

#endif // CUDA_MEMORY_UTILS_H
