#ifndef CUDA_SHARED_POINTER_H
#define CUDA_SHARED_POINTER_H

#include <cuda_runtime.h>
#include <atomic>

template <typename T>
class CudaSharedPointer {
private:
    T* device_ptr;
    std::atomic<int>* ref_count;

public:
    explicit CudaSharedPointer(size_t size);
    ~CudaSharedPointer();

    T* get() const;

    void copyToDevice(const T* host_ptr, size_t size);
    void copyToHost(T* host_ptr, size_t size) const;

    // Copy constructor
    CudaSharedPointer(const CudaSharedPointer& other);
    // Copy assignment
    CudaSharedPointer& operator=(const CudaSharedPointer& other);

    // Move constructor
    CudaSharedPointer(CudaSharedPointer&& other) noexcept;
    // Move assignment
    CudaSharedPointer& operator=(CudaSharedPointer&& other) noexcept;
};

#endif // CUDA_SHARED_POINTER_H
