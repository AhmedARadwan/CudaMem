#ifndef CUDA_UNIQUE_POINTER_H
#define CUDA_UNIQUE_POINTER_H

#include <cuda_runtime.h>

template <typename T>
class CudaUniquePointer {
private:
    T* device_ptr;

public:
    explicit CudaUniquePointer(size_t size);
    ~CudaUniquePointer();

    T* get() const;

    void copyToDevice(const T* host_ptr, size_t size);
    void copyToHost(T* host_ptr, size_t size) const;

    // Prevent copying
    CudaUniquePointer(const CudaUniquePointer&) = delete;
    CudaUniquePointer& operator=(const CudaUniquePointer&) = delete;

    // Allow moving
    CudaUniquePointer(CudaUniquePointer&& other) noexcept;
    CudaUniquePointer& operator=(CudaUniquePointer&& other) noexcept;
};

#endif // CUDA_UNIQUE_POINTER_H
