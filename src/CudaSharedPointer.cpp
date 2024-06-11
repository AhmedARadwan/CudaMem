#include "CudaSharedPointer.h"
#include <stdexcept>

template <typename T>
CudaSharedPointer<T>::CudaSharedPointer(size_t size) : ref_count(new std::atomic<int>(1)) {
    cudaError_t err = cudaMalloc(&device_ptr, size * sizeof(T));
    if (err != cudaSuccess) {
        delete ref_count;
        throw std::runtime_error("Failed to allocate device memory");
    }
}

template <typename T>
CudaSharedPointer<T>::~CudaSharedPointer() {
    if (--(*ref_count) == 0) {
        cudaFree(device_ptr);
        delete ref_count;
    }
}

template <typename T>
T* CudaSharedPointer<T>::get() const {
    return device_ptr;
}

template <typename T>
void CudaSharedPointer<T>::copyToDevice(const T* host_ptr, size_t size) {
    cudaError_t err = cudaMemcpy(device_ptr, host_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data to device");
    }
}

template <typename T>
void CudaSharedPointer<T>::copyToHost(T* host_ptr, size_t size) const {
    cudaError_t err = cudaMemcpy(host_ptr, device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data to host");
    }
}

template <typename T>
CudaSharedPointer<T>::CudaSharedPointer(const CudaSharedPointer& other) : device_ptr(other.device_ptr), ref_count(other.ref_count) {
    ++(*ref_count);
}

template <typename T>
CudaSharedPointer<T>& CudaSharedPointer<T>::operator=(const CudaSharedPointer& other) {
    if (this != &other) {
        if (--(*ref_count) == 0) {
            cudaFree(device_ptr);
            delete ref_count;
        }
        device_ptr = other.device_ptr;
        ref_count = other.ref_count;
        ++(*ref_count);
    }
    return *this;
}

template <typename T>
CudaSharedPointer<T>::CudaSharedPointer(CudaSharedPointer&& other) noexcept : device_ptr(other.device_ptr), ref_count(other.ref_count) {
    other.device_ptr = nullptr;
    other.ref_count = nullptr;
}

template <typename T>
CudaSharedPointer<T>& CudaSharedPointer<T>::operator=(CudaSharedPointer&& other) noexcept {
    if (this != &other) {
        if (--(*ref_count) == 0) {
            cudaFree(device_ptr);
            delete ref_count;
        }
        device_ptr = other.device_ptr;
        ref_count = other.ref_count;
        other.device_ptr = nullptr;
        other.ref_count = nullptr;
    }
    return *this;
}

// Explicit template instantiation
template class CudaSharedPointer<int>;  // Add other types as needed
