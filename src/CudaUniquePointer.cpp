#include "CudaUniquePointer.h"
#include <stdexcept>

template <typename T>
CudaUniquePointer<T>::CudaUniquePointer(size_t size) {
    cudaError_t err = cudaMalloc(&device_ptr, size * sizeof(T));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory");
    }
}

template <typename T>
CudaUniquePointer<T>::~CudaUniquePointer() {
    cudaFree(device_ptr);
}

template <typename T>
T* CudaUniquePointer<T>::get() const {
    return device_ptr;
}

template <typename T>
void CudaUniquePointer<T>::copyToDevice(const T* host_ptr, size_t size) {
    cudaError_t err = cudaMemcpy(device_ptr, host_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data to device");
    }
}

template <typename T>
void CudaUniquePointer<T>::copyToHost(T* host_ptr, size_t size) const {
    cudaError_t err = cudaMemcpy(host_ptr, device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data to host");
    }
}

template <typename T>
CudaUniquePointer<T>::CudaUniquePointer(CudaUniquePointer&& other) noexcept : device_ptr(other.device_ptr) {
    other.device_ptr = nullptr;
}

template <typename T>
CudaUniquePointer<T>& CudaUniquePointer<T>::operator=(CudaUniquePointer&& other) noexcept {
    if (this != &other) {
        cudaFree(device_ptr);
        device_ptr = other.device_ptr;
        other.device_ptr = nullptr;
    }
    return *this;
}

// Explicit template instantiation
template class CudaUniquePointer<int>;  // Add other types as needed
