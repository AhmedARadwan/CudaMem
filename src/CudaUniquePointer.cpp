#include "CudaMem/CudaUniquePointer.h"

namespace CudaMem {

    template <typename T>
    unique_ptr<T>::unique_ptr(size_t size) {
        cudaMalloc(&device_ptr, size * sizeof(T));
    }

    template <typename T>
    unique_ptr<T>::~unique_ptr() {
        cudaFree(device_ptr);
    }

    template <typename T>
    T* unique_ptr<T>::get() const {
        return device_ptr;
    }

    template <typename T>
    void unique_ptr<T>::copyToDevice(const T* host_ptr, size_t size) {
        cudaMemcpy(device_ptr, host_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
    }

    template <typename T>
    void unique_ptr<T>::copyToHost(T* host_ptr, size_t size) const {
        cudaMemcpy(host_ptr, device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
    }

    template <typename T>
    unique_ptr<T>::unique_ptr(unique_ptr&& other) noexcept : device_ptr(other.device_ptr) {
        other.device_ptr = nullptr;
    }

    template <typename T>
    unique_ptr<T>& unique_ptr<T>::operator=(unique_ptr&& other) noexcept {
        if (this != &other) {
            cudaFree(device_ptr);
            device_ptr = other.device_ptr;
            other.device_ptr = nullptr;
        }
        return *this;
    }

    // Explicit instantiation for supported types
    template class unique_ptr<int>;
    template class unique_ptr<float>;

} // namespace CudaMem
