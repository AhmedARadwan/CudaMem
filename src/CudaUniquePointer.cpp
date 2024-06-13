#include "CudaMem/CudaUniquePointer.h"

namespace CudaMem {

    template <typename T>
    unique_ptr<T>::unique_ptr() : device_ptr(nullptr), size(0), allocated(false) {}

    template <typename T>
    unique_ptr<T>::unique_ptr(size_t size) : size(size), allocated(true) {
        cudaError_t err = cudaMalloc(&device_ptr, size * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory");
        }
    }

    template <typename T>
    unique_ptr<T>::~unique_ptr() {
        if (allocated) {
            cudaFree(device_ptr);
        }
    }

    template <typename T>
    T* unique_ptr<T>::get() const {
        return device_ptr;
    }

    template <typename T>
    size_t unique_ptr<T>::getSize() const {
        return size;
    }

    template <typename T>
    void unique_ptr<T>::copyToDevice(const T* host_ptr, size_t size) {
        if (!allocated) {
            this->size = size;
            cudaError_t err = cudaMalloc(&device_ptr, size * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate device memory");
            }
            allocated = true;
        } else if (size != 0 && size != this->size) {
            throw std::runtime_error("Size mismatch during copy to device");
        }
        cudaError_t err = cudaMemcpy(device_ptr, host_ptr, this->size * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy data to device");
        }
    }

    template <typename T>
    void unique_ptr<T>::copyToDevice(const std::shared_ptr<T>& host_ptr, size_t size) {
        copyToDevice(host_ptr.get(), size ? size : this->size);
    }

    template <typename T>
    void unique_ptr<T>::copyToDevice(const std::unique_ptr<T[]>& host_ptr, size_t size) {
        copyToDevice(host_ptr.get(), size ? size : this->size);
    }

    template <typename T>
    void unique_ptr<T>::copyToDevice(const std::vector<T>& host_vector) {
        copyToDevice(host_vector.data(), host_vector.size());
    }

    template <typename T>
    void unique_ptr<T>::copyToHost(T* host_ptr) const {
        if (!allocated) {
            throw std::runtime_error("Device memory not allocated");
        }
        cudaError_t err = cudaMemcpy(host_ptr, device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy data to host");
        }
    }

    template <typename T>
    void unique_ptr<T>::copyToHost(std::shared_ptr<T>& host_ptr) const {
        copyToHost(host_ptr.get());
    }

    template <typename T>
    void unique_ptr<T>::copyToHost(std::unique_ptr<T[]>& host_ptr) const {
        copyToHost(host_ptr.get());
    }

    template <typename T>
    void unique_ptr<T>::copyToHost(std::vector<T>& host_vector) const {
        if (host_vector.size() != size) {
            throw std::runtime_error("Size mismatch between device and host vector");
        }
        copyToHost(host_vector.data());
    }

    template <typename T>
    unique_ptr<T>::unique_ptr(unique_ptr&& other) noexcept : device_ptr(other.device_ptr), size(other.size), allocated(other.allocated) {
        other.device_ptr = nullptr;
        other.size = 0;
        other.allocated = false;
    }

    template <typename T>
    unique_ptr<T>& unique_ptr<T>::operator=(unique_ptr&& other) noexcept {
        if (this != &other) {
            if (allocated) {
                cudaFree(device_ptr);
            }
            device_ptr = other.device_ptr;
            size = other.size;
            allocated = other.allocated;
            other.device_ptr = nullptr;
            other.size = 0;
            other.allocated = false;
        }
        return *this;
    }

    // Explicit instantiation for supported types
    template class unique_ptr<int>;
    template class unique_ptr<float>;

} // namespace CudaMem
