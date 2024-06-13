#include "CudaMem/CudaSharedPointer.h"

namespace CudaMem {
    template <typename T>
    shared_ptr<T>::shared_ptr() : device_ptr(nullptr), ref_count(new std::atomic<int>(1)), size(0), allocated(false) {}

    template <typename T>
    shared_ptr<T>::shared_ptr(size_t size) : ref_count(new std::atomic<int>(1)), size(size), allocated(true) {
        cudaError_t err = cudaMalloc(&device_ptr, size * sizeof(T));
        if (err != cudaSuccess) {
            delete ref_count;
            throw std::runtime_error("Failed to allocate device memory");
        }
    }

    template <typename T>
    shared_ptr<T>::~shared_ptr() {
        if (--(*ref_count) == 0) {
            cudaFree(device_ptr);
            delete ref_count;
        }
    }

    template <typename T>
    T* shared_ptr<T>::get() const {
        return device_ptr;
    }

    template <typename T>
    size_t shared_ptr<T>::getSize() const {
        return size;
    }

    template <typename T>
    void shared_ptr<T>::copyToDevice(const T* host_ptr, size_t size) {
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
    void shared_ptr<T>::copyToDevice(const std::shared_ptr<T>& host_ptr, size_t size) {
        copyToDevice(host_ptr.get(), size ? size : this->size);
    }

    template <typename T>
    void shared_ptr<T>::copyToDevice(const std::unique_ptr<T[]>& host_ptr, size_t size) {
        copyToDevice(host_ptr.get(), size ? size : this->size);
    }

    template <typename T>
    void shared_ptr<T>::copyToDevice(const std::vector<T>& host_vector) {
        copyToDevice(host_vector.data(), host_vector.size());
    }

    template <typename T>
    void shared_ptr<T>::copyToHost(T* host_ptr) const {
        if (!allocated) {
            throw std::runtime_error("Device memory not allocated");
        }
        cudaError_t err = cudaMemcpy(host_ptr, device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy data to host");
        }
    }

    template <typename T>
    void shared_ptr<T>::copyToHost(std::shared_ptr<T>& host_ptr) const {
        copyToHost(host_ptr.get());
    }

    template <typename T>
    void shared_ptr<T>::copyToHost(std::unique_ptr<T[]>& host_ptr) const {
        copyToHost(host_ptr.get());
    }

    template <typename T>
    void shared_ptr<T>::copyToHost(std::vector<T>& host_vector) const {
        if (host_vector.size() != size) {
            throw std::runtime_error("Size mismatch between device and host vector");
        }
        copyToHost(host_vector.data());
    }

    template <typename T>
    shared_ptr<T>::shared_ptr(const shared_ptr& other) : device_ptr(other.device_ptr), ref_count(other.ref_count), size(other.size), allocated(other.allocated) {
        ++(*ref_count);
    }

    template <typename T>
    shared_ptr<T>& shared_ptr<T>::operator=(const shared_ptr& other) {
        if (this != &other) {
            if (--(*ref_count) == 0) {
                cudaFree(device_ptr);
                delete ref_count;
            }
            device_ptr = other.device_ptr;
            ref_count = other.ref_count;
            size = other.size;
            allocated = other.allocated;
            ++(*ref_count);
        }
        return *this;
    }

    template <typename T>
    shared_ptr<T>::shared_ptr(shared_ptr&& other) noexcept : device_ptr(other.device_ptr), ref_count(other.ref_count), size(other.size), allocated(other.allocated) {
        other.device_ptr = nullptr;
        other.ref_count = nullptr;
    }

    template <typename T>
    shared_ptr<T>& shared_ptr<T>::operator=(shared_ptr&& other) noexcept {
        if (this != &other) {
            if (--(*ref_count) == 0) {
                cudaFree(device_ptr);
                delete ref_count;
            }
            device_ptr = other.device_ptr;
            ref_count = other.ref_count;
            size = other.size;
            allocated = other.allocated;
            other.device_ptr = nullptr;
            other.ref_count = nullptr;
        }
        return *this;
    }

    // Explicit template instantiation
    template class shared_ptr<int>;
    template class shared_ptr<float>;
    template class shared_ptr<long>;

} // namespace CudaMem
