#ifndef CUDA_SHARED_POINTER_H
#define CUDA_SHARED_POINTER_H

#include <cuda_runtime.h>
#include <atomic>
#include <stdexcept>
#include <memory>
#include <vector>

namespace CudaMem {
    template <typename T>
    class shared_ptr {
    private:
        T* device_ptr;
        std::atomic<int>* ref_count;
        size_t size;
        bool allocated;

    public:
        shared_ptr();
        explicit shared_ptr(size_t size);
        ~shared_ptr();

        T* get() const;
        size_t getSize() const;

        void copyToDevice(const T* host_ptr, size_t size = 0);
        void copyToDevice(const std::shared_ptr<T>& host_ptr, size_t size = 0);
        void copyToDevice(const std::unique_ptr<T[]>& host_ptr, size_t size = 0);
        void copyToDevice(const std::vector<T>& host_vector);

        void copyToHost(T* host_ptr) const;
        void copyToHost(std::shared_ptr<T>& host_ptr) const;
        void copyToHost(std::unique_ptr<T[]>& host_ptr) const;
        void copyToHost(std::vector<T>& host_vector) const;

        // Copy constructor
        shared_ptr(const shared_ptr& other);
        // Copy assignment
        shared_ptr& operator=(const shared_ptr& other);

        // Move constructor
        shared_ptr(shared_ptr&& other) noexcept;
        // Move assignment
        shared_ptr& operator=(shared_ptr&& other) noexcept;
    };
}

#endif // CUDA_SHARED_POINTER_H
