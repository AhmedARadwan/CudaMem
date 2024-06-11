#ifndef CUDA_SHARED_POINTER_H
#define CUDA_SHARED_POINTER_H

#include <cuda_runtime.h>
#include <atomic>
#include <stdexcept>

namespace CudaMem {
    template <typename T>
    class shared_ptr {
    private:
        T* device_ptr;
        std::atomic<int>* ref_count;

    public:
        explicit shared_ptr(size_t size);
        ~shared_ptr();

        T* get() const;

        void copyToDevice(const T* host_ptr, size_t size);
        void copyToHost(T* host_ptr, size_t size) const;

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
