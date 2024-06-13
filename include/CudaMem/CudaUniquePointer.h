#ifndef CUDA_UNIQUE_POINTER_H
#define CUDA_UNIQUE_POINTER_H

#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace CudaMem {
    template <typename T>
    class unique_ptr {
    private:
        T* device_ptr;
        size_t size;
        bool allocated;

    public:
        unique_ptr();
        explicit unique_ptr(size_t size);
        ~unique_ptr();

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

        // Prevent copying
        unique_ptr(const unique_ptr&) = delete;
        unique_ptr& operator=(const unique_ptr&) = delete;

        // Allow moving
        unique_ptr(unique_ptr&& other) noexcept;
        unique_ptr& operator=(unique_ptr&& other) noexcept;
    };
}

#endif // CUDA_UNIQUE_POINTER_H
