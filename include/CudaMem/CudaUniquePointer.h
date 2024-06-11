#ifndef CUDA_UNIQUE_POINTER_H
#define CUDA_UNIQUE_POINTER_H

#include <cuda_runtime.h>

namespace CudaMem{

    template <typename T>
    class unique_ptr {
    private:
        T* device_ptr;

    public:
        explicit unique_ptr(size_t size);
        ~unique_ptr();

        T* get() const;

        void copyToDevice(const T* host_ptr, size_t size);
        void copyToHost(T* host_ptr, size_t size) const;

        // Prevent copying
        unique_ptr(const unique_ptr&) = delete;
        unique_ptr& operator=(const unique_ptr&) = delete;

        // Allow moving
        unique_ptr(unique_ptr&& other) noexcept;
        unique_ptr& operator=(unique_ptr&& other) noexcept;
    };

}


#endif // CUDA_UNIQUE_POINTER_H
