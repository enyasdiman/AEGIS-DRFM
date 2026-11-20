#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace aegis::rt {

/// Cache-line-padded Lock-Free Single-Producer Single-Consumer (SPSC) Ring Buffer
/// for deterministic, zero-allocation real-time IQ sample streaming.
template <typename T>
class SpscRingBuffer {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

public:
    explicit SpscRingBuffer(std::size_t capacity_pow2)
        : mask_(capacity_pow2 - 1), buffer_(capacity_pow2) {
        // Require power-of-two capacity for branchless wrap-around
        if ((capacity_pow2 == 0) || ((capacity_pow2 & mask_) != 0)) {
            std::size_t p = 1;
            while (p < capacity_pow2) p <<= 1;
            mask_ = p - 1;
            buffer_.resize(p);
        }
    }

    std::size_t capacity() const noexcept { return buffer_.size(); }

    /// Push up to `count` elements from producer thread. Returns number actually pushed.
    std::size_t push(const T* src, std::size_t count) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        const std::size_t free_space = buffer_.size() - (head - tail);
        const std::size_t n = (count < free_space) ? count : free_space;

        for (std::size_t i = 0; i < n; ++i) {
            buffer_[(head + i) & mask_] = src[i];
        }
        head_.store(head + n, std::memory_order_release);
        return n;
    }

    /// Pop up to `count` elements from consumer thread. Returns number actually popped.
    std::size_t pop(T* dst, std::size_t count) noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t head = head_.load(std::memory_order_acquire);
        const std::size_t avail = head - tail;
        const std::size_t n = (count < avail) ? count : avail;

        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = buffer_[(tail + i) & mask_];
        }
        tail_.store(tail + n, std::memory_order_release);
        return n;
    }

    std::size_t available() const noexcept {
        return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_relaxed);
    }

private:
    std::size_t mask_;
    std::vector<T> buffer_;

    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
};

} // namespace aegis::rt
