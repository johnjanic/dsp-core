#pragma once

#include <atomic>
#include <array>
#include <cstddef>

namespace dsp {

/**
 * @brief Lock-free single-producer single-consumer FIFO.
 *
 * Thread-safe queue for communication between producer (e.g., UI thread)
 * and consumer (e.g., audio thread) without locks.
 *
 * @tparam T Element type (must be trivially copyable)
 * @tparam Capacity Maximum number of elements (must be power of 2)
 */
template<typename T, size_t Capacity>
class LockFreeFIFO
{
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

public:
    LockFreeFIFO() = default;

    // Non-copyable
    LockFreeFIFO(const LockFreeFIFO&) = delete;
    LockFreeFIFO& operator=(const LockFreeFIFO&) = delete;

    /**
     * @brief Try to push an element (producer side).
     *
     * @param item The item to push.
     * @return true if successful, false if FIFO is full.
     */
    bool tryPush(const T& item) noexcept
    {
        const size_t currentWrite = writeIndex_.load(std::memory_order_relaxed);
        const size_t nextWrite = (currentWrite + 1) & kMask;

        // Check if full
        if (nextWrite == readIndex_.load(std::memory_order_acquire))
            return false;

        buffer_[currentWrite] = item;
        writeIndex_.store(nextWrite, std::memory_order_release);
        return true;
    }

    /**
     * @brief Try to pop an element (consumer side).
     *
     * @param item Output parameter for the popped item.
     * @return true if successful, false if FIFO is empty.
     */
    bool tryPop(T& item) noexcept
    {
        const size_t currentRead = readIndex_.load(std::memory_order_relaxed);

        // Check if empty
        if (currentRead == writeIndex_.load(std::memory_order_acquire))
            return false;

        item = buffer_[currentRead];
        readIndex_.store((currentRead + 1) & kMask, std::memory_order_release);
        return true;
    }

    /**
     * @brief Check if FIFO is empty.
     */
    [[nodiscard]] bool isEmpty() const noexcept
    {
        return readIndex_.load(std::memory_order_acquire) ==
               writeIndex_.load(std::memory_order_acquire);
    }

    /**
     * @brief Check if FIFO is full.
     */
    [[nodiscard]] bool isFull() const noexcept
    {
        const size_t nextWrite = (writeIndex_.load(std::memory_order_acquire) + 1) & kMask;
        return nextWrite == readIndex_.load(std::memory_order_acquire);
    }

    /**
     * @brief Get approximate number of elements.
     *
     * Note: May be inaccurate if called during concurrent access.
     */
    [[nodiscard]] size_t sizeApprox() const noexcept
    {
        const size_t write = writeIndex_.load(std::memory_order_acquire);
        const size_t read = readIndex_.load(std::memory_order_acquire);
        return (write - read) & kMask;
    }

    /**
     * @brief Get capacity.
     */
    [[nodiscard]] constexpr size_t capacity() const noexcept
    {
        return Capacity - 1;  // One slot reserved for full/empty detection
    }

    /**
     * @brief Clear all elements (single-threaded only).
     */
    void clear() noexcept
    {
        readIndex_.store(0, std::memory_order_relaxed);
        writeIndex_.store(0, std::memory_order_relaxed);
    }

private:
    static constexpr size_t kMask = Capacity - 1;

    std::array<T, Capacity> buffer_{};
    alignas(64) std::atomic<size_t> readIndex_{0};   // Cache line aligned
    alignas(64) std::atomic<size_t> writeIndex_{0};  // Separate cache line
};


/**
 * @brief Triple buffer for single-producer, single-consumer scenarios.
 *
 * Provides a wait-free mechanism for the consumer to always read
 * the most recent complete data without blocking.
 *
 * @tparam T The data type to buffer
 */
template<typename T>
class TripleBuffer
{
public:
    TripleBuffer() = default;

    // Non-copyable
    TripleBuffer(const TripleBuffer&) = delete;
    TripleBuffer& operator=(const TripleBuffer&) = delete;

    /**
     * @brief Write new data (producer side).
     *
     * Writes to the back buffer and swaps it to become available.
     *
     * @param data The data to write.
     */
    void write(const T& data) noexcept
    {
        // Write to back buffer
        size_t back = backIndex_.load(std::memory_order_relaxed);
        buffers_[back] = data;

        // Swap back with middle
        size_t middle = middleIndex_.load(std::memory_order_relaxed);
        backIndex_.store(middle, std::memory_order_relaxed);
        middleIndex_.store(back, std::memory_order_release);

        // Mark new data available
        newDataAvailable_.store(true, std::memory_order_release);
    }

    /**
     * @brief Read latest data (consumer side).
     *
     * Returns reference to the most recently completed write.
     * Wait-free - always returns immediately.
     *
     * @return Reference to the front buffer data.
     */
    const T& read() noexcept
    {
        // Check if new data is available
        if (newDataAvailable_.exchange(false, std::memory_order_acquire))
        {
            // Swap front with middle to get new data
            size_t front = frontIndex_.load(std::memory_order_relaxed);
            size_t middle = middleIndex_.load(std::memory_order_relaxed);
            frontIndex_.store(middle, std::memory_order_relaxed);
            middleIndex_.store(front, std::memory_order_relaxed);
        }

        return buffers_[frontIndex_.load(std::memory_order_relaxed)];
    }

    /**
     * @brief Check if new data is available without consuming it.
     */
    [[nodiscard]] bool hasNewData() const noexcept
    {
        return newDataAvailable_.load(std::memory_order_acquire);
    }

    /**
     * @brief Direct access to write buffer for in-place construction.
     */
    T& getWriteBuffer() noexcept
    {
        return buffers_[backIndex_.load(std::memory_order_relaxed)];
    }

    /**
     * @brief Publish the write buffer (call after modifying via getWriteBuffer).
     */
    void publish() noexcept
    {
        size_t back = backIndex_.load(std::memory_order_relaxed);
        size_t middle = middleIndex_.load(std::memory_order_relaxed);
        backIndex_.store(middle, std::memory_order_relaxed);
        middleIndex_.store(back, std::memory_order_release);
        newDataAvailable_.store(true, std::memory_order_release);
    }

private:
    std::array<T, 3> buffers_{};

    // Buffer indices - use separate cache lines to avoid false sharing
    alignas(64) std::atomic<size_t> frontIndex_{0};
    alignas(64) std::atomic<size_t> middleIndex_{1};
    alignas(64) std::atomic<size_t> backIndex_{2};
    alignas(64) std::atomic<bool> newDataAvailable_{false};
};

/**
 * @brief Lock-free FIFO coordination for circular buffers.
 *
 * Provides index management for a single-producer single-consumer circular buffer
 * where the actual data storage is managed externally. This is a drop-in replacement
 * for juce::AbstractFifo.
 *
 * Thread safety:
 * - prepareToWrite/finishedWrite: Called only from producer thread
 * - prepareToRead/finishedRead: Called only from consumer thread
 * - getNumReady: Can be called from either thread (returns approximate count)
 */
class AbstractFIFO
{
public:
    /**
     * @brief Construct with buffer size.
     * @param size Total size of the external buffer
     */
    explicit AbstractFIFO(int size) noexcept
        : bufferSize_(size) {}

    /**
     * @brief Reset to empty state.
     * Only safe to call when no concurrent operations are in progress.
     */
    void reset() noexcept
    {
        readPos_.store(0, std::memory_order_relaxed);
        writePos_.store(0, std::memory_order_relaxed);
    }

    /**
     * @brief Get number of items ready to read.
     */
    [[nodiscard]] int getNumReady() const noexcept
    {
        const int write = writePos_.load(std::memory_order_acquire);
        const int read = readPos_.load(std::memory_order_acquire);
        return (write - read + bufferSize_) % bufferSize_;
    }

    /**
     * @brief Get available space for writing.
     */
    [[nodiscard]] int getFreeSpace() const noexcept
    {
        return bufferSize_ - 1 - getNumReady();
    }

    /**
     * @brief Prepare to write numToWrite items.
     *
     * Returns two contiguous regions that may wrap around the buffer end.
     * Call finishedWrite() after writing to make data available to reader.
     *
     * @param numToWrite Number of items to write
     * @param start1 Output: Start index of first region
     * @param size1 Output: Size of first region
     * @param start2 Output: Start index of second region (0 if no wrap)
     * @param size2 Output: Size of second region (0 if no wrap)
     */
    void prepareToWrite(int numToWrite, int& start1, int& size1, int& start2, int& size2) noexcept
    {
        const int write = writePos_.load(std::memory_order_relaxed);
        const int available = getFreeSpace();
        const int toWrite = std::min(numToWrite, available);

        if (toWrite == 0)
        {
            start1 = 0; size1 = 0;
            start2 = 0; size2 = 0;
            return;
        }

        start1 = write;
        const int firstPart = std::min(toWrite, bufferSize_ - write);
        size1 = firstPart;

        const int remaining = toWrite - firstPart;
        if (remaining > 0)
        {
            start2 = 0;
            size2 = remaining;
        }
        else
        {
            start2 = 0;
            size2 = 0;
        }
    }

    /**
     * @brief Complete a write operation.
     * @param numWritten Number of items actually written
     */
    void finishedWrite(int numWritten) noexcept
    {
        const int write = writePos_.load(std::memory_order_relaxed);
        writePos_.store((write + numWritten) % bufferSize_, std::memory_order_release);
    }

    /**
     * @brief Prepare to read up to numWanted items.
     *
     * Returns two contiguous regions that may wrap around the buffer end.
     * Call finishedRead() after reading to free space for writer.
     *
     * @param numWanted Number of items to read
     * @param start1 Output: Start index of first region
     * @param size1 Output: Size of first region
     * @param start2 Output: Start index of second region (0 if no wrap)
     * @param size2 Output: Size of second region (0 if no wrap)
     */
    void prepareToRead(int numWanted, int& start1, int& size1, int& start2, int& size2) noexcept
    {
        const int read = readPos_.load(std::memory_order_relaxed);
        const int ready = getNumReady();
        const int toRead = std::min(numWanted, ready);

        if (toRead == 0)
        {
            start1 = 0; size1 = 0;
            start2 = 0; size2 = 0;
            return;
        }

        start1 = read;
        const int firstPart = std::min(toRead, bufferSize_ - read);
        size1 = firstPart;

        const int remaining = toRead - firstPart;
        if (remaining > 0)
        {
            start2 = 0;
            size2 = remaining;
        }
        else
        {
            start2 = 0;
            size2 = 0;
        }
    }

    /**
     * @brief Complete a read operation.
     * @param numRead Number of items actually read
     */
    void finishedRead(int numRead) noexcept
    {
        const int read = readPos_.load(std::memory_order_relaxed);
        readPos_.store((read + numRead) % bufferSize_, std::memory_order_release);
    }

private:
    const int bufferSize_;
    alignas(64) std::atomic<int> readPos_{0};
    alignas(64) std::atomic<int> writePos_{0};
};

} // namespace dsp
