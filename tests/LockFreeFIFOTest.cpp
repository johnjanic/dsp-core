#include <gtest/gtest.h>
#include "dsp_core/Source/primitives/LockFreeFIFO.h"
#include <thread>
#include <vector>

using namespace dsp;

class LockFreeFIFOTest : public ::testing::Test {};

// =============================================================================
// Basic Functionality
// =============================================================================

TEST_F(LockFreeFIFOTest, NewFIFO_IsEmpty)
{
    LockFreeFIFO<int, 16> fifo;
    EXPECT_TRUE(fifo.isEmpty());
    EXPECT_FALSE(fifo.isFull());
    EXPECT_EQ(fifo.sizeApprox(), 0u);
}

TEST_F(LockFreeFIFOTest, Push_IncreasesSize)
{
    LockFreeFIFO<int, 16> fifo;
    EXPECT_TRUE(fifo.tryPush(42));
    EXPECT_FALSE(fifo.isEmpty());
    EXPECT_EQ(fifo.sizeApprox(), 1u);
}

TEST_F(LockFreeFIFOTest, Pop_DecreasesSize)
{
    LockFreeFIFO<int, 16> fifo;
    fifo.tryPush(42);

    int value;
    EXPECT_TRUE(fifo.tryPop(value));
    EXPECT_EQ(value, 42);
    EXPECT_TRUE(fifo.isEmpty());
}

TEST_F(LockFreeFIFOTest, FIFO_Order)
{
    LockFreeFIFO<int, 16> fifo;

    fifo.tryPush(1);
    fifo.tryPush(2);
    fifo.tryPush(3);

    int value;
    EXPECT_TRUE(fifo.tryPop(value)); EXPECT_EQ(value, 1);
    EXPECT_TRUE(fifo.tryPop(value)); EXPECT_EQ(value, 2);
    EXPECT_TRUE(fifo.tryPop(value)); EXPECT_EQ(value, 3);
}

TEST_F(LockFreeFIFOTest, Full_RejectsPush)
{
    LockFreeFIFO<int, 4> fifo;  // Capacity 3 (one slot reserved)

    EXPECT_TRUE(fifo.tryPush(1));
    EXPECT_TRUE(fifo.tryPush(2));
    EXPECT_TRUE(fifo.tryPush(3));
    EXPECT_FALSE(fifo.tryPush(4));  // Should fail - full

    EXPECT_TRUE(fifo.isFull());
}

TEST_F(LockFreeFIFOTest, Empty_RejectsPop)
{
    LockFreeFIFO<int, 16> fifo;
    int value;
    EXPECT_FALSE(fifo.tryPop(value));
}

TEST_F(LockFreeFIFOTest, Capacity_ReturnsCorrectValue)
{
    LockFreeFIFO<int, 16> fifo;
    EXPECT_EQ(fifo.capacity(), 15u);  // One slot reserved
}

TEST_F(LockFreeFIFOTest, Clear_ResetsToEmpty)
{
    LockFreeFIFO<int, 16> fifo;
    fifo.tryPush(1);
    fifo.tryPush(2);
    fifo.tryPush(3);

    fifo.clear();

    EXPECT_TRUE(fifo.isEmpty());
    EXPECT_EQ(fifo.sizeApprox(), 0u);
}

TEST_F(LockFreeFIFOTest, Wraparound_WorksCorrectly)
{
    LockFreeFIFO<int, 4> fifo;  // Small buffer to force wraparound

    // Fill and empty multiple times to test wraparound
    for (int round = 0; round < 5; ++round)
    {
        EXPECT_TRUE(fifo.tryPush(round * 10 + 1));
        EXPECT_TRUE(fifo.tryPush(round * 10 + 2));
        EXPECT_TRUE(fifo.tryPush(round * 10 + 3));

        int value;
        EXPECT_TRUE(fifo.tryPop(value)); EXPECT_EQ(value, round * 10 + 1);
        EXPECT_TRUE(fifo.tryPop(value)); EXPECT_EQ(value, round * 10 + 2);
        EXPECT_TRUE(fifo.tryPop(value)); EXPECT_EQ(value, round * 10 + 3);
    }
}

// =============================================================================
// Multi-threaded Tests
// =============================================================================

TEST_F(LockFreeFIFOTest, ConcurrentPushPop_NoLostData)
{
    LockFreeFIFO<int, 1024> fifo;
    constexpr int kCount = 10000;

    std::atomic<int> producedSum{0};
    std::atomic<int> consumedSum{0};

    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < kCount; ++i)
        {
            while (!fifo.tryPush(i))
            {
                std::this_thread::yield();
            }
            producedSum += i;
        }
    });

    // Consumer thread
    std::thread consumer([&]() {
        int consumed = 0;
        while (consumed < kCount)
        {
            int value;
            if (fifo.tryPop(value))
            {
                consumedSum += value;
                ++consumed;
            }
            else
            {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(producedSum.load(), consumedSum.load());
}

TEST_F(LockFreeFIFOTest, ConcurrentPushPop_PreservesOrder)
{
    LockFreeFIFO<int, 256> fifo;
    constexpr int kCount = 5000;

    std::atomic<bool> orderViolation{false};

    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < kCount; ++i)
        {
            while (!fifo.tryPush(i))
            {
                std::this_thread::yield();
            }
        }
    });

    // Consumer thread
    std::thread consumer([&]() {
        int expected = 0;
        while (expected < kCount)
        {
            int value;
            if (fifo.tryPop(value))
            {
                if (value != expected)
                {
                    orderViolation = true;
                }
                ++expected;
            }
            else
            {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_FALSE(orderViolation.load());
}

// =============================================================================
// Triple Buffer Tests
// =============================================================================

TEST_F(LockFreeFIFOTest, TripleBuffer_ReadLatestData)
{
    TripleBuffer<int> tb;

    tb.write(1);
    tb.write(2);
    tb.write(3);

    // Should get latest (3)
    EXPECT_EQ(tb.read(), 3);
}

TEST_F(LockFreeFIFOTest, TripleBuffer_ReadReturnsLastValue)
{
    TripleBuffer<int> tb;

    tb.write(42);
    EXPECT_EQ(tb.read(), 42);
    EXPECT_EQ(tb.read(), 42);  // Same value again
    EXPECT_EQ(tb.read(), 42);  // And again
}

TEST_F(LockFreeFIFOTest, TripleBuffer_HasNewData)
{
    TripleBuffer<int> tb;

    EXPECT_FALSE(tb.hasNewData());

    tb.write(1);
    EXPECT_TRUE(tb.hasNewData());

    tb.read();  // Consumes the "new" flag
    EXPECT_FALSE(tb.hasNewData());
}

TEST_F(LockFreeFIFOTest, TripleBuffer_GetWriteBufferAndPublish)
{
    TripleBuffer<int> tb;

    tb.getWriteBuffer() = 100;
    tb.publish();

    EXPECT_EQ(tb.read(), 100);
}

TEST_F(LockFreeFIFOTest, TripleBuffer_MultipleWritesBeforeRead)
{
    TripleBuffer<int> tb;

    tb.write(1);
    tb.write(2);
    tb.write(3);
    tb.write(4);
    tb.write(5);

    // Should always get the latest value
    EXPECT_EQ(tb.read(), 5);
}

TEST_F(LockFreeFIFOTest, TripleBuffer_ConcurrentReadWrite)
{
    TripleBuffer<int> tb;
    constexpr int kIterations = 100000;

    std::atomic<bool> running{true};
    std::atomic<int> lastRead{0};

    // Writer thread
    std::thread writer([&]() {
        for (int i = 1; i <= kIterations; ++i)
        {
            tb.write(i);
        }
        running = false;
    });

    // Reader thread
    std::thread reader([&]() {
        int prev = 0;
        while (running || tb.hasNewData())
        {
            int value = tb.read();
            // Values should be monotonically increasing (no torn reads)
            EXPECT_GE(value, prev);
            prev = value;
            lastRead = value;
        }
    });

    writer.join();
    reader.join();

    // Should have read the final value eventually
    EXPECT_EQ(lastRead.load(), kIterations);
}

TEST_F(LockFreeFIFOTest, TripleBuffer_WithStruct)
{
    struct TestData
    {
        int a;
        float b;
        double c;
    };

    TripleBuffer<TestData> tb;

    tb.write({1, 2.0f, 3.0});

    auto& data = tb.read();
    EXPECT_EQ(data.a, 1);
    EXPECT_FLOAT_EQ(data.b, 2.0f);
    EXPECT_DOUBLE_EQ(data.c, 3.0);
}
