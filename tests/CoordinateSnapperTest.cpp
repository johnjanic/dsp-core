#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>

using namespace dsp_core::Services;

// ============================================================================
// snapValue() Tests
// ============================================================================

TEST(CoordinateSnapperTest, SnapValue_PositiveValues) {
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.47, 0.1), 0.5);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.44, 0.1), 0.4);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.45, 0.1), 0.5); // Midpoint rounds up
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.51, 0.1), 0.5);
}

TEST(CoordinateSnapperTest, SnapValue_NegativeValues) {
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(-0.47, 0.1), -0.5);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(-0.44, 0.1), -0.4);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(-0.45, 0.1), -0.5); // Midpoint rounds to even
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(-0.51, 0.1), -0.5);
}

TEST(CoordinateSnapperTest, SnapValue_FractionalGridStep) {
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.123, 0.05), 0.10);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.126, 0.05), 0.15);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.137, 0.05), 0.15);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.149, 0.05), 0.15);
}

TEST(CoordinateSnapperTest, SnapValue_ExactGridLine) {
    // Values already on grid lines should remain unchanged
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.5, 0.1), 0.5);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(-0.3, 0.1), -0.3);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.0, 0.1), 0.0);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(1.0, 0.1), 1.0);
}

TEST(CoordinateSnapperTest, SnapValue_InvalidGridStep) {
    // Should return original value for invalid grid steps
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.47, 0.0), 0.47);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.47, -0.1), 0.47);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(-0.23, 0.0), -0.23);
}

TEST(CoordinateSnapperTest, SnapValue_LargeGridStep) {
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.47, 0.5), 0.5);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.24, 0.5), 0.0);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.76, 0.5), 1.0);
}

TEST(CoordinateSnapperTest, SnapValue_SmallGridStep) {
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.123456, 0.01), 0.12);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.126789, 0.01), 0.13);
}

// ============================================================================
// snapPoint() Tests
// ============================================================================

TEST(CoordinateSnapperTest, SnapPoint_BothAxes) {
    auto result = CoordinateSnapper::snapPoint({0.47, 0.23}, 0.1, true, true);
    EXPECT_DOUBLE_EQ(result.x, 0.5);
    EXPECT_DOUBLE_EQ(result.y, 0.2);
}

TEST(CoordinateSnapperTest, SnapPoint_XOnly) {
    auto result = CoordinateSnapper::snapPoint({0.47, 0.23}, 0.1, true, false);
    EXPECT_DOUBLE_EQ(result.x, 0.5);
    EXPECT_DOUBLE_EQ(result.y, 0.23);
}

TEST(CoordinateSnapperTest, SnapPoint_YOnly) {
    auto result = CoordinateSnapper::snapPoint({0.47, 0.23}, 0.1, false, true);
    EXPECT_DOUBLE_EQ(result.x, 0.47);
    EXPECT_DOUBLE_EQ(result.y, 0.2);
}

TEST(CoordinateSnapperTest, SnapPoint_NeitherAxis) {
    auto result = CoordinateSnapper::snapPoint({0.47, 0.23}, 0.1, false, false);
    EXPECT_DOUBLE_EQ(result.x, 0.47);
    EXPECT_DOUBLE_EQ(result.y, 0.23);
}

TEST(CoordinateSnapperTest, SnapPoint_NegativeCoordinates) {
    auto result = CoordinateSnapper::snapPoint({-0.47, -0.23}, 0.1, true, true);
    EXPECT_DOUBLE_EQ(result.x, -0.5);
    EXPECT_DOUBLE_EQ(result.y, -0.2);
}

TEST(CoordinateSnapperTest, SnapPoint_MixedSigns) {
    auto result = CoordinateSnapper::snapPoint({0.47, -0.23}, 0.1, true, true);
    EXPECT_DOUBLE_EQ(result.x, 0.5);
    EXPECT_DOUBLE_EQ(result.y, -0.2);
}

TEST(CoordinateSnapperTest, SnapPoint_DefaultParameters) {
    // Test default parameters (snapX=true, snapY=true)
    auto result = CoordinateSnapper::snapPoint({0.47, 0.23}, 0.1);
    EXPECT_DOUBLE_EQ(result.x, 0.5);
    EXPECT_DOUBLE_EQ(result.y, 0.2);
}

// ============================================================================
// isNearGridLine() Tests
// ============================================================================

TEST(CoordinateSnapperTest, IsNearGridLine_WithinThreshold) {
    EXPECT_TRUE(CoordinateSnapper::isNearGridLine(0.48, 0.1, 0.05));
    EXPECT_TRUE(CoordinateSnapper::isNearGridLine(0.52, 0.1, 0.05));
    EXPECT_TRUE(CoordinateSnapper::isNearGridLine(0.50, 0.1, 0.05)); // Exactly on line
}

TEST(CoordinateSnapperTest, IsNearGridLine_OutsideThreshold) {
    EXPECT_FALSE(CoordinateSnapper::isNearGridLine(0.36, 0.1, 0.03)); // nearest=0.4, distance=0.04 > 0.03
    EXPECT_FALSE(CoordinateSnapper::isNearGridLine(0.64, 0.1, 0.03)); // nearest=0.6, distance=0.04 > 0.03
    EXPECT_FALSE(CoordinateSnapper::isNearGridLine(0.35, 0.1, 0.04)); // nearest=0.4, distance=0.05 > 0.04
}

TEST(CoordinateSnapperTest, IsNearGridLine_ExactlyOnBoundary) {
    // Exactly at threshold distance - should be true (<=)
    EXPECT_TRUE(CoordinateSnapper::isNearGridLine(0.45, 0.1, 0.05)); // nearest=0.5, distance=0.05
    EXPECT_TRUE(CoordinateSnapper::isNearGridLine(0.35, 0.1, 0.05)); // nearest=0.4, distance=0.05
}

TEST(CoordinateSnapperTest, IsNearGridLine_NegativeValues) {
    EXPECT_TRUE(CoordinateSnapper::isNearGridLine(-0.48, 0.1, 0.05));  // nearest=-0.5, distance=0.02
    EXPECT_TRUE(CoordinateSnapper::isNearGridLine(-0.52, 0.1, 0.05));  // nearest=-0.5, distance=0.02
    EXPECT_FALSE(CoordinateSnapper::isNearGridLine(-0.36, 0.1, 0.03)); // nearest=-0.4, distance=0.04 > 0.03
}

TEST(CoordinateSnapperTest, IsNearGridLine_InvalidGridStep) {
    EXPECT_FALSE(CoordinateSnapper::isNearGridLine(0.48, 0.0, 0.05));
    EXPECT_FALSE(CoordinateSnapper::isNearGridLine(0.48, -0.1, 0.05));
}

TEST(CoordinateSnapperTest, IsNearGridLine_ZeroThreshold) {
    // Zero threshold means only exact matches
    EXPECT_TRUE(CoordinateSnapper::isNearGridLine(0.5, 0.1, 0.0));
    EXPECT_FALSE(CoordinateSnapper::isNearGridLine(0.51, 0.1, 0.0));
}

// ============================================================================
// nearestGridLine() Tests
// ============================================================================

TEST(CoordinateSnapperTest, NearestGridLine_PositiveValues) {
    EXPECT_DOUBLE_EQ(CoordinateSnapper::nearestGridLine(0.47, 0.1), 0.5);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::nearestGridLine(0.44, 0.1), 0.4);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::nearestGridLine(0.45, 0.1), 0.5);
}

TEST(CoordinateSnapperTest, NearestGridLine_NegativeValues) {
    EXPECT_DOUBLE_EQ(CoordinateSnapper::nearestGridLine(-0.47, 0.1), -0.5);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::nearestGridLine(-0.44, 0.1), -0.4);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::nearestGridLine(-0.45, 0.1), -0.5);
}

TEST(CoordinateSnapperTest, NearestGridLine_ExactlyOnLine) {
    EXPECT_DOUBLE_EQ(CoordinateSnapper::nearestGridLine(0.5, 0.1), 0.5);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::nearestGridLine(-0.3, 0.1), -0.3);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::nearestGridLine(0.0, 0.1), 0.0);
}

TEST(CoordinateSnapperTest, NearestGridLine_InvalidGridStep) {
    EXPECT_DOUBLE_EQ(CoordinateSnapper::nearestGridLine(0.47, 0.0), 0.47);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::nearestGridLine(0.47, -0.1), 0.47);
}

TEST(CoordinateSnapperTest, NearestGridLine_FractionalGridStep) {
    EXPECT_DOUBLE_EQ(CoordinateSnapper::nearestGridLine(0.123, 0.05), 0.10);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::nearestGridLine(0.126, 0.05), 0.15);
}

// ============================================================================
// Edge Cases and Symmetry Tests
// ============================================================================

TEST(CoordinateSnapperTest, Symmetry_PositiveAndNegative) {
    // Verify symmetric behavior for positive and negative values
    double gridStep = 0.1;
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.47, gridStep), -CoordinateSnapper::snapValue(-0.47, gridStep));
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.23, gridStep), -CoordinateSnapper::snapValue(-0.23, gridStep));
}

TEST(CoordinateSnapperTest, ZeroValue) {
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.0, 0.1), 0.0);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::nearestGridLine(0.0, 0.1), 0.0);
    EXPECT_TRUE(CoordinateSnapper::isNearGridLine(0.0, 0.1, 0.01));
}

TEST(CoordinateSnapperTest, VerySmallValues) {
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(0.001, 0.1), 0.0);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(-0.001, 0.1), 0.0);
}

TEST(CoordinateSnapperTest, LargeValues) {
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(10.47, 0.1), 10.5);
    EXPECT_DOUBLE_EQ(CoordinateSnapper::snapValue(-10.47, 0.1), -10.5);
}
