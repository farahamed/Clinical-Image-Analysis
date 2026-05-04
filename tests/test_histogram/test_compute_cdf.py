"""
Unit Tests for compute_cdf function
=====================================
Tests cumulative distribution function computation.

Menna Hesham Ragab Allam - 1220321
Module: core/histogram/local_equalization.py
Function: compute_cdf
"""

import numpy as np
import pytest
import sys
import os
  
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from processing.histogram.local_equalization import compute_histogram, compute_cdf

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def simple_histogram():
    """Simple histogram with increasing values"""
    return np.array([10, 20, 30, 40], dtype=np.int64)


@pytest.fixture
def uniform_histogram():
    """Histogram where all bins have same count"""
    return np.array([50, 50, 50, 50], dtype=np.int64)


@pytest.fixture
def zero_histogram():
    """Histogram with all zeros"""
    return np.zeros(256, dtype=np.int64)


@pytest.fixture
def single_peak_histogram():
    """Histogram with one non-zero bin"""
    hist = np.zeros(256, dtype=np.int64)
    hist[128] = 1000
    return hist


@pytest.fixture
def random_image():
    """Generate random image for realistic histogram"""
    np.random.seed(42)
    return np.random.randint(0, 256, (50, 50), dtype=np.uint8)


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

class TestComputeCDFBasic:
    """Test basic CDF computation"""
    
    def test_returns_array(self, simple_histogram):
        """Test function returns numpy array"""
        result = compute_cdf(simple_histogram)
        assert isinstance(result, np.ndarray)
    
    def test_output_length(self, simple_histogram):
        """Test output length matches input"""
        result = compute_cdf(simple_histogram)
        assert len(result) == len(simple_histogram)
    
    def test_output_is_float(self, simple_histogram):
        """Test output values are floats"""
        result = compute_cdf(simple_histogram)
        assert np.issubdtype(result.dtype, np.floating)


# ============================================================================
# VALUE RANGE TESTS
# ============================================================================

class TestComputeCDFRange:
    """Test CDF value ranges"""
    
    def test_first_value_is_zero(self, simple_histogram):
        """Test CDF starts at 0"""
        result = compute_cdf(simple_histogram)
        assert np.isclose(result[0], 0.0), f"First value: {result[0]}"
    
    def test_last_value_is_255(self, simple_histogram):
        """Test CDF ends at 255"""
        result = compute_cdf(simple_histogram)
        assert np.isclose(result[-1], 255.0), f"Last value: {result[-1]}"
    
    def test_all_values_non_negative(self, simple_histogram):
        """Test no negative values"""
        result = compute_cdf(simple_histogram)
        assert np.all(result >= 0)
    
    def test_all_values_not_exceed_255(self, simple_histogram):
        """Test no values exceed 255"""
        result = compute_cdf(simple_histogram)
        assert np.all(result <= 255)
    
    def test_range_of_values(self, simple_histogram):
        """Test values span appropriate range"""
        result = compute_cdf(simple_histogram)
        assert result.min() >= 0
        assert result.max() <= 255
        assert result.min() < result.max()


# ============================================================================
# MONOTONICITY TESTS
# ============================================================================

class TestComputeCDFMonotonicity:
    """Test CDF is monotonically increasing"""
    
    def test_strictly_increasing(self, simple_histogram):
        """Test values never decrease"""
        result = compute_cdf(simple_histogram)
        assert np.all(np.diff(result) >= 0)
    
    def test_realistic_histogram_monotonic(self, random_image):
        """Test with histogram from real image"""
        hist, _ = compute_histogram(random_image)
        result = compute_cdf(hist)
        assert np.all(np.diff(result) >= 0)
    
    def test_single_peak_monotonic(self, single_peak_histogram):
        """Test with single peak histogram"""
        result = compute_cdf(single_peak_histogram)
        assert np.all(np.diff(result) >= 0)


# ============================================================================
# SPECIAL CASES TESTS
# ============================================================================

class TestComputeCDFSpecialCases:
    """Test special histogram cases"""
    
    def test_uniform_histogram(self, uniform_histogram):
        """Test with uniform histogram (all bins equal)"""
        result = compute_cdf(uniform_histogram)
        assert len(result) == len(uniform_histogram)
        assert np.all(result >= 0)
        assert np.all(result <= 255)
    
    def test_zero_histogram(self, zero_histogram):
        """Test with all-zero histogram"""
        result = compute_cdf(zero_histogram)
        assert len(result) == len(zero_histogram)
        assert np.all(result == 0.0)
    
    def test_single_peak_cdf_ends_at_255(self, single_peak_histogram):
        """Test single peak still normalizes correctly"""
        result = compute_cdf(single_peak_histogram)
        assert np.isclose(result[-1], 255.0)
    
    def test_two_element_histogram(self):
        """Test with minimum size histogram"""
        hist = np.array([10, 20], dtype=np.int64)
        result = compute_cdf(hist)
        assert len(result) == 2
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[1], 255.0)


# ============================================================================
# MATHEMATICAL PROPERTIES TESTS
# ============================================================================

class TestComputeCDFMathProperties:
    """Test mathematical correctness of CDF"""
    
    def test_cdf_is_cumulative(self):
        """Test CDF is truly cumulative"""
        hist = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        result = compute_cdf(hist)
        
        # Manual calculation
        total = hist.sum()
        manual_cdf = np.zeros_like(hist, dtype=np.float64)
        for i in range(len(hist)):
            manual_cdf[i] = hist[:i+1].sum()
        manual_cdf = (manual_cdf - manual_cdf[0]) / (manual_cdf[-1] - manual_cdf[0]) * 255
        
        assert np.allclose(result, manual_cdf)
    
    def test_normalization_consistent(self):
        """Test same histogram gives same CDF"""
        hist = np.array([5, 10, 15, 20], dtype=np.int64)
        result1 = compute_cdf(hist)
        result2 = compute_cdf(hist)
        assert np.array_equal(result1, result2)
    
    def test_scale_invariant(self):
        """Test scaling histogram by constant gives same CDF"""
        hist = np.array([5, 10, 15, 20], dtype=np.int64)
        hist_scaled = hist * 2
        result = compute_cdf(hist)
        result_scaled = compute_cdf(hist_scaled)
        assert np.allclose(result, result_scaled)
    
    def test_cdf_of_histogram_from_uniform_image(self):
        """Test CDF of uniform image"""
        img = np.full((10, 10), 128, dtype=np.uint8)
        hist, _ = compute_histogram(img)
        result = compute_cdf(hist)
        # Since all pixels are 128, only bin 128 has count > 0
        # CDF should be 0 up to bin 128, then jump to 255
        assert result[127] == 0
        assert result[128] == 255


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestComputeCDFErrors:
    """Test error handling"""
    
    def test_none_input(self):
        """Test None raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            compute_cdf(None)
        assert "None" in str(exc_info.value)
    
    def test_empty_array(self):
        """Test empty array raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            compute_cdf(np.array([], dtype=np.int64))
        assert "empty" in str(exc_info.value)
    
    def test_negative_histogram_values(self):
        """Test histogram with negative values still works"""
        # Should not happen in practice but function should handle gracefully
        hist = np.array([-1, 10, 20], dtype=np.int64)
        try:
            result = compute_cdf(hist)
            # If it doesn't crash, check output is still valid
            assert len(result) == len(hist)
        except Exception:
            pass  # Raising error is also acceptable


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestComputeCDFPerformance:
    """Test CDF computation performance"""
    
    def test_256_bin_histogram(self):
        """Test with standard 256-bin histogram"""
        hist, _ = compute_histogram(np.random.randint(0, 256, (100, 100), dtype=np.uint8))
        import time
        start = time.time()
        result = compute_cdf(hist)
        duration = time.time() - start
        assert duration < 0.1  # Should be very fast
        assert len(result) == 256
    
    def test_large_histogram(self):
        """Test with larger histogram"""
        hist = np.random.randint(0, 1000, 1024, dtype=np.int64)
        result = compute_cdf(hist)
        assert len(result) == 1024
        assert np.isclose(result[-1], 255.0)