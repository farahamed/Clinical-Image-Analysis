"""
Unit Tests for global_histogram_equalization function
======================================================
Tests global histogram equalization from scratch.

Menna Hesham Ragab Allam - 1220321
Module: processing/histogram/local_equalization.py
Function: global_histogram_equalization
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from processing.histogram.local_equalization import (
    global_histogram_equalization,
    compute_histogram
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def simple_image():
    """Small test image"""
    return np.array([[50, 100], [150, 200]], dtype=np.uint8)


@pytest.fixture
def uniform_image():
    """All pixels same value"""
    return np.full((20, 20), 128, dtype=np.uint8)


@pytest.fixture
def dark_image():
    """Predominantly dark image"""
    return np.full((50, 50), 30, dtype=np.uint8)


@pytest.fixture
def bright_image():
    """Predominantly bright image"""
    return np.full((50, 50), 220, dtype=np.uint8)


@pytest.fixture
def low_contrast_image():
    """Image with narrow intensity range"""
    img = np.full((50, 50), 100, dtype=np.uint8)
    img[10:40, 10:40] = 120
    return img


@pytest.fixture
def gradient_image():
    """Horizontal gradient 0-255"""
    return np.tile(np.arange(256, dtype=np.uint8), (256, 1))


@pytest.fixture
def random_image():
    """Random test image"""
    np.random.seed(42)
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8)


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

class TestGlobalHEBasic:
    """Basic functionality tests"""
    
    def test_returns_array(self, simple_image):
        """Test returns numpy array"""
        result = global_histogram_equalization(simple_image)
        assert isinstance(result, np.ndarray)
    
    def test_same_shape(self, simple_image):
        """Test output shape matches input"""
        result = global_histogram_equalization(simple_image)
        assert result.shape == simple_image.shape
    
    def test_same_shape_random(self, random_image):
        """Test shape preserved with random image"""
        result = global_histogram_equalization(random_image)
        assert result.shape == random_image.shape
    
    def test_output_is_uint8(self, simple_image):
        """Test output dtype is uint8"""
        result = global_histogram_equalization(simple_image)
        assert result.dtype == np.uint8
    
    def test_output_is_not_float(self, simple_image):
        """Test output is integer type"""
        result = global_histogram_equalization(simple_image)
        assert np.issubdtype(result.dtype, np.integer)


# ============================================================================
# VALUE RANGE TESTS
# ============================================================================

class TestGlobalHEValueRange:
    """Test output value ranges"""
    
    def test_no_values_below_zero(self, random_image):
        """Test all values >= 0"""
        result = global_histogram_equalization(random_image)
        assert result.min() >= 0
    
    def test_no_values_above_255(self, random_image):
        """Test all values <= 255"""
        result = global_histogram_equalization(random_image)
        assert result.max() <= 255
    
    def test_dark_image_gets_brighter(self, dark_image):
        """Test dark images become brighter"""
        result = global_histogram_equalization(dark_image)
        assert result.mean() > dark_image.mean(), \
            f"Original mean: {dark_image.mean()}, Equalized mean: {result.mean()}"
    
    def test_bright_image_gets_balanced(self, bright_image):
        """Test bright images have values spread out"""
        result = global_histogram_equalization(bright_image)
        # Should have both dark and bright values
        assert result.min() < bright_image.min() or result.max() > bright_image.max()
    
    def test_min_value_is_zero_after_he(self, random_image):
        """Test minimum value becomes 0 after equalization"""
        result = global_histogram_equalization(random_image)
        assert result.min() == 0, f"Expected min 0, got {result.min()}"
    
    def test_max_value_is_255_after_he(self, random_image):
        """Test maximum value becomes 255 after equalization"""
        result = global_histogram_equalization(random_image)
        assert result.max() == 255, f"Expected max 255, got {result.max()}"


# ============================================================================
# CONTRAST ENHANCEMENT TESTS
# ============================================================================

class TestGlobalHEContrast:
    """Test contrast enhancement properties"""
    
    def test_histogram_spread_range(self, low_contrast_image):
        """Test output histogram spans wider range than input histogram"""
        hist_orig, _ = compute_histogram(low_contrast_image)
        result = global_histogram_equalization(low_contrast_image)
        hist_eq, _ = compute_histogram(result)
        
        # Find the range of non-zero bins
        nonzero_orig = np.where(hist_orig > 0)[0]
        nonzero_eq = np.where(hist_eq > 0)[0]
        
        range_orig = nonzero_orig[-1] - nonzero_orig[0] if len(nonzero_orig) > 0 else 0
        range_eq = nonzero_eq[-1] - nonzero_eq[0] if len(nonzero_eq) > 0 else 0
        
        # After equalization, non-zero bins should span a wider range
        assert range_eq >= range_orig, \
            f"Original bin range: {range_orig}, Equalized bin range: {range_eq}"
    
    def test_full_range_after_equalization(self, random_image):
        """Test equalized image uses full 0-255 range"""
        result = global_histogram_equalization(random_image)
        
        # Medical imaging: output should span full dynamic range
        assert result.min() == 0, f"Min value should be 0, got {result.min()}"
        assert result.max() == 255, f"Max value should be 255, got {result.max()}"
    
    def test_contrast_improvement(self, low_contrast_image):
        """Test standard deviation increases for low-contrast images"""
        result = global_histogram_equalization(low_contrast_image)
        std_orig = low_contrast_image.std()
        std_eq = result.std()
        assert std_eq > std_orig, \
            f"Original std: {std_orig:.2f}, Equalized std: {std_eq:.2f}"
    
    def test_dynamic_range_increases(self, random_image):
        """Test output has wider range than input"""
        result = global_histogram_equalization(random_image)
        range_orig = random_image.max() - random_image.min()
        range_eq = result.max() - result.min()
        assert range_eq >= range_orig, \
            f"Original range: {range_orig}, Equalized range: {range_eq}"
    
    def test_preserves_lesion_visibility(self):
        """Test that equalization preserves clinically visible lesions"""
        # Create a synthetic medical image with moderate contrast lesions
        img = np.full((100, 100), 80, dtype=np.uint8)
        img[45:55, 45:55] = 120  # Lesion with 40 intensity difference
        img[20:30, 70:80] = 150  # Brighter lesion
        
        result = global_histogram_equalization(img)
        
        # Lesions should remain brighter than background after equalization
        lesion_pixels_1 = result[45:55, 45:55]
        lesion_pixels_2 = result[20:30, 70:80]
        background_pixels = result[img == 80]
        
        assert lesion_pixels_1.mean() > background_pixels.mean(), \
            "Lesion 1 should be brighter than background"
        assert lesion_pixels_2.mean() > background_pixels.mean(), \
            "Lesion 2 should be brighter than background"
    
    def test_enhances_subtle_differences(self):
        """Test that HE enhances subtle tissue differences (key medical benefit)"""
        # Create image with subtle tissue variation (like CT liver/kidney contrast)
        img = np.full((100, 100), 100, dtype=np.uint8)
        img[30:70, 30:70] = 108  # Organ with 8 HU difference (subtle but real)
        
        result = global_histogram_equalization(img)
        
        # The organ should have higher contrast against background after HE
        organ_pixels = result[30:70, 30:70]
        background_pixels = result[img == 100]
        
        contrast_orig = 108 - 100  # 8 HU difference
        contrast_eq = organ_pixels.mean() - background_pixels.mean()
        
        # HE should increase the relative contrast
        # (contrast relative to the total range)
        range_orig = img.max() - img.min()
        range_eq = result.max() - result.min()
        
        relative_contrast_orig = contrast_orig / max(range_orig, 1)
        relative_contrast_eq = contrast_eq / max(range_eq, 1)
        
        assert relative_contrast_eq >= relative_contrast_orig, \
            f"Relative contrast decreased: {relative_contrast_orig:.3f} -> {relative_contrast_eq:.3f}"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestGlobalHEEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_uniform_image(self, uniform_image):
        """Test with completely uniform image"""
        result = global_histogram_equalization(uniform_image)
        assert result.shape == uniform_image.shape
        assert result.dtype == np.uint8
        # All values should be preserved (or all become 128)
        assert result.min() == result.max()
    
    def test_single_pixel(self):
        """Test 1x1 image"""
        img = np.array([[100]], dtype=np.uint8)
        result = global_histogram_equalization(img)
        assert result.shape == (1, 1)
        assert result.dtype == np.uint8
    
    def test_single_row(self):
        """Test 1xN image"""
        img = np.array([[10, 50, 100, 150, 200]], dtype=np.uint8)
        result = global_histogram_equalization(img)
        assert result.shape == (1, 5)
        assert result.dtype == np.uint8
    
    def test_single_column(self):
        """Test Nx1 image"""
        img = np.array([[10], [50], [100], [150], [200]], dtype=np.uint8)
        result = global_histogram_equalization(img)
        assert result.shape == (5, 1)
        assert result.dtype == np.uint8
    
    def test_binary_image(self):
        """Test black and white image"""
        img = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        result = global_histogram_equalization(img)
        assert result.shape == img.shape
        assert result.min() == 0
        assert result.max() == 255
    
    def test_gradient_image(self, gradient_image):
        """Test linear gradient image"""
        result = global_histogram_equalization(gradient_image)
        assert result.shape == gradient_image.shape
        assert result.min() == 0
        assert result.max() == 255


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestGlobalHEErrors:
    """Test error handling"""
    
    def test_none_input(self):
        """Test None raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            global_histogram_equalization(None)
        assert "None" in str(exc_info.value)
    
    def test_empty_array(self):
        """Test empty array raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            global_histogram_equalization(np.array([], dtype=np.uint8))
        assert "empty" in str(exc_info.value)
    
    def test_3d_array(self):
        """Test RGB (3D) input raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            global_histogram_equalization(np.zeros((10, 10, 3), dtype=np.uint8))
        assert "2D" in str(exc_info.value)
    
    def test_1d_array(self):
        """Test 1D input raises ValueError"""
        with pytest.raises(ValueError):
            global_histogram_equalization(np.array([1, 2, 3], dtype=np.uint8))


# ============================================================================
# LARGE IMAGE TESTS
# ============================================================================

class TestGlobalHELargeImages:
    """Test performance with larger images"""
    
    def test_512x512_image(self):
        """Test with medical-image-sized input"""
        large_img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        import time
        start = time.time()
        result = global_histogram_equalization(large_img)
        duration = time.time() - start
        
        assert result.shape == large_img.shape
        assert duration < 5.0, f"Took too long: {duration:.2f}s"
    
    def test_different_aspect_ratios(self):
        """Test with non-square images"""
        for shape in [(100, 200), (200, 100), (50, 150), (300, 100)]:
            img = np.random.randint(0, 256, shape, dtype=np.uint8)
            result = global_histogram_equalization(img)
            assert result.shape == shape