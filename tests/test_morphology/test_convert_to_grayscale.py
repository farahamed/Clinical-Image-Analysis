import numpy as np
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from processing.morphology.binary_morphology import convert_to_grayscale


class TestConvertToGrayscale(unittest.TestCase):
    """Tests for convert_to_grayscale function"""
    
    def setUp(self):
        """Create test images"""
        # Grayscale 5x5 image
        self.gray_image = np.array([
            [0, 50, 100, 150, 200],
            [25, 75, 125, 175, 225],
            [50, 100, 150, 200, 250],
            [75, 125, 175, 225, 255],
            [100, 150, 200, 250, 255]
        ], dtype=np.uint8)
        
        # Color RGB 5x5 image (all same color for testing)
        self.color_image = np.zeros((5, 5, 3), dtype=np.uint8)
        self.color_image[:, :, 0] = 100  # Red
        self.color_image[:, :, 1] = 150  # Green
        self.color_image[:, :, 2] = 200  # Blue
        
        # Single pixel color image
        self.single_pixel = np.array([[[100, 150, 200]]], dtype=np.uint8)
    
    def test_grayscale_input_returns_same(self):
        """Grayscale image should pass through unchanged"""
        result = convert_to_grayscale(self.gray_image)
        np.testing.assert_array_equal(result, self.gray_image)
        print("✓ PASS: Grayscale input unchanged")
    
    def test_output_is_2d(self):
        """Output should always be 2D"""
        result = convert_to_grayscale(self.gray_image)
        self.assertEqual(len(result.shape), 2)
        print("✓ PASS: Output is 2D")
    
    def test_color_conversion_formula(self):
        """Test RGB to grayscale conversion formula"""
        result = convert_to_grayscale(self.color_image)
        # Expected: 0.299*100 + 0.587*150 + 0.114*200
        expected_value = int(0.299 * 100 + 0.587 * 150 + 0.114 * 200)
        self.assertEqual(result[0, 0], expected_value)
        print(f"✓ PASS: Color conversion correct (expected {expected_value}, got {result[0, 0]})")
    
    def test_color_output_shape(self):
        """Color image should lose channel dimension"""
        result = convert_to_grayscale(self.color_image)
        self.assertEqual(result.shape, (5, 5))
        print("✓ PASS: Color output correct shape")
    
    def test_single_pixel_color(self):
        """Single pixel color conversion"""
        result = convert_to_grayscale(self.single_pixel)
        expected = int(0.299 * 100 + 0.587 * 150 + 0.114 * 200)
        self.assertEqual(result[0, 0], expected)
        print(f"✓ PASS: Single pixel conversion correct ({expected})")
    
    def test_none_input_raises_error(self):
        """None input should raise ValueError"""
        with self.assertRaises(ValueError):
            convert_to_grayscale(None)
        print("✓ PASS: None input raises TypeError")
    
    def test_output_dtype_is_uint8(self):
        """Output should be uint8 type"""
        result = convert_to_grayscale(self.gray_image)
        self.assertEqual(result.dtype, np.uint8)
        print("✓ PASS: Output dtype is uint8")
    
    def test_different_color_values(self):
        """Test conversion with different color values"""
        # Pure red
        pure_red = np.array([[[255, 0, 0]]], dtype=np.uint8)
        result_red = convert_to_grayscale(pure_red)
        expected_red = int(0.299 * 255)
        self.assertEqual(result_red[0, 0], expected_red)
        print(f"✓ PASS: Pure red conversion ({expected_red})")
        
        # Pure green
        pure_green = np.array([[[0, 255, 0]]], dtype=np.uint8)
        result_green = convert_to_grayscale(pure_green)
        expected_green = int(0.587 * 255)
        self.assertEqual(result_green[0, 0], expected_green)
        print(f"✓ PASS: Pure green conversion ({expected_green})")
        
        # Pure blue
        pure_blue = np.array([[[0, 0, 255]]], dtype=np.uint8)
        result_blue = convert_to_grayscale(pure_blue)
        expected_blue = int(0.114 * 255)
        self.assertEqual(result_blue[0, 0], expected_blue)
        print(f"✓ PASS: Pure blue conversion ({expected_blue})")
        
        # White
        white = np.array([[[255, 255, 255]]], dtype=np.uint8)
        result_white = convert_to_grayscale(white)
        self.assertEqual(result_white[0, 0], 255)
        print("✓ PASS: White conversion (255)")
        
        # Black
        black = np.array([[[0, 0, 0]]], dtype=np.uint8)
        result_black = convert_to_grayscale(black)
        self.assertEqual(result_black[0, 0], 0)
        print("✓ PASS: Black conversion (0)")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTING: convert_to_grayscale")
    print("="*60 + "\n")
    
    # Create test suite and run
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConvertToGrayscale)
    unittest.TextTestRunner(verbosity=2).run(suite)