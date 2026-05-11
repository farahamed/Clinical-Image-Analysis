import numpy as np
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from processing.morphology.binary_morphology import erode, create_structuring_element


class TestErosion(unittest.TestCase):
    """Tests for erode function"""
    
    def setUp(self):
        """Create test images and structuring elements"""
        # 3x3 square SE
        self.square_se_3x3 = create_structuring_element(3, "square")
        
        # 3x3 cross SE
        self.cross_se_3x3 = create_structuring_element(3, "cross")
        
        # 5x5 square SE
        self.square_se_5x5 = create_structuring_element(5, "square")
        
        # 5x5 white square in 7x7 image (center)
        self.square_5x5_in_7x7 = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 255, 255, 255, 255, 255, 0],
            [0, 255, 255, 255, 255, 255, 0],
            [0, 255, 255, 255, 255, 255, 0],
            [0, 255, 255, 255, 255, 255, 0],
            [0, 255, 255, 255, 255, 255, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        # 3x3 white square in 5x5 image
        self.square_3x3_in_5x5 = np.array([
            [0, 0, 0, 0, 0],
            [0, 255, 255, 255, 0],
            [0, 255, 255, 255, 0],
            [0, 255, 255, 255, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        # All white image
        self.all_white = np.ones((5, 5), dtype=np.uint8) * 255
        
        # All black image
        self.all_black = np.zeros((5, 5), dtype=np.uint8)
        
        # Single white pixel
        self.single_pixel = np.zeros((5, 5), dtype=np.uint8)
        self.single_pixel[2, 2] = 255
        
        # Two separate white pixels
        self.two_pixels = np.zeros((7, 7), dtype=np.uint8)
        self.two_pixels[2, 2] = 255
        self.two_pixels[4, 4] = 255
    
    def test_erosion_3x3_square_se_on_5x5_square(self):
        """3x3 white square eroded by 3x3 square SE should become 1 pixel"""
        result = erode(self.square_3x3_in_5x5, self.square_se_3x3)
        
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 255, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: 3x3 square eroded by 3x3 SE becomes 1px")
    
    def test_erosion_5x5_square_se_on_5x5_square(self):
        """5x5 white square eroded by 5x5 square SE should become 1 pixel"""
        result = erode(self.square_5x5_in_7x7, self.square_se_5x5)
        
        # Should have single white pixel at center
        center_value = result[3, 3]
        self.assertEqual(center_value, 255)
        
        # All other pixels should be 0
        result[3, 3] = 0
        self.assertTrue(np.all(result == 0))
        
        print("✓ PASS: 5x5 square eroded by 5x5 SE becomes 1px")
    
    def test_erosion_cross_se_on_square(self):
        """Erosion with cross SE on white square"""
        result = erode(self.square_3x3_in_5x5, self.cross_se_3x3)
        
        # With cross SE, only center pixel survives 3x3 square
        # because edge pixels lack white neighbors in all 4 directions
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 255, 0, 0],    # Only center survives
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: Erosion with cross SE - only center survives")
    
    def test_erosion_single_pixel_vanishes(self):
        """Single pixel eroded by 3x3 SE should vanish"""
        result = erode(self.single_pixel, self.square_se_3x3)
        expected = np.zeros((5, 5), dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: Single pixel vanishes after erosion")
    
    def test_erosion_all_white_shrinks(self):
        """All white image should shrink from borders"""
        result = erode(self.all_white, self.square_se_3x3)
        
        # Border should be black
        self.assertTrue(np.all(result[0, :] == 0))
        self.assertTrue(np.all(result[-1, :] == 0))
        self.assertTrue(np.all(result[:, 0] == 0))
        self.assertTrue(np.all(result[:, -1] == 0))
        
        # Interior should remain white
        self.assertTrue(np.all(result[1:4, 1:4] == 255))
        
        print("✓ PASS: All white shrinks from borders")
    
    def test_erosion_all_black_stays_black(self):
        """All black image should stay black"""
        result = erode(self.all_black, self.square_se_3x3)
        np.testing.assert_array_equal(result, self.all_black)
        print("✓ PASS: All black stays black")
    
    def test_erosion_separates_objects(self):
        """Erosion should separate two nearby white regions"""
        # Create two white squares with a GAP between them
        test_img = np.zeros((10, 10), dtype=np.uint8)
        test_img[1:5, 1:4] = 255  # Left 4×3 square (cols 1-3)
        test_img[1:5, 6:9] = 255  # Right 4×3 square (cols 6-8)
        # Gap at columns 4-5 (2 black columns between squares)
        
        result = erode(test_img, self.square_se_3x3)
        
        # After erosion, they should be separated (shrunk from all sides)
        # The gap should remain
        self.assertEqual(result[2, 4], 0)  # Gap column should be 0
        self.assertEqual(result[2, 5], 0)  # Gap column should be 0
        
        # Both squares should still have some white pixels (just smaller)
        left_white = np.sum(result[2:4, 2:3] == 255)
        right_white = np.sum(result[2:4, 7:8] == 255)
        self.assertGreater(left_white, 0, "Left square should have surviving pixels")
        self.assertGreater(right_white, 0, "Right square should have surviving pixels")
        
        print("✓ PASS: Erosion separates objects with gap")
    
    def test_erosion_removes_thin_lines(self):
        """Thin lines should be removed by erosion"""
        # Horizontal line (1px thick)
        line = np.zeros((5, 5), dtype=np.uint8)
        line[2, :] = 255
        
        result = erode(line, self.square_se_3x3)
        
        # Line should vanish
        expected = np.zeros((5, 5), dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
        
        print("✓ PASS: Thin horizontal line removed")
    
    def test_erosion_output_is_binary(self):
        """Output should only contain 0 and 255"""
        result = erode(self.square_3x3_in_5x5, self.square_se_3x3)
        unique_vals = np.unique(result)
        self.assertTrue(np.all(np.isin(unique_vals, [0, 255])))
        print("✓ PASS: Output is binary")
    
    def test_erosion_larger_se_than_object(self):
        """SE larger than object should make it disappear"""
        result = erode(self.square_3x3_in_5x5, self.square_se_5x5)
        expected = np.zeros((5, 5), dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: SE larger than object removes it")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTING: erode")
    print("="*60 + "\n")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestErosion)
    unittest.TextTestRunner(verbosity=2).run(suite)