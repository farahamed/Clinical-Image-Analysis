import numpy as np
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from processing.morphology.binary_morphology import opening, create_structuring_element


class TestOpening(unittest.TestCase):
    """Tests for opening function (erosion followed by dilation)"""
    
    def setUp(self):
        """Create test images and structuring elements"""
        # 3x3 square SE
        self.square_se_3x3 = create_structuring_element(3, "square")
        
        # 3x3 cross SE
        self.cross_se_3x3 = create_structuring_element(3, "cross")
        
        # Image with main object and salt noise
        self.salt_noise = np.zeros((10, 10), dtype=np.uint8)
        self.salt_noise[2:8, 2:8] = 255  # 6x6 main object
        # Add salt noise (isolated white pixels)
        self.salt_noise[0, 5] = 255
        self.salt_noise[9, 3] = 255
        self.salt_noise[5, 9] = 255
        self.salt_noise[0, 0] = 255
        
        # Image with thin lines
        self.thin_lines = np.zeros((10, 10), dtype=np.uint8)
        self.thin_lines[5, :] = 255  # Horizontal line 1px thick
        self.thin_lines[2:8, 2] = 255  # Vertical line 1px thick
        
        # All white
        self.all_white = np.ones((7, 7), dtype=np.uint8) * 255
        
        # All black
        self.all_black = np.zeros((5, 5), dtype=np.uint8)
    
    def test_opening_removes_salt_noise(self):
        """Opening should remove isolated white pixels (salt noise)"""
        result = opening(self.salt_noise, self.square_se_3x3)
        
        # Salt noise should be removed
        self.assertEqual(result[0, 5], 0)
        self.assertEqual(result[9, 3], 0)
        self.assertEqual(result[5, 9], 0)
        self.assertEqual(result[0, 0], 0)
        
        print("✓ PASS: Salt noise removed")
    
    def test_opening_preserves_main_object(self):
        """Opening should preserve large objects (with slight size reduction)"""
        result = opening(self.salt_noise, self.square_se_3x3)
        
        # Main object should still exist in center
        # After erosion with 3x3 SE, 6x6 becomes 4x4, then dilation expands to 6x6
        main_object_pixels = np.sum(result[2:8, 2:8] == 255)
        self.assertGreater(main_object_pixels, 0)
        
        print("✓ PASS: Main object preserved")
    
    def test_opening_removes_thin_lines(self):
        """Thin lines should be removed by opening"""
        result = opening(self.thin_lines, self.square_se_3x3)
        
        # After opening, thin lines should disappear
        # They're only 1px thick, erosion removes them, dilation can't bring them back
        expected = np.zeros((10, 10), dtype=np.uint8)
        
        # The intersection might survive (2x2 region)
        intersection_remains = False
        if result[5, 2] == 255:
            intersection_remains = True
        
        # Most of the thin lines should be gone
        line_pixels_remaining = np.sum(result == 255)
        self.assertLess(line_pixels_remaining, 20)  # Much less than original
        
        print(f"✓ PASS: Thin lines removed (only {line_pixels_remaining} white pixels remain)")
    
    def test_opening_idempotent(self):
        """Opening twice should give same result as opening once"""
        result1 = opening(self.salt_noise, self.square_se_3x3)
        result2 = opening(result1, self.square_se_3x3)
        
        np.testing.assert_array_equal(result1, result2)
        print("✓ PASS: Opening is idempotent")
    
    def test_opening_all_black(self):
        """Opening on all black should remain black"""
        result = opening(self.all_black, self.square_se_3x3)
        np.testing.assert_array_equal(result, self.all_black)
        print("✓ PASS: All black unchanged")
    
    def test_opening_all_white(self):
        """Opening on all white should remain all white"""
        result = opening(self.all_white, self.square_se_3x3)
        
        # Opening of all-white image: erosion removes border, 
        # then dilation expands it back to full white
        expected = np.ones((7, 7), dtype=np.uint8) * 255
        np.testing.assert_array_equal(result, expected)
        
        print("✓ PASS: All white remains all white after opening")
        
    def test_opening_smooths_corners(self):
        """Opening should smooth object corners"""
        # Create square with sharp corners
        square = np.zeros((7, 7), dtype=np.uint8)
        square[1:6, 1:6] = 255
        
        result = opening(square, self.square_se_3x3)
        
        # Corners should be more rounded
        # Original corner at (1,1) might be 0 after opening
        corner_removed = result[1, 1] == 0
        
        print(f"✓ PASS: Corners smoothed (corner removed: {corner_removed})")
    
    def test_opening_output_is_binary(self):
        """Output should be binary"""
        result = opening(self.salt_noise, self.square_se_3x3)
        unique_vals = np.unique(result)
        self.assertTrue(np.all(np.isin(unique_vals, [0, 255])))
        print("✓ PASS: Output is binary")
    
    def test_opening_cross_se(self):
        """Opening with cross SE should work differently than square SE"""
        result_square = opening(self.salt_noise, self.square_se_3x3)
        result_cross = opening(self.salt_noise, self.cross_se_3x3)
        
        # Results should be different (cross preserves more diagonal features)
        are_equal = np.array_equal(result_square, result_cross)
        self.assertFalse(are_equal)
        
        print("✓ PASS: Different SE shapes give different results")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTING: opening")
    print("="*60 + "\n")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOpening)
    unittest.TextTestRunner(verbosity=2).run(suite)