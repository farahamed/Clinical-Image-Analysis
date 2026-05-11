import numpy as np
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from processing.morphology.binary_morphology import create_structuring_element


class TestCreateStructuringElement(unittest.TestCase):
    """Tests for create_structuring_element function"""
    
    def test_square_se_3x3(self):
        """Create 3x3 square structuring element"""
        se = create_structuring_element(3, "square")
        expected = np.ones((3, 3), dtype=np.uint8)
        np.testing.assert_array_equal(se, expected)
        print("✓ PASS: 3x3 square SE")
    
    def test_square_se_5x5(self):
        """Create 5x5 square structuring element"""
        se = create_structuring_element(5, "square")
        expected = np.ones((5, 5), dtype=np.uint8)
        np.testing.assert_array_equal(se, expected)
        print("✓ PASS: 5x5 square SE")
    
    def test_square_se_7x7(self):
        """Create 7x7 square structuring element"""
        se = create_structuring_element(7, "square")
        self.assertEqual(se.shape, (7, 7))
        self.assertTrue(np.all(se == 1))
        print("✓ PASS: 7x7 square SE")
    
    def test_cross_se_3x3(self):
        """Create 3x3 cross structuring element"""
        se = create_structuring_element(3, "cross")
        expected = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=np.uint8)
        np.testing.assert_array_equal(se, expected)
        print("✓ PASS: 3x3 cross SE")
    
    def test_cross_se_5x5(self):
        """Create 5x5 cross structuring element"""
        se = create_structuring_element(5, "cross")
        
        # Center row should be all 1s
        self.assertTrue(np.all(se[2, :] == 1))
        
        # Center column should be all 1s
        self.assertTrue(np.all(se[:, 2] == 1))
        
        # Corners should be 0
        self.assertEqual(se[0, 0], 0)
        self.assertEqual(se[0, 4], 0)
        self.assertEqual(se[4, 0], 0)
        self.assertEqual(se[4, 4], 0)
        
        # Edge centers (not corners) should be 1
        self.assertEqual(se[0, 2], 1)  # Top center
        self.assertEqual(se[2, 0], 1)  # Left center
        self.assertEqual(se[2, 4], 1)  # Right center
        self.assertEqual(se[4, 2], 1)  # Bottom center
        
        print("✓ PASS: 5x5 cross SE")
    
    def test_cross_se_7x7(self):
        """Create 7x7 cross structuring element"""
        se = create_structuring_element(7, "cross")
        
        # Center row and column should be 1
        self.assertTrue(np.all(se[3, :] == 1))
        self.assertTrue(np.all(se[:, 3] == 1))
        
        # Corners should be 0
        self.assertEqual(se[0, 0], 0)
        self.assertEqual(se[0, 6], 0)
        self.assertEqual(se[6, 0], 0)
        self.assertEqual(se[6, 6], 0)
        
        print("✓ PASS: 7x7 cross SE")
    
    def test_size_less_than_3_error(self):
        """Size less than 3 should raise ValueError"""
        with self.assertRaises(ValueError):
            create_structuring_element(1, "square")
        with self.assertRaises(ValueError):
            create_structuring_element(2, "square")
        print("✓ PASS: Size < 3 raises ValueError")
    
    def test_even_size_error(self):
        """Even size should raise ValueError"""
        with self.assertRaises(ValueError):
            create_structuring_element(4, "square")
        with self.assertRaises(ValueError):
            create_structuring_element(6, "cross")
        print("✓ PASS: Even size raises ValueError")
    
    def test_invalid_shape_error(self):
        """Invalid shape should raise ValueError"""
        with self.assertRaises(ValueError):
            create_structuring_element(3, "circle")
        with self.assertRaises(ValueError):
            create_structuring_element(3, "triangle")
        print("✓ PASS: Invalid shape raises ValueError")
    
    def test_case_insensitive_shape(self):
        """Shape parameter should be case-insensitive"""
        se_lower = create_structuring_element(3, "square")
        se_upper = create_structuring_element(3, "SQUARE")
        se_mixed = create_structuring_element(3, "Square")
        
        np.testing.assert_array_equal(se_lower, se_upper)
        np.testing.assert_array_equal(se_upper, se_mixed)
        print("✓ PASS: Shape is case-insensitive")
    
    def test_cross_case_insensitive(self):
        """Cross shape should be case-insensitive"""
        se_lower = create_structuring_element(3, "cross")
        se_upper = create_structuring_element(3, "CROSS")
        se_mixed = create_structuring_element(3, "Cross")
        
        np.testing.assert_array_equal(se_lower, se_upper)
        np.testing.assert_array_equal(se_upper, se_mixed)
        print("✓ PASS: Cross shape case-insensitive")
    
    def test_output_dtype(self):
        """Output should be uint8"""
        se = create_structuring_element(3, "square")
        self.assertEqual(se.dtype, np.uint8)
        print("✓ PASS: Output dtype is uint8")
    
    def test_large_odd_sizes(self):
        """Test larger odd sizes"""
        for size in [9, 11, 15]:
            se = create_structuring_element(size, "square")
            self.assertEqual(se.shape, (size, size))
            self.assertTrue(np.all(se == 1))
        print("✓ PASS: Large odd sizes work")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTING: create_structuring_element")
    print("="*60 + "\n")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCreateStructuringElement)
    unittest.TextTestRunner(verbosity=2).run(suite)