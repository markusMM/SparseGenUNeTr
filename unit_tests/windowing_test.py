import unittest
from src.windowing import *


class TestWindowFunctions(unittest.TestCase):

    def test_window_partition_2d(self):
        # Create a 2D tensor of shape [2, 3, 16, 16]
        x = torch.arange(2 * 3 * 16 * 16).view(2, 3, 16, 16)
        window_size = 4

        # Partition windows
        windows = window_partition_2d(x, window_size)
        
        # Check the shape after partition
        expected_shape = (2, 3 * (16 // window_size)**2, window_size, window_size)
        self.assertEqual(windows.shape, expected_shape)

        # Check the values
        expected_first_window = x[0, 0, :window_size, :window_size]
        self.assertTrue(torch.equal(windows[0, 0], expected_first_window))

    def test_window_reverse_2d(self):
        # Create a tensor of partitioned windows [2, 16, 4, 4]
        windows = torch.arange(2 * 16 * 4 * 4).view(2, 16, 4, 4)
        window_size = 4
        H, W = 16, 16

        # Reverse window partitioning
        x = window_reverse_2d(windows, window_size, H, W)
        
        # Check the shape after reversing
        expected_shape = (2, 1, 16, 16)  # 4 channels as expected output channels
        self.assertEqual(x.shape, expected_shape)

    def test_window_partition_3d(self):
        # Create a 3D tensor of shape [2, 3, 8, 8, 8]
        x = torch.arange(2 * 3 * 8 * 8 * 8).view(2, 3, 8, 8, 8)
        window_size = 4

        # Partition windows
        windows = window_partition_3d(x, window_size)
        
        # Check the shape after partition
        expected_shape = (2, 3 * (8 // window_size)**3, window_size, window_size, window_size)
        self.assertEqual(windows.shape, expected_shape)

        # Check the values
        expected_first_window = x[0, 0, :window_size, :window_size, :window_size]
        self.assertTrue(torch.equal(windows[0, 0], expected_first_window))

    def test_window_reverse_3d(self):
        # Create a tensor of partitioned windows [2, 8, 4, 4, 4]
        windows = torch.arange(2 * 8 * 4 * 4 * 4).view(2, 8, 4, 4, 4)
        window_size = 4
        H, W, D = 8, 8, 8

        # Reverse window partitioning
        x = window_reverse_3d(windows, window_size, H, W, D)
        
        # Check the shape after reversing
        expected_shape = (2, 1, 8, 8, 8)  # 4 channels as expected output channels
        self.assertEqual(x.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
