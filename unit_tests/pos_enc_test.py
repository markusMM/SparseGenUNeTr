import unittest
import torch
from src.pos_enc import MultiWeightedPositionalEncoding
from src.pos_enc import MultiRotationalEncoding


class TestMultiRotationalEncoding(unittest.TestCase):

    def test_shape_of_rotational_encoding_3d(self):
        # Test that the 3D output shape matches expectations
        kernel_size = (4, 4, 4)
        embed_dim = 32
        num_encodings = 3
        batch_size = 8

        # Create instance of MultiRotationalEncoding for 3D
        rot_enc = MultiRotationalEncoding(
            kernel_size=kernel_size,  # type: ignore
            embed_dim=embed_dim, 
            num_encodings=num_encodings, 
            is_3d=True
        )

        # Input tensor with (batch_size, channels, depth, height, width)
        x = torch.randn(batch_size, embed_dim, *kernel_size)

        # Forward pass
        encodings = rot_enc(x)

        # Expected shape: (batch_size, num_encodings, flattened)
        expected_shape = (batch_size, num_encodings, -1)
        self.assertEqual(encodings.shape[0], batch_size)
        self.assertEqual(encodings.shape[1], num_encodings)

    def test_shape_of_rotational_encoding_2d(self):
        # Test that the 2D output shape matches expectations
        kernel_size = (4, 4)
        embed_dim = 32
        num_encodings = 4
        batch_size = 8

        # Create instance of MultiRotationalEncoding for 2D
        rot_enc = MultiRotationalEncoding(
            kernel_size=kernel_size,  # type: ignore
            embed_dim=embed_dim, 
            num_encodings=num_encodings, 
            is_3d=False
        )

        # Input tensor with (batch_size, channels, height, width)
        x = torch.randn(batch_size, embed_dim, *kernel_size)

        # Forward pass
        encodings = rot_enc(x)

        # Expected shape: (batch_size, num_encodings, flattened)
        expected_shape = (batch_size, num_encodings, -1)
        self.assertEqual(encodings.shape[0], batch_size)
        self.assertEqual(encodings.shape[1], num_encodings)

    def test_learnable_theta_parameter(self):
        # Test that the theta parameter is learnable and has correct dimensions
        kernel_size = (4, 4, 4)
        embed_dim = 32
        num_encodings = 3

        rot_enc = MultiRotationalEncoding(
            kernel_size=kernel_size,  # type: ignore
            embed_dim=embed_dim, 
            num_encodings=num_encodings, 
            is_3d=True
        )

        # Check that the theta parameter is learnable
        self.assertTrue(rot_enc.theta.requires_grad)

        # Check the shape of the learnable theta parameters
        self.assertEqual(rot_enc.theta.shape, (num_encodings // len(kernel_size), *kernel_size))


class TestMultiWeightedPositionalEncoding(unittest.TestCase):
    
    def test_shape_of_positional_encoding(self):
        # Test that the output shape matches expectations
        num_patches = 16
        embed_dim = 32
        num_encodings = 3
        batch_size = 8

        # Create an instance of the MultiWeightedPositionalEncoding
        pos_enc = MultiWeightedPositionalEncoding(num_patches=num_patches, embed_dim=embed_dim, num_encodings=num_encodings)

        # Input tensor simulating an arbitrary tensor to pass through the positional encoding
        x = torch.randn(batch_size, embed_dim, num_patches)

        # Forward pass
        pos_embedding = pos_enc(x)

        # Expected shape: (batch_size, num_encodings, num_patches, embed_dim)
        expected_shape = (batch_size, num_encodings, 1, num_patches, embed_dim)
        self.assertEqual(pos_embedding.shape, expected_shape)

    def test_learnable_parameters(self):
        # Test that the position embedding is learnable and initialized correctly
        num_patches = 16
        embed_dim = 32
        num_encodings = 3

        pos_enc = MultiWeightedPositionalEncoding(num_patches=num_patches, embed_dim=embed_dim, num_encodings=num_encodings)

        # Check that the position_embedding parameter is learnable
        self.assertTrue(pos_enc.position_embedding.requires_grad)

        # Check the shape of the learnable parameters
        self.assertEqual(pos_enc.position_embedding.shape, (num_encodings, 1, num_patches, embed_dim))


if __name__ == "__main__":
    unittest.main()
