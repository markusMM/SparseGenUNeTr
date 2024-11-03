import unittest
import torch
import torch.nn as nn
from src.embedders import Embedder, SumEmbedder, CrossAttentionEmbedder
from src.pos_enc import MultipleEncoding
from src.gating_blocks import BernoulliGating
from src.pos_enc import MultiWeightedPositionalEncoding

class MockPositionalEncoding(MultipleEncoding):
    def __init__(self, num_encodings: int, embed_dim: int):
        super().__init__(num_encodings, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return a dummy positional encoding (just repeating the input)
        return x.unsqueeze(1).repeat(1, self.num_encodings, 1, 1)

class TestEmbedder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.num_patches = 16
        self.num_channels = 32
        self.embed_dim = 32
        self.num_heads = 8

    def test_embedder_initialization(self):
        """
        Test the initialization of the Embedder class with a mock positional encoding.
        """
        # Create a mock positional encoding
        encoding = MockPositionalEncoding(num_encodings=3, embed_dim=self.embed_dim)

        # Initialize the Embedder with mock encoding
        embedder = Embedder(
            embedding=encoding, 
            num_patches=self.num_patches, 
            num_channels=self.num_channels
        )

        # Check that the embedder was initialized correctly
        self.assertIsInstance(embedder.embedding, MockPositionalEncoding)

    def test_embedder_with_provided_gating(self):
        """
        Test the Embedder class with provided gating.
        """
        encoding = MockPositionalEncoding(num_encodings=3, embed_dim=self.embed_dim)
        gating = BernoulliGating(
            num_patches=self.num_patches, 
            num_channels=self.num_channels, 
            num_particles=3
        )

        # Initialize Embedder with gating
        embedder = Embedder(
            embedding=encoding,
            gating=gating,
            num_patches=self.num_patches,
            num_channels=self.num_channels
        )

        # Check that the provided gating is used
        self.assertEqual(embedder.gating, gating)

    def test_sum_embedder_forward(self):
        """
        Test the forward pass of the SumEmbedder class.
        """
        encoding = MultiWeightedPositionalEncoding(
            num_encodings=1,
            embed_dim=self.embed_dim,
            num_patches=self.num_patches
        )
        sum_embedder = SumEmbedder(
            encoder=encoding,
            num_patches=self.num_patches,
            num_channels=self.num_channels
        )

        # Create a dummy input tensor
        x = torch.randn(self.batch_size, self.num_channels, self.num_patches, 8, 8)

        # Perform forward pass
        output = sum_embedder(x)

        # Check that the output has the correct shape
        self.assertEqual(output.shape, x.shape)

        # Check that the output is the sum of the input and the embedding
        pe = encoding(x)
        if len(pe.shape) > 4:
            pe = pe.sum(1)
        pe = pe.squeeze(1)
        expected_output = x + pe.permute(0, 2, 1)[:, :, :, None, None]
        self.assertTrue(torch.allclose(output, expected_output))

    def test_cross_attention_embedder_forward(self):
        """
        Test the forward pass of the CrossAttentionEmbedder class.
        """
        encoding = MockPositionalEncoding(num_encodings=1, embed_dim=self.embed_dim)

        cross_attention_embedder = CrossAttentionEmbedder(
            embedding=encoding,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_patches=self.num_patches,
            num_channels=self.num_channels
        )

        # Create a dummy input tensor with spatial dimensions
        x = torch.randn(self.batch_size, self.num_channels, self.num_patches)

        # Perform forward pass
        output = cross_attention_embedder(x)

        # Check that the output has the correct shape
        self.assertEqual(output.shape, x.shape)

    def test_cross_attention_embedder_attention_mechanism(self):
        """
        Test that the CrossAttentionEmbedder applies multi-head attention correctly.
        """
        encoding = MockPositionalEncoding(num_encodings=1, embed_dim=self.embed_dim)

        cross_attention_embedder = CrossAttentionEmbedder(
            embedding=encoding,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_patches=self.num_patches,
            num_channels=self.num_channels
        )

        # Create a dummy input tensor with spatial dimensions
        x = torch.randn(self.batch_size, self.num_channels, self.num_patches)

        # Perform forward pass and obtain attention output
        with torch.no_grad():
            attn_output = cross_attention_embedder(x)

        # Ensure the attention output is not simply a copy of x
        self.assertFalse(torch.allclose(attn_output, x))


if __name__ == "__main__":
    unittest.main()
