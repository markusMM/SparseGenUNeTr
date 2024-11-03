import unittest
import numpy as np
import torch
from src.gating_blocks import BernoulliGating


class TestBernoulliGating(unittest.TestCase):

    def setUp(self):
        """ Set up parameters to be used in each test. """
        self.num_patches = 16
        self.num_channels = 32
        self.num_particles = 3
        self.batch_size = 8
        self.use_bayesian = True

    def test_output_shape(self):
        """
        Test that the output shape of the gating matches expectations.
        """
        # Create an instance of BernoulliGating
        gating = BernoulliGating(
            num_patches=self.num_patches, 
            num_channels=self.num_channels, 
            num_particles=self.num_particles,
            use_bayesian=self.use_bayesian
        )

        # Create an input tensor of encodings
        encodings = torch.randn(self.batch_size, self.num_particles, self.num_patches, self.num_channels)

        # Perform forward pass
        output = gating(encodings)

        # Expected output shape: (batch_size, num_patches, num_channels)
        expected_shape = (self.batch_size, self.num_patches, self.num_channels)
        self.assertEqual(output.shape, expected_shape)

    def test_bayesian_gating_enabled(self):
        """
        Test that the Bayesian gating is applied correctly by checking the
        state tensor (should be Bernoulli samples).
        """
        # Create an instance of BernoulliGating with Bayesian gating enabled
        gating = BernoulliGating(
            num_patches=self.num_patches, 
            num_channels=self.num_channels, 
            num_particles=self.num_particles,
            use_bayesian=True,
            patchvise=False
        )

        # Create an input tensor of encodings
        encodings = torch.randn(self.batch_size, self.num_particles, self.num_patches, self.num_channels)

        # Perform forward pass
        output = gating(encodings)

        # Check that state has been updated and matches z_filter's shape
        self.assertEqual(gating.state.shape, (self.num_particles, self.num_channels))
        
        # Check if the state values are either 0 or 1 (Bernoulli samples)
        self.assertTrue(torch.all((gating.state == 0) | (gating.state == 1)))

    def test_bayesian_gating_disabled(self):
        """
        Test that the gating does not apply the Bayesian sparsification when disabled.
        """
        # Create an instance of BernoulliGating with Bayesian gating disabled
        gating = BernoulliGating(
            num_patches=self.num_patches, 
            num_channels=self.num_channels, 
            num_particles=self.num_particles,
            use_bayesian=False
        )

        # Create an input tensor of encodings
        encodings = torch.randn(self.batch_size, self.num_particles, self.num_patches, self.num_channels)

        # Perform forward pass
        output = gating(encodings)

        # Ensure that no state exists when Bayesian gating is disabled
        self.assertFalse(hasattr(gating, 'state'))

    def test_learnable_parameters(self):
        """
        Test that the z_filter is a learnable parameter when Bayesian gating is enabled.
        """
        # Create an instance of BernoulliGating with Bayesian gating enabled
        gating = BernoulliGating(
            num_patches=self.num_patches, 
            num_channels=self.num_channels, 
            num_particles=self.num_particles,
            use_bayesian=True,
            patchvise=True
        )

        # Check if z_filter is a learnable parameter
        self.assertTrue(gating.z_filter.requires_grad)

        # Check that z_filter has the correct shape
        self.assertEqual(gating.z_filter.shape, (self.num_patches, self.num_channels))

    def test_init_parameters(self):
        """
        Test that the z_filter is an intially set parameter when Bayesian gating is enabled.
        """
        # Create an instance of BernoulliGating with Bayesian gating enabled
        gating = BernoulliGating(
            num_patches=self.num_patches, 
            num_channels=self.num_channels, 
            num_particles=1000,
            use_bayesian=True,
            patchvise=True,
            p_default=torch.tensor([.2])
        )

        # Create an input tensor of encodings
        encodings = torch.randn(self.batch_size, 1000, self.num_patches, self.num_channels)

        # Perform forward pass
        _ = gating(encodings)

        # Check if z_filter is a learnable parameter
        self.assertTrue(gating.z_filter.requires_grad)

        # Check that z_filter has the correct shape
        assert np.allclose(gating.state.mean(0), .2, .025)


if __name__ == "__main__":
    unittest.main()
