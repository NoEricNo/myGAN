import torch
import unittest
from generators import TendencyGenerator, TrendMaskGenerator, UserGenerator, UserMaskRefiner, InteractionGenerator

class TestTendencyGenerator(unittest.TestCase):
    def test_output_shape(self):
        noise = torch.randn(16, 100)
        generator = TendencyGenerator(100, 50)
        output = generator(noise)
        self.assertEqual(output.shape, (16, 50))

class TestTrendMaskGenerator(unittest.TestCase):
    def test_output_shape(self):
        tendency = torch.randn(16, 50)
        generator = TrendMaskGenerator(50, 100)
        output = generator(tendency)
        self.assertEqual(output.shape, (16, 100))

class TestUserGenerator(unittest.TestCase):
    def test_output_shape(self):
        tendency = torch.randn(16, 50)
        noise = torch.randn(16, 100)
        generator = UserGenerator(50, 100, 50)
        output = generator(tendency, noise)
        self.assertEqual(output.shape, (16, 50))

class TestUserMaskRefiner(unittest.TestCase):
    def test_output_shape(self):
        trend_mask = torch.randn(16, 100)
        noise = torch.randn(16, 100)
        generator = UserMaskRefiner(100, 100)
        output = generator(trend_mask, noise)
        self.assertEqual(output.shape, (16, 100))

class TestInteractionGenerator(unittest.TestCase):
    def test_output_shape(self):
        user_latent = torch.randn(16, 50)
        item_latent = torch.randn(16, 50)
        user_mask = torch.randn(16, 100)
        generator = InteractionGenerator(50, 50, 100, 1)
        output = generator(user_latent, item_latent, user_mask)
        self.assertEqual(output.shape, (16, 1))

if __name__ == '__main__':
    unittest.main()
