import unittest
from custom_dataset import SlidingWindowDataset


class TestSlidingWindowDataset(unittest.TestCase):
    def test_len(self):
        # Test using a regular Python list
        ds = SlidingWindowDataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
        self.assertEqual(len(ds), 7)

        ds1 = SlidingWindowDataset([0, 1, 2], 1)
        self.assertEqual(len(ds1), 2)

    def test_getitem(self):
        ds = SlidingWindowDataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
        self.assertEqual(ds[1], ([1, 2, 3], 4))
        self.assertEqual(ds[6], ([6, 7, 8], 9))

        ds1 = SlidingWindowDataset([0, 1, 2], 1)
        self.assertEqual(ds1[0], ([0], 1))


unittest.main(argv=[""], verbosity=2, exit=False)
