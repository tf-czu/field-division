import unittest
import numpy as np
from eval_plot import rotate_cnt


class TestEvalPlot(unittest.TestCase):
    def test_rotate_cnt(self):
        cnt = np.array([[[10, 10]],
                        [[5, 5]]], dtype=np.int32)
        center = [5, 10]
        expected_result = np.array([[[5, 15]],
                                    [[10, 10]]], dtype=np.int32)
        result = rotate_cnt(center, cnt, 90)
        np.testing.assert_array_equal(expected_result, result)


if __name__ == '__main__':
    unittest.main()
