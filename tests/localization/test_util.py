import unittest

import localization.filters.base
import localization.util


class TestUtil(unittest.TestCase):
    def test_measurement_struct(self):
        meas1 = localization.util.Measurement(0., None, None, None)
        meas2 = localization.util.Measurement(0., None, None, None)
        self.assertEqual(0., meas1.time)
        self.assertEqual(0., meas2.time)
        self.assertFalse(meas1 > meas2)
        self.assertFalse(meas2 > meas1)
        meas1.time = 100
        meas2.time = 200
        self.assertFalse(meas1 > meas2)
        self.assertTrue(meas2 > meas1)

    def test_clamp_rotation(self):
        self.assertAlmostEqual(
                0.716814, localization.util.clampRotation(7.), delta=0.001)
        self.assertAlmostEqual(
                2.884961, localization.util.clampRotation(72.), delta=0.001)
        self.assertAlmostEqual(
                -0.716814, localization.util.clampRotation(-7.), delta=0.001)
        self.assertAlmostEqual(
                -2.884961, localization.util.clampRotation(-72.), delta=0.001)


if __name__ == '__main__':
    unittest.main()
