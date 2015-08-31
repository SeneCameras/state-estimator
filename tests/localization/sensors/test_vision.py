import numpy
import unittest

import localization.sensors.vision
import localization.util


class TestVision(unittest.TestCase):
    def test_constructor(self):
        # Set a zero covariance matrix to make tests deterministic
        vision = localization.sensors.vision.Vision(
                10., 0.5, numpy.zeros([3, 3]))
        self.assertSequenceEqual(
                [False] * 6 + [True] * 3 + [False] * 6, vision.update_vector)
        self.assertAlmostEqual(10., vision.next_measurement_time)
        self.assertAlmostEqual(2., vision.delta_time)

    def test_get_state_measurements_until(self):
        vision = localization.sensors.vision.Vision(
                10., 0.5, numpy.zeros([3, 3]))
        self.assertAlmostEqual(10., vision.next_measurement_time)

        meas = vision.getStateMeasurementsUntil(
                numpy.arange(15.).reshape([15, 1]), 10.3)
        self.assertIsNotNone(meas)
        self.assertSequenceEqual([15, 1], meas.measurement.shape)
        self.assertAlmostEqual(6., meas.measurement[6, 0])
        self.assertAlmostEqual(7., meas.measurement[7, 0])
        self.assertAlmostEqual(8., meas.measurement[8, 0])
        self.assertAlmostEqual(12., vision.next_measurement_time)

        meas = vision.getStateMeasurementsUntil(
                numpy.arange(2., 17.).reshape([15, 1]), 11.6)
        self.assertIsNone(meas)
        self.assertAlmostEqual(12., vision.next_measurement_time)

        meas = vision.getStateMeasurementsUntil(
                numpy.arange(4., 19.).reshape([15, 1]), 12.2)
        self.assertIsNotNone(meas)
        self.assertSequenceEqual([15, 1], meas.measurement.shape)
        self.assertAlmostEqual(10., meas.measurement[6, 0])
        self.assertAlmostEqual(11., meas.measurement[7, 0])
        self.assertAlmostEqual(12., meas.measurement[8, 0])
        self.assertAlmostEqual(14., vision.next_measurement_time)


if __name__ == '__main__':
    unittest.main()
