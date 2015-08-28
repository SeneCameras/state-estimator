import numpy
import unittest

import localization.sensors.imu
import localization.util


class TestImu(unittest.TestCase):
    def test_constructor(self):
        imu = localization.sensors.imu.InvensenseMPU9250(10., 0.5)
        self.assertSequenceEqual(
                [False] * 9 + [True] * 6, imu.update_vector)
        self.assertAlmostEqual(10., imu.next_measurement_time)
        self.assertAlmostEqual(2., imu.delta_time)

    def test_get_state_measurements_until(self):
        imu = localization.sensors.imu.InvensenseMPU9250(10., 0.5)
        self.assertAlmostEqual(10., imu.next_measurement_time)

        desired_covariance = numpy.diag(
                [8.726646259971648e-05] * 3 + [1.4709975] * 3)
        self.assertSequenceEqual([6, 6], imu.covariance.shape)
        for i in xrange(6):
            for j in xrange(6):
                self.assertAlmostEqual(
                        desired_covariance[i, j], imu.covariance[i, j])


        # Set a zero covariance matrix to make tests deterministic
        imu.covariance = numpy.zeros([6, 6])

        meas = imu.getStateMeasurementsUntil(
                numpy.arange(15.).reshape([15, 1]), 10.3)
        self.assertIsNotNone(meas)
        self.assertSequenceEqual([6, 1], meas.measurement.shape)
        self.assertAlmostEqual(9., meas.measurement[0, 0])
        self.assertAlmostEqual(10., meas.measurement[1, 0])
        self.assertAlmostEqual(11., meas.measurement[2, 0])
        self.assertAlmostEqual(12., meas.measurement[3, 0])
        self.assertAlmostEqual(13., meas.measurement[4, 0])
        self.assertAlmostEqual(14., meas.measurement[5, 0])
        self.assertAlmostEqual(12., imu.next_measurement_time)

        meas = imu.getStateMeasurementsUntil(
                numpy.arange(2., 17.).reshape([15, 1]), 11.6)
        self.assertIsNone(meas)
        self.assertAlmostEqual(12., imu.next_measurement_time)

        meas = imu.getStateMeasurementsUntil(
                numpy.arange(4., 19.).reshape([15, 1]), 12.2)
        self.assertIsNotNone(meas)
        self.assertSequenceEqual([6, 1], meas.measurement.shape)
        self.assertAlmostEqual(13., meas.measurement[0, 0])
        self.assertAlmostEqual(14., meas.measurement[1, 0])
        self.assertAlmostEqual(15., meas.measurement[2, 0])
        self.assertAlmostEqual(16., meas.measurement[3, 0])
        self.assertAlmostEqual(17., meas.measurement[4, 0])
        self.assertAlmostEqual(18., meas.measurement[5, 0])


if __name__ == '__main__':
    unittest.main()
