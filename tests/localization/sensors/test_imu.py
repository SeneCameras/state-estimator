import numpy
import unittest

import localization.sensors.imu
import localization.util


class TestImu(unittest.TestCase):
    def test_constructor(self):
        imu = localization.sensors.imu.InvensenseMPU9250(10., 0.5)
        self.assertSequenceEqual([9, 10, 11, 12, 13, 14], imu.update_vector)
        self.assertAlmostEqual(10., imu.next_measurement_time)
        self.assertAlmostEqual(2., imu.delta_time)
        self.assertSequenceEqual([3, 1], imu.gravity_vector.shape)
        self.assertAlmostEqual(0., imu.gravity_vector[0, 0])
        self.assertAlmostEqual(0., imu.gravity_vector[1, 0])
        self.assertAlmostEqual(0., imu.gravity_vector[2, 0])
        self.assertSequenceEqual([3, 1], imu.gyro_drift.shape)
        self.assertAlmostEqual(0., imu.gyro_drift[0, 0])
        self.assertAlmostEqual(0., imu.gyro_drift[1, 0])
        self.assertAlmostEqual(0., imu.gyro_drift[2, 0])

        imu = localization.sensors.imu.InvensenseMPU9250(
                10., 0.5, [1., 2., 3.], 9.2)
        self.assertSequenceEqual([9, 10, 11, 12, 13, 14], imu.update_vector)
        self.assertAlmostEqual(10., imu.next_measurement_time)
        self.assertAlmostEqual(2., imu.delta_time)
        self.assertSequenceEqual([3, 1], imu.gravity_vector.shape)
        self.assertAlmostEqual(0., imu.gravity_vector[0, 0])
        self.assertAlmostEqual(0., imu.gravity_vector[1, 0])
        self.assertAlmostEqual(9.2, imu.gravity_vector[2, 0])
        self.assertSequenceEqual([3, 1], imu.gyro_drift.shape)
        self.assertAlmostEqual(1., imu.gyro_drift[0, 0])
        self.assertAlmostEqual(2., imu.gyro_drift[1, 0])
        self.assertAlmostEqual(3., imu.gyro_drift[2, 0])

    def test_get_state_measurements_until_no_drift(self):
        imu = localization.sensors.imu.InvensenseMPU9250(10., 0.5)
        self.assertAlmostEqual(10., imu.next_measurement_time)

        desired_covariance = numpy.diag(
                [8.726646259971648e-05] * 3 + [1.4709975e-3] * 3)
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

    def test_get_state_measurements_until_with_drift(self):
        imu = localization.sensors.imu.InvensenseMPU9250(
                10., 0.5, [1., 2., 3.], 9.81)
        self.assertAlmostEqual(10., imu.next_measurement_time)

        desired_covariance = numpy.diag(
                [8.726646259971648e-05] * 3 + [1.4709975e-3] * 3)
        self.assertSequenceEqual([6, 6], imu.covariance.shape)
        for i in xrange(6):
            for j in xrange(6):
                self.assertAlmostEqual(
                        desired_covariance[i, j], imu.covariance[i, j])


        # Set a zero covariance matrix to make tests deterministic
        imu.covariance = numpy.zeros([6, 6])

        # Test without tilting
        meas_val = numpy.arange(15.).reshape([15, 1])
        meas_val[3:6] = numpy.zeros([3, 1])
        meas = imu.getStateMeasurementsUntil(meas_val, 10.3)
        self.assertIsNotNone(meas)
        self.assertSequenceEqual([6, 1], meas.measurement.shape)
        self.assertAlmostEqual(10., meas.measurement[0, 0])
        self.assertAlmostEqual(12., meas.measurement[1, 0])
        self.assertAlmostEqual(14., meas.measurement[2, 0])
        self.assertAlmostEqual(12., meas.measurement[3, 0])
        self.assertAlmostEqual(13., meas.measurement[4, 0])
        self.assertAlmostEqual(23.81, meas.measurement[5, 0])
        self.assertAlmostEqual(12., imu.next_measurement_time)

        meas = imu.getStateMeasurementsUntil(
                numpy.arange(2., 17.).reshape([15, 1]), 11.6)
        self.assertIsNone(meas)
        self.assertAlmostEqual(12., imu.next_measurement_time)

        # Test with tilting

        g = numpy.zeros([3, 1])
        g[2, 0] = 9.81
        meas = imu.getStateMeasurementsUntil(
                numpy.arange(15.).reshape([15, 1]), 10.3)
        g_offs = localization.util.rpyToRotationMatrix(3., 4., 5.).dot(g)

        meas = imu.getStateMeasurementsUntil(
                numpy.arange(4., 19.).reshape([15, 1]), 12.2)
        g_offs = localization.util.rpyToRotationMatrix(7., 8., 9.).dot(g)
        self.assertIsNotNone(meas)
        self.assertSequenceEqual([6, 1], meas.measurement.shape)
        self.assertAlmostEqual(14., meas.measurement[0, 0])
        self.assertAlmostEqual(16., meas.measurement[1, 0])
        self.assertAlmostEqual(18., meas.measurement[2, 0])
        self.assertAlmostEqual(16. + g_offs[0, 0], meas.measurement[3, 0])
        self.assertAlmostEqual(17. + g_offs[1, 0], meas.measurement[4, 0])
        self.assertAlmostEqual(18. + g_offs[2, 0], meas.measurement[5, 0])


if __name__ == '__main__':
    unittest.main()
