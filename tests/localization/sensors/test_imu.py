import numpy
import unittest
import math

import localization.sensors.imu
import localization.util


class TestUtilities(unittest.TestCase):
    def test_z_axis_rotation(self):
        phi = 0.8
        cos_phi_half = math.cos(phi / 2.)
        sin_phi_half = math.sin(phi / 2.)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)

        # Rotate z axis around z axis
        result = localization.sensors.imu.rotateZAxisByQuaternion(
                [cos_phi_half, 0., 0., sin_phi_half])
        self.assertSequenceEqual([3, 1], result.shape)
        self.assertAlmostEqual(0., result[0, 0])
        self.assertAlmostEqual(0., result[1, 0])
        self.assertAlmostEqual(1., result[2, 0])

        # Rotate z axis around x axis
        result = localization.sensors.imu.rotateZAxisByQuaternion(
                [cos_phi_half, sin_phi_half, 0., 0.])
        self.assertSequenceEqual([3, 1], result.shape)
        self.assertAlmostEqual(0., result[0, 0])
        self.assertAlmostEqual(sin_phi, result[1, 0])
        self.assertAlmostEqual(cos_phi, result[2, 0])

        # Rotate z axis around y axis
        result = localization.sensors.imu.rotateZAxisByQuaternion(
                [cos_phi_half, 0., sin_phi_half, 0.])
        self.assertSequenceEqual([3, 1], result.shape)
        self.assertAlmostEqual(-sin_phi, result[0, 0])
        self.assertAlmostEqual(0., result[1, 0])
        self.assertAlmostEqual(cos_phi, result[2, 0])


class TestInvensenseMPU9250(unittest.TestCase):
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


class TestImuCompensated(unittest.TestCase):
    def test_constructor(self):
        imu = localization.sensors.imu.ImuCompensated(10., 4000., 'Madgwick')
        self.assertSequenceEqual([9, 10, 11, 12, 13, 14], imu.update_vector)
        self.assertAlmostEqual(10., imu.next_measurement_time)
        self.assertAlmostEqual(0.00025, imu.delta_time)
        self.assertAlmostEqual(0., imu.gravity)
        self.assertSequenceEqual([3, 1], imu.sensor.gravity_vector.shape)
        self.assertAlmostEqual(0., imu.sensor.gravity_vector[0, 0])
        self.assertAlmostEqual(0., imu.sensor.gravity_vector[1, 0])
        self.assertAlmostEqual(0., imu.sensor.gravity_vector[2, 0])
        self.assertSequenceEqual([3, 1], imu.sensor.gyro_drift.shape)
        self.assertAlmostEqual(0., imu.sensor.gyro_drift[0, 0])
        self.assertAlmostEqual(0., imu.sensor.gyro_drift[1, 0])
        self.assertAlmostEqual(0., imu.sensor.gyro_drift[2, 0])
        self.assertSequenceEqual([3, 1], imu.gyro_compensation.shape)
        self.assertAlmostEqual(0., imu.gyro_compensation[0, 0])
        self.assertAlmostEqual(0., imu.gyro_compensation[1, 0])
        self.assertAlmostEqual(0., imu.gyro_compensation[2, 0])

        imu = localization.sensors.imu.ImuCompensated(
                10., 4000., 'Mahony', [1., 2., 3.], 9.2)
        self.assertSequenceEqual([9, 10, 11, 12, 13, 14], imu.update_vector)
        self.assertAlmostEqual(10., imu.next_measurement_time)
        self.assertAlmostEqual(0.00025, imu.delta_time)
        self.assertAlmostEqual(9.2, imu.gravity)
        self.assertSequenceEqual([3, 1], imu.sensor.gravity_vector.shape)
        self.assertAlmostEqual(0., imu.sensor.gravity_vector[0, 0])
        self.assertAlmostEqual(0., imu.sensor.gravity_vector[1, 0])
        self.assertAlmostEqual(9.2, imu.sensor.gravity_vector[2, 0])
        self.assertSequenceEqual([3, 1], imu.sensor.gyro_drift.shape)
        self.assertAlmostEqual(1., imu.sensor.gyro_drift[0, 0])
        self.assertAlmostEqual(2., imu.sensor.gyro_drift[1, 0])
        self.assertAlmostEqual(3., imu.sensor.gyro_drift[2, 0])
        self.assertSequenceEqual([3, 1], imu.gyro_compensation.shape)
        self.assertAlmostEqual(1., imu.gyro_compensation[0, 0])
        self.assertAlmostEqual(2., imu.gyro_compensation[1, 0])
        self.assertAlmostEqual(3., imu.gyro_compensation[2, 0])

    def test_level_hovering_no_drift(self):
        imu = localization.sensors.imu.ImuCompensated(10., 0.5, 'Madgwick')
        self.assertAlmostEqual(0., imu.gravity)
        self.assertAlmostEqual(10., imu.next_measurement_time)

        desired_covariance = numpy.diag(
                [8.726646259971648e-05] * 3 + [1.4709975e-3] * 3)
        self.assertSequenceEqual([6, 6], imu.covariance.shape)
        for i in xrange(6):
            for j in xrange(6):
                self.assertAlmostEqual(
                        desired_covariance[i, j], imu.covariance[i, j])

        # Set a zero covariance matrix to make tests deterministic
        imu.sensor.covariance = numpy.zeros([6, 6])

        state_vector = numpy.zeros([15, 1])

        for t in numpy.arange(10., 12., 1. / 4000.):
            imu.getStateMeasurementsUntil(state_vector, t)

        meas = imu.getStateMeasurementsUntil(state_vector, 12.)

        self.assertAlmostEqual(0., imu.gravity)

        self.assertIsNotNone(meas)
        self.assertSequenceEqual([6, 1], meas.measurement.shape)
        self.assertAlmostEqual(0., meas.measurement[0, 0])
        self.assertAlmostEqual(0., meas.measurement[1, 0])
        self.assertAlmostEqual(0., meas.measurement[2, 0])
        self.assertAlmostEqual(0., meas.measurement[3, 0])
        self.assertAlmostEqual(0., meas.measurement[4, 0])
        self.assertAlmostEqual(0., meas.measurement[5, 0])

    def test_level_hovering_with_gravity(self):
        imu = localization.sensors.imu.ImuCompensated(
                10., 0.5, 'Madgwick', gravity_drift=9.81)
        self.assertAlmostEqual(9.81, imu.gravity)
        self.assertAlmostEqual(10., imu.next_measurement_time)

        desired_covariance = numpy.diag(
                [8.726646259971648e-05] * 3 + [1.4709975e-3] * 3)
        self.assertSequenceEqual([6, 6], imu.covariance.shape)
        for i in xrange(6):
            for j in xrange(6):
                self.assertAlmostEqual(
                        desired_covariance[i, j], imu.covariance[i, j])

        # Set a zero covariance matrix to make tests deterministic
        imu.sensor.covariance = numpy.zeros([6, 6])

        state_vector = numpy.zeros([15, 1])

        for t in numpy.arange(10., 12., 1. / 4000.):
            imu.getStateMeasurementsUntil(state_vector, t)

        meas = imu.getStateMeasurementsUntil(state_vector, 12.)

        self.assertAlmostEqual(9.81, imu.gravity)

        self.assertIsNotNone(meas)
        self.assertSequenceEqual([6, 1], meas.measurement.shape)
        self.assertAlmostEqual(0., meas.measurement[0, 0])
        self.assertAlmostEqual(0., meas.measurement[1, 0])
        self.assertAlmostEqual(0., meas.measurement[2, 0])
        self.assertAlmostEqual(0., meas.measurement[3, 0])
        self.assertAlmostEqual(0., meas.measurement[4, 0])
        self.assertAlmostEqual(0., meas.measurement[5, 0])

    def test_level_hovering_with_bad_gravity_corrected(self):
        imu = localization.sensors.imu.ImuCompensated(
                10., 0.5, 'Madgwick', gravity_drift=9.81)
        imu.setGravity(9.5)
        imu.setGravityCorrectionRate(0.1)
        self.assertAlmostEqual(9.5, imu.gravity)
        self.assertAlmostEqual(10., imu.next_measurement_time)

        desired_covariance = numpy.diag(
                [8.726646259971648e-05] * 3 + [1.4709975e-3] * 3)
        self.assertSequenceEqual([6, 6], imu.covariance.shape)
        for i in xrange(6):
            for j in xrange(6):
                self.assertAlmostEqual(
                        desired_covariance[i, j], imu.covariance[i, j])

        # Set a zero covariance matrix to make tests deterministic
        imu.sensor.covariance = numpy.zeros([6, 6])

        state_vector = numpy.zeros([15, 1])

        for t in numpy.arange(10., 1000., 1. / 4000.):
            imu.getStateMeasurementsUntil(state_vector, t)

        meas = imu.getStateMeasurementsUntil(state_vector, 1000.)

        self.assertAlmostEqual(9.81, imu.gravity)

        self.assertIsNotNone(meas)
        self.assertSequenceEqual([6, 1], meas.measurement.shape)
        self.assertAlmostEqual(0., meas.measurement[0, 0])
        self.assertAlmostEqual(0., meas.measurement[1, 0])
        self.assertAlmostEqual(0., meas.measurement[2, 0])
        self.assertAlmostEqual(0., meas.measurement[3, 0])
        self.assertAlmostEqual(0., meas.measurement[4, 0])
        self.assertAlmostEqual(0., meas.measurement[5, 0])

    def test_level_hovering_with_bad_attitude(self):
        imu = localization.sensors.imu.ImuCompensated(
                10., 4000., 'Madgwick', gravity_drift=9.81)

        # Set a zero covariance matrix to make tests deterministic
        imu.sensor.covariance = numpy.zeros([6, 6])

        state_vector = numpy.zeros([15, 1])

        state_vector[localization.util.StateMember.roll, 0] = 0.1
        state_vector[localization.util.StateMember.pitch, 0] = 0.2

        for t in numpy.arange(10., 15., 1. / 4000.):
            imu.getStateMeasurementsUntil(state_vector, t)

        meas = imu.getStateMeasurementsUntil(state_vector, 15.)

        self.assertAlmostEqual(9.81, imu.gravity)

        self.assertIsNotNone(meas)
        self.assertSequenceEqual([6, 1], meas.measurement.shape)
        self.assertAlmostEqual(0., meas.measurement[0, 0], delta=0.001)
        self.assertAlmostEqual(0., meas.measurement[1, 0], delta=0.001)
        self.assertAlmostEqual(0., meas.measurement[2, 0], delta=0.001)
        self.assertAlmostEqual(0., meas.measurement[3, 0], delta=0.001)
        self.assertAlmostEqual(0., meas.measurement[4, 0], delta=0.001)
        self.assertAlmostEqual(0., meas.measurement[5, 0], delta=0.001)

    def test_level_hovering_with_bad_attitude_and_gravity(self):
        imu = localization.sensors.imu.ImuCompensated(
                10., 1000., 'Madgwick', gravity_drift=9.81)
        imu.setGravity(9.75)
        imu.setGravityCorrectionRate(0.5)
        self.assertAlmostEqual(9.75, imu.gravity)

        # Set a zero covariance matrix to make tests deterministic
        imu.sensor.covariance = numpy.zeros([6, 6])

        state_vector = numpy.zeros([15, 1])

        state_vector[localization.util.StateMember.roll, 0] = 0.1
        state_vector[localization.util.StateMember.pitch, 0] = 0.2

        for t in numpy.arange(10., 30., 1. / 1000.):
            imu.getStateMeasurementsUntil(state_vector, t)

        meas = imu.getStateMeasurementsUntil(state_vector, 30.)

        self.assertAlmostEqual(9.81, imu.gravity, delta=0.001)

        self.assertIsNotNone(meas)
        self.assertSequenceEqual([6, 1], meas.measurement.shape)
        self.assertAlmostEqual(0., meas.measurement[0, 0], delta=0.001)
        self.assertAlmostEqual(0., meas.measurement[1, 0], delta=0.001)
        self.assertAlmostEqual(0., meas.measurement[2, 0], delta=0.001)
        self.assertAlmostEqual(0., meas.measurement[3, 0], delta=0.001)
        self.assertAlmostEqual(0., meas.measurement[4, 0], delta=0.001)
        self.assertAlmostEqual(0., meas.measurement[5, 0], delta=0.001)


if __name__ == '__main__':
    unittest.main()
