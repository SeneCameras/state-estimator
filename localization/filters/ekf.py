import math
import numpy

import localization.filters.base
from localization.util import clampRotation, StateMember
from localization.util import rpyToRotationMatrixAndDerivatives


class Ekf(localization.filters.base.FilterBase):
    """Sensor fusion using Extended Kalman Filters."""
    def __init__(self):
        super(Ekf, self).__init__()

    def correct(self, measurement):
        """Correct the state estimate and covariance matrix.

        State estimates are fused, while covariances are accumulated.

        Parameters
        ----------
        measurement: localization.util.Measurement
            Measured value used to correct the state prediction.
        """
        update_size = len(measurement.update_vector)

        state_subset = numpy.zeros([update_size, 1]) # x
        measurement_subset = numpy.zeros([update_size, 1]) # z
        measurement_covariance_subset = numpy.zeros(
                [update_size, update_size]) # R
        state_to_measurement_subset = numpy.zeros(
                [update_size, self.state.shape[0]]) # H
        kalman_gain_subset = numpy.zeros(
                [self.state.shape[0], update_size]) # K

        for idx in xrange(update_size):
            measurement_subset[idx] = measurement.measurement[idx]
            state_subset[idx] = self.state[iui]
            for jdx in xrange(update_size):
                measurement_covariance_subset[idx, jdx] = (
                        measurement.covariance[idx, jdx])
            if measurement_covariance_subset[idx, idx] < 0.0:
                measurement_covariance_subset[idx, idx] = math.fabs(
                        measurement_covariance_subset[idx, idx])
            if measurement_covariance_subset[idx, idx] < 1e-9:
                measurement_covariance_subset[idx, idx] = 1e-9

        for idx, iui in enumerate(update_indices):
            state_to_measurement_subset[idx, iui] = 1.0

        pht = self.estimate_error_covariance.dot(state_to_measurement_subset.T)
        hphr_inv  = numpy.linalg.inv(state_to_measurement_subset.dot(pht) +
                measurement_covariance_subset)
        kalman_gain_subset = pht.dot(hphr_inv)
        innovation_subset = measurement_subset - state_subset # z - Hx

        if self.checkMahalanobisThreshold(
                innovation_subset, hphr_inv, measurement.mahalanobis_threshold):
            for idx, iui in enumerate(update_indices):
                if iui >= StateMember.roll and iui <= StateMember.yaw:
                    innovation_subset[idx] = clampRotation(
                            innovation_subset[idx])
            self.state += kalman_gain_subset.dot(innovation_subset)
            gain_residual = numpy.identity(self.state.shape[0])
            gain_residual -= kalman_gain_subset.dot(state_to_measurement_subset)
            self.estimate_error_covariance = gain_residual.dot(
                    self.estimate_error_covariance).dot(
                            gain_residual.T)
            self.estimate_error_covariance += kalman_gain_subset.dot(
                    measurement_covariance_subset).dot(
                            kalman_gain_subset.T)
            self._wrapStateAngles(
                    StateMember.roll, StateMember.pitch, StateMember.yaw)

    def predict(self, delta):
        """Predict the mean and covariance estimate after a certain delay.

        The mean is predicted using the transfer function.
        The covariance is predicted using the Jacobian.

        Parameters
        ----------
        delta: float
            Time in seconds since the last measurement.
        """
        orientation = self.state[StateMember.roll:StateMember.yaw+1]
        roll, pitch, yaw = orientation.reshape(3)

        vel = self.state[StateMember.v_x:StateMember.v_z+1]
        accel = self.state[StateMember.a_x:StateMember.a_z+1]
        angular_vel = self.state[StateMember.v_roll:StateMember.v_yaw+1]

        i_delta = delta * delta * 0.5
        rot, r_dr, r_dp, r_dy = rpyToRotationMatrixAndDerivatives(
                roll, pitch, yaw)
        rot_i = rot * delta
        rot_ii = rot * i_delta

        self._transfer_function[StateMember.x:StateMember.z+1,
                                StateMember.v_x:StateMember.v_z+1] = rot_i
        self._transfer_function[StateMember.x:StateMember.z+1,
                                StateMember.a_x:StateMember.a_z+1] = rot_ii
        self._transfer_function[StateMember.roll:StateMember.yaw+1,
                                StateMember.v_roll:StateMember.v_yaw+1] = rot_i
        self._transfer_function[StateMember.v_x, StateMember.a_x] = delta
        self._transfer_function[StateMember.v_y, StateMember.a_y] = delta
        self._transfer_function[StateMember.v_z, StateMember.a_z] = delta

        linear_mult = vel * delta + accel * i_delta
        angular_mult = angular_vel * delta

        self._transfer_function_jacobian = self._transfer_function.copy();

        self._transfer_function_jacobian[
                StateMember.x:StateMember.z+1,
                StateMember.roll:StateMember.roll+1] += r_dr.dot(linear_mult)
        self._transfer_function_jacobian[
                StateMember.x:StateMember.z+1,
                StateMember.pitch:StateMember.pitch+1] += r_dp.dot(linear_mult)
        self._transfer_function_jacobian[
                StateMember.x:StateMember.z+1,
                StateMember.yaw:StateMember.yaw+1] += r_dy.dot(linear_mult)

        self._transfer_function_jacobian[
                StateMember.roll:StateMember.yaw+1,
                StateMember.roll:StateMember.roll+1] += r_dr.dot(angular_mult)
        self._transfer_function_jacobian[
                StateMember.roll:StateMember.yaw+1,
                StateMember.pitch:StateMember.pitch+1] += r_dp.dot(angular_mult)
        self._transfer_function_jacobian[
                StateMember.roll:StateMember.yaw+1,
                StateMember.yaw:StateMember.yaw+1] += r_dy.dot(angular_mult)

        self.state = self._transfer_function.dot(self.state)
        self._wrapStateAngles(
                StateMember.roll, StateMember.pitch, StateMember.yaw)

        self.estimate_error_covariance = self._transfer_function_jacobian.dot(
                self.estimate_error_covariance).dot(
                        self._transfer_function_jacobian.T)
        self.estimate_error_covariance += self.process_noise_covariance * delta
