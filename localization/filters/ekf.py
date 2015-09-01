import math
import numpy

import localization.filters.base
from localization.util import clampRotation, StateMember
from localization.util import rpyToRotationMatrixAndDerivatives


class Ekf(localization.filters.base.FilterBase):
    """Implementation of the Extended Kalman Filter"""
    def __init__(self):
        super(Ekf, self).__init__()

    def correct(self, measurement):
        update_indices = []
        for idx, cond in enumerate(measurement.update_vector):
            if not cond:
                continue
            measure = measurement.measurement[idx, 0]
            if math.isnan(measure) or math.isinf(measure):
                continue
            update_indices.append(idx)

        update_size = len(update_indices)

        state_subset = numpy.zeros([update_size, 1]) # x
        measurement_subset = numpy.zeros([update_size, 1]) # z
        measurement_covariance_subset = numpy.zeros(
                [update_size, update_size]) # R
        state_to_measurement_subset = numpy.zeros(
                [update_size, self.state.shape[0]]) # H
        kalman_gain_subset = numpy.zeros(
                [self.state.shape[0], update_size]) # K
        innovation_subset = numpy.zeros([update_size, 1]) # z - Hx

        for idx, iui in enumerate(update_indices):
            measurement_subset[idx] = measurement.measurement[iui]
            state_subset[idx] = self.state[iui]
            for jdx, jui in enumerate(update_indices):
                measurement_covariance_subset[idx, jdx] = (
                        measurement.covariance[iui, jui])
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
        innovation_subset = measurement_subset - state_subset

        if self.checkMahalanobisThreshold(
                innovation_subset, hphr_inv, measurement.mahalanobis_threshold):
            for idx, iui in enumerate(update_indices):
                if iui >= StateMember.roll and iui <= StateMember.yaw:
                    innovation_subset[idx] = clampRotation(
                            innovation_subset[idx])
            self.state += kalman_gain_subset.dot(innovation_subset)
            gain_residual = self._identity
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
        orientation = self.state[StateMember.roll:StateMember.yaw+1]
        roll, pitch, yaw = orientation.reshape(3)

        vel = self.state[StateMember.v_x:StateMember.v_z+1]
        accel = self.state[StateMember.a_x:StateMember.a_z+1]
        angular_vel = self.state[StateMember.v_roll:StateMember.v_yaw+1]

        cr = math.cos(roll)
        cp = math.cos(pitch)
        cy = math.cos(yaw)
        sr = math.sin(roll)
        sp = math.sin(pitch)
        sy = math.sin(yaw)

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
