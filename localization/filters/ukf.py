import math
import numpy

import localization.filters.base
from localization.util import clampRotation, StateMember, rpyToRotationMatrix


class Ukf(localization.filters.base.FilterBase):
    """Implementation of the Extended Kalman Filter"""
    def __init__(self, alpha=None, kappa=None, beta=None):
        super(Ukf, self).__init__()
        if alpha is None:
            alpha = 0.001
        if kappa is None:
            kappa = 0.
        if beta is None:
            beta = 2.
        self._uncorrected = True

        state_size = self.state.shape[0]
        sigma_count = state_size * 2 + 1
        self._sigma_points = [
                numpy.zeros([state_size, 1]) for _ in xrange(sigma_count)]

        self._lambda_adj = alpha * alpha * (state_size + kappa)
        self._lambda = self._lambda_adj - state_size

        weight = 0.5 / self._lambda_adj
        self._state_weights = [weight] * sigma_count
        self._covar_weights = [weight] * sigma_count

        self._state_weights[0] = self._lambda / self._lambda_adj
        self._covar_weights[0] = self._state_weights[0] + (
                1. - (alpha * alpha) + beta)

    def __setSigmaPoints(self):
        state_size = self.state.shape[0]

        self._weighted_covar_sqrt = numpy.linalg.cholesky(
                self._lambda_adj * self.estimate_error_covariance)

        self._sigma_points[0] = self.state.copy()

        for i in xrange(state_size):
            self._sigma_points[i + 1] = (
                    self.state + self._weighted_covar_sqrt[:, i:i+1])
            self._sigma_points[i + 1 + state_size] = (
                    self.state - self._weighted_covar_sqrt[:, i:i+1])

    def __setSigmaPointsWithTF(self):
        state_size = self.state.shape[0]

        self._weighted_covar_sqrt = numpy.linalg.cholesky(
                self._lambda_adj * self.estimate_error_covariance)

        self._sigma_points[0] = self._transfer_function.dot(self.state)

        for i in xrange(state_size):
            self._sigma_points[i + 1] = self._transfer_function.dot(
                    self.state + self._weighted_covar_sqrt[:, i:i+1])
            self._sigma_points[i + 1 + state_size] = (
                    self._transfer_function.dot(
                            self.state - self._weighted_covar_sqrt[:, i:i+1]))

    def correct(self, measurement):
        if not self._uncorrected:
            self.__setSigmaPoints()

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
        predicted_measurement = numpy.zeros([update_size, 1])
        predicted_meas_covar = numpy.zeros([update_size, update_size])
        cross_covar = numpy.zeros([self.state.shape[0], update_size])

        sigma_point_measurements = [numpy.zeros([update_size, 1])
                for _ in xrange(len(self._sigma_points))]

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

        for idx in xrange(len(self._sigma_points)):
            sigma_point_measurements[idx] = state_to_measurement_subset.dot(
                    self._sigma_points[idx])
            predicted_measurement += (
                    self._state_weights[idx] * sigma_point_measurements[idx])

        for idx in xrange(len(self._sigma_points)):
            sigma_diff = sigma_point_measurements[idx] - predicted_measurement
            predicted_meas_covar += self._covar_weights[idx] * (
                    sigma_diff.dot(sigma_diff.T))
            cross_covar += self._covar_weights[idx] * (
                    (self._sigma_points[idx] - self.state).dot(sigma_diff.T))

        inv_innov_cov = numpy.linalg.inv(
                predicted_meas_covar + measurement_covariance_subset)
        kalman_gain_subset = cross_covar.dot(inv_innov_cov)
        innovation_subset = measurement_subset - predicted_measurement

        if self.checkMahalanobisThreshold(
                innovation_subset, inv_innov_cov,
                measurement.mahalanobis_threshold):
            for idx, iui in enumerate(update_indices):
                if iui >= StateMember.roll and iui <= StateMember.yaw:
                    innovation_subset[idx] = clampRotation(
                            innovation_subset[idx])
            self.state += kalman_gain_subset.dot(innovation_subset)

            self.estimate_error_covariance -= kalman_gain_subset.dot(
                    predicted_meas_covar).dot(
                            kalman_gain_subset.T)
            self._wrapStateAngles(
                    StateMember.roll, StateMember.pitch, StateMember.yaw)

            self._uncorrected = False

    def predict(self, delta):
        orientation = self.state[StateMember.roll:StateMember.yaw+1]
        roll, pitch, yaw = orientation.reshape(3)

        i_delta = delta * delta * 0.5
        rot = rpyToRotationMatrix(roll, pitch, yaw)
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

        self.__setSigmaPointsWithTF()

        self.state.fill(0.)
        for weight, point in zip(self._state_weights, self._sigma_points):
            self.state += weight * point
        self._wrapStateAngles(
                StateMember.roll, StateMember.pitch, StateMember.yaw)
        self._uncorrected = True

        self.estimate_error_covariance.fill(0.)
        for weight, point in zip(self._covar_weights, self._sigma_points):
            sigma_diff = point - self.state
            self.estimate_error_covariance += sigma_diff.dot(
                    sigma_diff.T) * weight

        self.estimate_error_covariance += self.process_noise_covariance * delta
