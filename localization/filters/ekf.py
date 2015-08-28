import math
import numpy

import localization.filters.base
from localization.util import clampRotation, StateMember


def sinCosToRotationMatrix(cr, cp, cy, sr, sp, sy):
    return numpy.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]])


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
                state_to_measurement_subset[idx, iui] = 1.0
                if iui >= StateMember.roll and iui <= StateMember.yaw:
                    innovation_subset[idx] = clampRotation(
                            innovation_subset[idx])
            self.state = kalman_gain_subset.dot(innovation_subset)
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
        (x, y, z, roll, pitch, yaw, v_x, v_y, v_z, v_roll, v_pitch, v_yaw,
            a_x, a_y, a_z) = self.state

        cr = math.cos(roll)
        cp = math.cos(pitch)
        cy = math.cos(yaw)
        sr = math.sin(roll)
        sp = math.sin(pitch)
        sy = math.sin(yaw)

        i_delta = delta * delta * 0.5
        rot = sinCosToRotationMatrix(cr, cp, cy, sr, sp, sy)
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

        c_x = 0.
        c_y = cy * sp * cr + sy * sr
        c_z = -cy * sp * sr + sy * cr
        dF0dr = ((c_y * v_y + c_z * v_z) * delta +
                 (c_y * a_y + c_z * a_z) * i_delta)
        dF6dr = 1. + (c_y * v_roll + c_z * v_yaw) * delta

        c_x = -cy * sp
        c_y = cy * cp * sr
        c_z = cy * cp * cr
        dF0dp = ((c_x * v_x + c_y * v_y + c_z * v_z) * delta +
                 (c_x * a_x + c_y * a_y + c_z * a_z) * i_delta)
        dF6dp = (c_x * v_roll + c_y * v_pitch + c_z * v_yaw) * delta

        c_x = -sy * cp
        c_y = -sy * sp * sr - cy * cr
        c_z = -sy * sp * cr + cy * sr
        dF0dy = ((c_x * v_x + c_y * v_y + c_z * v_z) * delta +
                 (c_x * a_x + c_y * a_y + c_z * a_z) * i_delta)
        dF6dy = (c_x * v_roll + c_y * v_pitch + c_z * v_yaw) * delta

        c_y = sy * sp * cr - cy * sr
        c_z = -sy * sp * sr - cy * cr
        dF1dr = ((c_y * v_y + c_z * v_z) * delta +
                 (c_y * a_y + c_z * a_z) * i_delta)
        dF7dr = (c_y * v_pitch + c_z * v_yaw) * delta

        c_x = -sy * sp
        c_y = sy * cp * sr
        c_z = sy * cp * cr
        dF1dp = ((c_x * v_x + c_y * v_y + c_z * v_z) * delta +
                 (c_x * a_x + c_y * a_y + c_z * a_z) * i_delta)
        dF7dp = 1. + (c_x * v_roll + c_y * v_pitch + c_z * v_yaw) * delta

        c_x = cy * cp
        c_y = cy * sp * sr - sy * cr
        c_z = cy * sp * cr + sy * sr
        dF1dy = ((c_x * v_x + c_y * v_y + c_z * v_z) * delta +
                 (c_x * a_x + c_y * a_y + c_z * a_z) * i_delta)
        dF7dy = (c_x * v_roll + c_y * v_pitch + c_z * v_yaw) * delta

        c_y = cp * cr
        c_z = -cp * sr
        dF2dr = ((c_y * v_y + c_z * v_z) * delta +
                 (c_y * a_y + c_z * a_z) * i_delta)
        dF8dr = (c_y * v_pitch + c_z * v_yaw) * delta

        c_x = -cp
        c_y = -sp * sr
        c_z = -sp * cr
        dF2dp = ((c_x * v_x + c_y * v_y + c_z * v_z) * delta +
                 (c_x * a_x + c_y * a_y + c_z * a_z) * i_delta)
        dF8dp = (c_x * v_roll + c_y * v_pitch + c_z * v_yaw) * delta

        self._transfer_function_jacobian = self._transfer_function.copy();
        dF2dy = self._transfer_function_jacobian[
                StateMember.z, StateMember.roll]
        dF8dy = self._transfer_function_jacobian[
                StateMember.z, StateMember.roll]
        self._transfer_function_jacobian[StateMember.x:StateMember.z+1,
                                         StateMember.roll:StateMember.yaw+1] = [
                [dF0dr, dF0dp, dF0dy],
                [dF1dr, dF1dp, dF1dy],
                [dF2dr, dF2dp, dF2dy]]
        self._transfer_function_jacobian[StateMember.roll:StateMember.yaw+1,
                                         StateMember.roll:StateMember.yaw+1] = [
                [dF6dr, dF6dp, dF6dy],
                [dF7dr, dF7dp, dF7dy],
                [dF8dr, dF8dp, dF8dy]]

        self.state = self._transfer_function.dot(self.state)
        self._wrapStateAngles(
                StateMember.roll, StateMember.pitch, StateMember.yaw)

        self.estimate_error_covariance = self._transfer_function_jacobian.dot(
                self.estimate_error_covariance).dot(
                        self._transfer_function_jacobian.T)
        self.estimate_error_covariance += self.process_noise_covariance * delta
