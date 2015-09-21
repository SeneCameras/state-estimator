import math
import numpy

import localization.sensors.base
from localization.util import Measurement, StateMember


class InvensenseMPU9250(localization.sensors.base.SensorBase):
    def __init__(self, start_time, frequency):
        delta_time = 1. / frequency

        g = 9.80665
        deg_to_rad = math.pi / 180.
        # Based on datasheet:
        npsd_gyro = 0.01 * deg_to_rad
        npsd_accel = 300e-6 * g

        covariance = numpy.diag([npsd_gyro] * 3 + [npsd_accel] * 3) * frequency
        super(InvensenseMPU9250, self).__init__(
                start_time, 1. / frequency, covariance,
                [StateMember.v_roll, StateMember.v_pitch, StateMember.v_yaw,
                 StateMember.a_x, StateMember.a_y, StateMember.a_z])

    def generateMeasurement(self, real_state):
        # Initial model, without the gravity vector taken into account
        meas = numpy.asarray(
                real_state[StateMember.v_roll:StateMember.a_z+1]).reshape(6)
        meas = numpy.random.multivariate_normal(meas, self.covariance)
        meas = numpy.asarray(meas).reshape([6, 1])
        return Measurement(0., meas, self.covariance, self.update_vector)
