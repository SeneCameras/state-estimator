import numpy

import localization.sensors.base
from localization.util import Measurement, StateMember


class Vision(localization.sensors.base.SensorBase):
    """Simulator of the vision system.

    Simulator of the vision system with the ability to choose the reading
    frequency and covariance.
    """
    def __init__(self, start_time, frequency, covariance):
        super(Vision, self).__init__(
                start_time, 1. / frequency, covariance,
                [StateMember.v_x, StateMember.v_y, StateMember.v_z])

    def generateMeasurement(self, real_state):
        meas = [real_state[i, 0] for i in self.measured_values]
        meas = numpy.random.multivariate_normal(meas, self.covariance)
        meas = numpy.asarray(meas).reshape([len(self.measured_values), 1])
        return Measurement(0., meas, self.covariance, self.update_vector)
