import numpy

import localization.sensors.base
from localization.util import Measurement, StateMember


class Vision(localization.sensors.base.SensorBase):
    """Simulator of the vision system.

    Simulator of the vision system with the ability to choose the reading
    frequency and covariance.
    """
    def __init__(self, start_time, frequency, covariance):
        covar = numpy.zeros([15, 15])
        covar[6:9, 6:9] = covariance
        super(Vision, self).__init__(
                start_time, 1. / frequency, covar,
                [StateMember.v_x, StateMember.v_y, StateMember.v_z])

    def generateMeasurement(self, real_state):
        meas = numpy.asarray(real_state).reshape(15)
        meas = numpy.random.multivariate_normal(meas, self.covariance)
        meas = numpy.asarray(meas).reshape([15, 1])
        return Measurement(0., meas, self.covariance, self.update_vector)
