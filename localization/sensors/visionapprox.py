import numpy

import localization.sensors.base
from localization.util import Measurement, StateMember


class Vision(localization.sensors.base.SensorBase):
    """Simulator of the vision system.

    A 3-DOF sensor for linear velocity.
    The coordinates are, in order: forward, left, up.

    Simulator of the vision system with the ability to choose the reading
    frequency and covariance. Covariances along the first coordinate should
    preferably be set higher than the other two, since that coordinate is the
    hardest to estimate properly.

    Parameters
    ----------
    start_time: float
        Start time of the sensor usage, in seconds. Preferably set to 0.
    frequency: float
        Frequency of sensor responses, in Hz.
    covariance: numpy.ndarray
        A 3x3 array describing the vision covariance.
    """
    def __init__(self, start_time, frequency, covariance):
        super(Vision, self).__init__(
                start_time, 1. / frequency, numpy.asarray(covariance),
                [StateMember.v_x, StateMember.v_y, StateMember.v_z])

    def generateMeasurement(self, real_state):
        """Generate a vision measurement based on the given state.

        The measurement depends on the vision parameters we've set.

        Parameters
        ----------
        real_state: numpy.ndarray
            A 15x1 array representing the actual state.

        Returns
        -------
        localization.util.Measurement
            Generate a measurement with added offsets, errors and noises.
        """
        meas = numpy.asarray(
                real_state[StateMember.v_x:StateMember.v_z+1]).reshape(3)
        meas = numpy.random.multivariate_normal(meas, self.covariance)
        meas = numpy.asarray(meas).reshape([3, 1])
        return Measurement(0., meas, self.covariance, self.update_vector)
