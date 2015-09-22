import math
import numpy

import localization.sensors.base
from localization.util import Measurement, StateMember, rpyToRotationMatrix


class InvensenseMPU9250(localization.sensors.base.SensorBase):
    """Simulator for the Invensense MPU 9250.

    A 6-DOF sensor for angular velocity and linear acceleration.

    All covariances are set based on the sensor's datasheet.

    Allows setting of optional gyro drift and gravity intensity.

    Parameters
    ----------
    start_time: float
        Start time of the sensor usage, in seconds. Preferably set to 0.
    frequency: float
        Frequency of sensor responses, in Hz.
    gyro_drift: numpy.ndarray, optional
        Drift of the gyroscope. Default value is no drift. 3x1 array.
    gravity_drift: float, optional
        Accelerometer error caused by gravity. Default value is no gravity.

    Attributes
    ----------
    gyro_drift
    gravity_vector: numpy.ndarray
        Stores vector representing the gravitational force.
    """
    def __init__(self, start_time, frequency,
                 gyro_drift = None, gravity_drift = None):
        if gyro_drift is None:
            gyro_drift = numpy.zeros([3, 1])
        if gravity_drift is None:
            gravity_drift = 0.
        delta_time = 1. / frequency

        g = 9.80665
        deg_to_rad = math.pi / 180.
        # Based on datasheet:
        npsd_gyro = 0.01 * deg_to_rad
        npsd_accel = 300e-6 * g

        self.gravity_vector = numpy.zeros([3, 1])
        self.gravity_vector[2, 0] = gravity_drift
        self.gyro_drift = numpy.array(gyro_drift).reshape([3, 1])

        covariance = numpy.diag([npsd_gyro] * 3 + [npsd_accel] * 3) * frequency
        super(InvensenseMPU9250, self).__init__(
                start_time, 1. / frequency, covariance,
                [StateMember.v_roll, StateMember.v_pitch, StateMember.v_yaw,
                 StateMember.a_x, StateMember.a_y, StateMember.a_z])

    def generateMeasurement(self, real_state):
        """Generate an IMU measurement based on the given state.

        Gravity and gyro drift are added to this measurement.

        Parameters
        ----------
        real_state: numpy.ndarray
            A 15x1 array representing the actual state.

        Returns
        -------
        localization.util.Measurement
            Generate a measurement with added offsets, errors and noises.
        """
        # Apply noise.
        meas = numpy.asarray(
                real_state[StateMember.v_roll:StateMember.a_z+1]).reshape(6)
        meas = numpy.random.multivariate_normal(meas, self.covariance)
        meas = numpy.asarray(meas).reshape([6, 1])
        # Apply gyro drift.
        meas[0:3] += self.gyro_drift
        # Apply gravity.
        rpy = real_state[StateMember.roll:StateMember.yaw+1, 0].tolist()
        meas[3:6] += rpyToRotationMatrix(*rpy).dot(self.gravity_vector)

        return Measurement(0., meas, self.covariance, self.update_vector)
