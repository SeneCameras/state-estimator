import math
import numpy

import localization.sensors.base
import localization.filters.ahrs
from localization.util import Measurement, StateMember, rpyToRotationMatrix


def rotateZAxisByQuaternion(q):
    """Get vector representing Z axis rotated by given quaternion.

    Calculated via p.v.p* formula.

    Parameters
    ----------
    q: list(float)
        List of 4 floats, representing quaternion coordinates.

    Returns
    -------
    numpy.ndarray
        A 3x1 array representing the rotated Z axis unit vector.
    """
    a, b, c, d = q
    return numpy.array([
        [2. * (b * d - a * c)],
        [2. * (c * d + a * b)],
        [a**2 - b**2 - c**2 + d**2]])


class ImuCompensated(localization.sensors.base.SensorBase):
    """Simulator for an IMU and AHRS combination.

    Combine output from an IMU simulator with a provided gyro drift and
    autoadjusting gravity estimate.

    Parameters
    ----------
    start_time: float
        Start time of the sensor usage, in seconds. Preferably set to 0.
    frequency: float
        Frequency of sensor responses, in Hz.
    ahrs_name: {'Madgwick', 'Mahony'}
        AHRS algorithm used for attitude and heading estimation.
    gyro_drift: numpy.ndarray, optional
        Drift of the gyroscope. Default value is no drift. 3x1 array.
    gravity_drift: float, optional
        Accelerometer error caused by gravity. Default value is no gravity.

    Attributes
    ----------
    ahrs: localization.filters.ahrs.Ahrs
        Instance of the AHRS estimator.
    sensor: InvensenseMPU9250
        Instance of the IMU sensor.
    gyro_compensation: numpy.ndarray
        Compensation of the gyro drift.
        Not set by the constructor, only manually.
        By default set to a 3x1 array of zeros.
    gravity: numpy.ndarray.
        Estimated gravity affecting the system.
        Not set by the constructor, only manually.
        Can be corrected over time by an IIR filter.
        By default set to zero.
    _gravity_correction_per_second: float
        Fraction of correction per second for the gravity IIR filter.
        Should be set to extremely small values, since gravity only changes
        with great changes in geographic location.
        By default set to zero.
        With g_e estimated gravity, and g_m measured gravity, the filter is:
        gravity_correction_per_second = 1 - (1-r)**frequency
        g_e[n + 1] = g_e[n] * (1 - k) + g_m * k

    Throws
    ------
    ValueError
        In case the AHRS name isn't either "Madgwick" or "Mahony".
    """
    def __init__(self, start_time, frequency, ahrs_name,
                 gyro_drift = None, gravity_drift = None):
        if gyro_drift is None:
            gyro_drift = numpy.zeros([3, 1])
        if gravity_drift is None:
            gravity_drift = 0.
        if ahrs_name is 'Madgwick':
            self.ahrs = localization.filters.ahrs.Madgwick(frequency)
        elif ahrs_name is 'Mahony':
            self.ahrs = localization.filters.ahrs.Mahony(frequency)
        else:
            raise ValueError('AHRS name must be either "Madgwick" or "Mahony"')
        self.sensor = InvensenseMPU9250(
                start_time, frequency, gyro_drift, gravity_drift)
        self.setGyroCompensation(*numpy.array(gyro_drift).flatten())
        self.setGravity(gravity_drift)
        self.setGravityCorrectionRate(0.)

        super(ImuCompensated, self).__init__(
                start_time, 1. / frequency, self.sensor.covariance,
                [StateMember.v_roll, StateMember.v_pitch, StateMember.v_yaw,
                 StateMember.a_x, StateMember.a_y, StateMember.a_z])

    def setGyroCompensation(self, roll, pitch, yaw):
        """Set the estimated gyro drift along each axis.

        These values are subtracted from gyro readings when generating output.

        Parameters
        ----------
        roll: float
            Estimated drift along the local X axis.
        pitch: float
            Estimated drift along the local Y axis.
        yaw: float
            Estimated drift along the local Z axis.
        """
        self.gyro_compensation = numpy.array(
                [roll, pitch, yaw]).reshape([3, 1])

    def setGravity(self, gravity):
        """Set the estimated gravity.

        This value is multiplied by the current attitude/heading, and then
        subtracted from accelerometer readings when generating output.
        """
        self.gravity = gravity

    def setGravityCorrectionRate(self, rate):
        """Set the rate at which gravity is corrected towards current readings.

        The used filter depends on frequency, and the correction k is
        gravity_correction_per_second = 1 - (1-k)**frequency

        Parameters
        ----------
        rate: float
            Number inside the [0, 1] range. Represents the percentage of a
            current measure being used for gravity correction.
        """
        self._gravity_correction_per_second = rate
        self._k = 1. - (1. - rate)**(1. / self.ahrs.frequency)

    def generateMeasurement(self, real_state):
        """Generate a corrected IMU measurement based on the given state.

        A compensation of gravity and gyro drift is attempted with an AHRS,
        IIR filter and given static drifts.

        Parameters
        ----------
        real_state: numpy.ndarray
            A 15x1 array representing the actual state.

        Returns
        -------
        localization.util.Measurement
            Generate a measurement with compensated offsets, errors,
            and with added noises.
        """
        # Generate noisy and biased IMU measurement.
        measurement = self.sensor.generateMeasurement(real_state)

        # Get accelerometer measurement magnitude
        accel = measurement.measurement[3:6]
        accel_norm = numpy.linalg.norm(accel)

        # Compensate the gyro bias.
        gyro = measurement.measurement[0:3]
        measurement.measurement[0:3] -= self.gyro_compensation

        # Compensate the gravity impact.
        self.ahrs.update(gyro, accel)
        g = rotateZAxisByQuaternion(self.ahrs.q.reshape(4).tolist())
        measurement.measurement[3:6] -= g * self.gravity

        # Adjust gravity
        self.setGravity(self.gravity * (1 - self._k) + accel_norm * self._k)

        # Return adjusted Measurement
        return measurement


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
