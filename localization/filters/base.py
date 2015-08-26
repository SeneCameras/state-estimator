import math
import numpy


class StateMember:
    x = 0
    y = 1
    z = 2
    roll = 3
    pitch = 4
    yaw = 5
    v_x = 6
    v_y = 7
    v_z = 8
    v_roll = 9
    v_pitch = 10
    v_yaw = 11
    a_x = 12
    a_y = 13
    a_z = 14


def clampRotation(angle):
    while (angle > math.pi):
        angle -= 2.0 * math.pi
    while (angle < math.pi):
        angle += 2.0 * math.pi
    return angle


class Measurement(object):
    """Structure for storing measurements"""
    def __init__(self, time, measurement, covariance, update_vector):
        super(Measurement, self).__init__()
        self.measurement = measurement
        self.covariance = covariance
        self.update_vector = update_vector
        self.time = time
        self.mahalanobis_threshold = float('inf')

    """Override of greater-than operator for sorting purposes"""
    def __gt__(self, other):
        return self.time > other.time


class FilterBase(object):
    """Base class for filters"""
    def __init__(self):
        super(FilterBase, self).__init__()
        state_size = 15
        self.state = numpy.zeros([state_size, 1])
        self._predicted_state = numpy.zeros([state_size, 1])
        self._transfer_function = numpy.identity(state_size)
        self._transfer_function_jacobian = numpy.zeros([state_size, state_size])
        self.estimate_error_covariance = numpy.identity(state_size) * 1e-9
        self._covariance_epsilon = numpy.identity(state_size) * 0.001
        self.process_noise_covariance = numpy.diag([
                0.05, 0.05, 0.06, 0.03, 0.03, 0.06, 0.025, 0.025, 0.04,
                0.01, 0.01, 0.02, 0.01, 0.01, 0.015
        ])
        self._identity = numpy.identity(state_size)
        self._initialized = False
        self.sensor_timeout = 0.001 # Assume 100 Hz sensor data
        self.last_update_time = 0
        self.last_measurement_time = 0

    def correct(self, measurement):
        raise NotImplementedError("Please Implement this method")

    def isInitialized(self):
        return self._initialized

    def __initialize(self):
        self._initialized = True

    def getPredictedState(self):
        return self._predicted_state

    def predict(self, delta):
        raise NotImplementedError("Please Implement this method")

    def processMeasurement(self, measurement):
        delta = 0.0
        if self.isInitialized():
            delta = measurement.time - self.last_measurement_time
            if delta > 0.0:
                self.predict(delta)
                self._predicted_state = self.state
            self.correct(measurement)
        else:
            for idx, cond in enumerate(measurement.update_vector):
                if cond:
                    self.state[idx] = measurement.measurement[idx]
                    for idy, cond2 in enumerate(measurement.update_vector):
                        if cond2:
                            self.estimate_error_covariance[idx, idy] = (
                                    measurement.covariance[idx, idy])
            self.__initialize()
        if delta >= 0.0:
            self.last_measurement_time = measurement.time

    def _wrapStateAngles(self, *args):
        for s in args:
            self.state[s] = clampRotation(self.state[s])

    def checkMahalanobisThreshold(self, innovation, inv_covariance, nsigmas):
        squared_mahalanobis = innovation.T.dot(inv_covariance).dot(innovation)
        threshold = nsigmas * nsigmas
        return squared_mahalanobis < threshold
