import math
import numpy
import Queue

from localization.util import clampRotation


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

        self._measurement_queue = Queue.PriorityQueue()

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

    def enqueueMeasurement(self, measurement):
        self._measurement_queue.put(measurement)

    def integrateMeasurements(self):
        while not self._measurement_queue.empty():
            self.processMeasurement(self._measurement_queue.get())
