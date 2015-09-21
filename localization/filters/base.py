import math
import numpy
import Queue

from localization.util import clampRotation


class FilterBase(object):
    """Base class for sensor fusion filters.

    Attributes
    ----------
    state: numpy.ndarray
        Array containing the current state estimate. Size is 15x1.
    estimate_error_covariance: numpy.ndarray
        Covariance of the error estimation. Size is 15x15.
    process_noise_covariance: numpy.ndarray
        Covariance of process noise. Size is 15x15.
    last_measurement_time: float
        Time, in seconds, of the last passed measurement.
    """
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
        self._initialized = False
        self.sensor_timeout = 0.001 # Assume 100 Hz sensor data
        self.last_update_time = 0
        self.last_measurement_time = 0

        self._measurement_queue = Queue.PriorityQueue()

    def correct(self, measurement):
        """Correct the current state estimate.

        Unimplemented in the base filter.

        Parameters
        ----------
        measurement: localization.util.Measurement
            Measurement used to correct the current state.
        """
        raise NotImplementedError("Please Implement this method")

    def isInitialized(self):
        """Check if the filter is initialized.

        Unimplemented in the base filter.

        Returns
        -------
        bool
            Returns True if filter is initialized. Otherwise returns False.
        """
        return self._initialized

    def __initialize(self):
        """Mark the filter as initialized"""
        self._initialized = True

    def getPredictedState(self):
        """Get the current state estimate with applied predictions.

        The state changes with the passage of time.
        The filter's state attribute updates only on correction steps.

        Returns
        -------
        numpy.ndarray
            The estimated state with predictions applied.
        """
        return self._predicted_state

    def predict(self, delta):
        """Predict the state estimate after a certain time period passes.

        Unimplemented in the base filter.

        Parameters
        ----------
        delta: float
            Time in seconds since the last measurement.
        """
        raise NotImplementedError("Please Implement this method")

    def processMeasurement(self, measurement):
        """Perform a prediction and correction cycle.

        Process measured data by predicting movement within the time advance,
        followed by correcting the state based on the measured variables.


        Parameters
        ----------
        measurement: localization.util.Measurement
            Measurement used to correct the current state.
        """
        delta = 0.0
        if self.isInitialized():
            delta = measurement.time - self.last_measurement_time
            if delta > 0.0:
                self.predict(delta)
                self._predicted_state = self.state
            self.correct(measurement)
        else:
            for idx_1, param_1 in enumerate(measurement.update_vector):
                self.state[param_1] = measurement.measurement[idx_1]
                for idx_2, param_2 in enumerate(measurement.update_vector):
                    self.estimate_error_covariance[idx_1, idx_2] = (
                            measurement.covariance[idx_1, idx_2])
            self.__initialize()
        if delta >= 0.0:
            self.last_measurement_time = measurement.time

    def _wrapStateAngles(self, *args):
        """Clamp all passed variables as if they were angles

        Parameters
        ----------
        *args: localization.util.StateMember
            States that should be clamped to [-pi, pi].
        """
        for s in args:
            self.state[s] = clampRotation(self.state[s])

    def checkMahalanobisThreshold(self, innovation, inv_covariance, nsigmas):
        """Check if innovation is within the allowed Mahalanobis distance."""
        squared_mahalanobis = innovation.T.dot(inv_covariance).dot(innovation)
        threshold = nsigmas * nsigmas
        return squared_mahalanobis < threshold

    def enqueueMeasurement(self, measurement):
        """Add measurement to sorted queue of measurements to process."""
        self._measurement_queue.put(measurement)

    def integrateMeasurements(self):
        """Process all measurement in the measurement queue.

        Measurements are processed chronologically."""
        while not self._measurement_queue.empty():
            self.processMeasurement(self._measurement_queue.get())
