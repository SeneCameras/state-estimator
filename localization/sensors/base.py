class SensorBase(object):
    """Base class for simulating sensors.

    For descriptions we'll assume that the sensor has N measured states.

    Parameters
    ----------
    start_time: float
        Start time of the sensor usage, in seconds. Preferably set to 0.
    delta_time: float
        Interval between sensor responses, in seconds.
    covariance: numpy.ndarray
        An NxN array representing the covariance matrix of the sensor.
    measured_values: list(localization.util.StateMember)
        List of N states that are measured by the sensor.
    """
    def __init__(self, start_time, delta_time, covariance, measured_values):
        super(SensorBase, self).__init__()
        state_size = 15
        self.covariance = covariance
        self.next_measurement_time = start_time
        self.delta_time = delta_time
        self.update_vector = measured_values

    def getStateMeasurementsUntil(self, real_state, end_time):
        """Generate a sensor measurement if the final time is far enough.

        Parameters
        ----------
        real_state: numpy.ndarray
            A 15x1 array representing the actual state.
        end_time: Time until which we're waiting to get a measurement.

        Returns
        -------
        None
            If we didn't wait long enough, to measurements are generated
        localization.util.Measurement
            If we waited the appropriate time, a measurement is generated.

        Throws
        ------
        ValueError
            If we wait too long, we've skipped measurements.
        """
        if self.next_measurement_time > end_time:
            return None
        retval = self.generateMeasurement(real_state)
        if retval is None:
            return None
        retval.time = self.next_measurement_time
        self.next_measurement_time += self.delta_time
        if self.next_measurement_time <= end_time:
            message = 'Sensor needs to be called no less often than every {}s'
            raise ValueError(message.format(self.delta_time))
        return retval

    def generateMeasurement(self, real_state):
        """Generate a sensor measurement based on the given state.

        Unimplemented for base class.

        Parameters
        ----------
        real_state: numpy.ndarray
            A 15x1 array representing the actual state.

        Returns
        -------
        localization.util.Measurement
            Generate a measurement with added offsets, errors and noises.
        None
            In case that a measurement fails to get generated.
        """
        raise NotImplementedError("Please Implement this method")
