class SensorBase(object):
    """Base class for simulating sensors"""
    def __init__(self, start_time, delta_time, covariance, measured_values):
        super(SensorBase, self).__init__()
        state_size = 15
        self.covariance = covariance
        self.next_measurement_time = start_time
        self.delta_time = delta_time
        self.update_vector = measured_values

    def getStateMeasurementsUntil(self, real_state, end_time):
        if self.next_measurement_time > end_time:
            return None
        retval = self.generateMeasurement(real_state)
        retval.time = self.next_measurement_time
        self.next_measurement_time += self.delta_time
        if self.next_measurement_time <= end_time:
            message = 'Sensor needs to be called no less often than every {}s'
            raise ValueError(message.format(self.delta_time))
        return retval

    def generateMeasurement(self, real_state):
        # Leaving measurement generation based on state to be fully open to
        # using any distribution for generating noise and drifts.
        raise NotImplementedError("Please Implement this method")
