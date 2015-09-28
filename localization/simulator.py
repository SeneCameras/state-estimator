import numpy

import localization.util
import localization.filters.ekf
import localization.filters.ukf
import localization.sensors.imu
import localization.sensors.visionapprox


def getSimulationUsecase(filter_name, vision_freq, vision_covariance_diagonal):
    """Generate a setup of filters and vision quality for a simulation.

    Parameters
    ----------
    filter_name: {'EKF, UKF'}
        Choice of either using Extended or Unscented Kalman Filters.
    vision_freq: float
        Frequency at which the vision system gives measurements.
    vision_covariance_diagonal: list-like
        Variance along each axis, listed as: forwards, sideways, vertical

    Returns
    -------
    Simulator
        Simulator containing:
        * EKF/UKF sensor fusion
        * 4kHz InvensenseMPU9250 6DOF IMU
        * A vision system estimation defined by the given parameters
    """
    if filter_name is 'EKF':
        filtering = localization.filters.ekf.Ekf()
    elif filter_name is 'UKF':
        filtering = localization.filters.ukf.Ukf()
    else:
        raise ValueError('Filter name must be either "EKF" or "UKF"')
    vision_covariance = numpy.diag(vision_covariance_diagonal)
    sensors = [
        localization.sensors.imu.InvensenseMPU9250(0., 4000.),
        localization.sensors.visionapprox.Vision(
                0., vision_freq, vision_covariance)]
    return Simulator(filtering, sensors)


class Simulator(object):
    """Simulation of sensor fusion.

    Simulates sensor fusion using a filter and multiple sensors.

    Parameters
    ----------
    filtering: localization.filters.base.FilterBase
        A sensor fusion filter, used to estimate state.
    sensors: list(localization.sensors.base.SensorBase)
        A list of sensor simulators, used to estimate state.

    Attributes
    ----------
    filtering
    sensors
    delta_time: float
        Interval of one time tick. Default is the smallest sensor interval.
    next_tick: float
        Time at which the next sensor tick will occur.
    p: numpy.ndarray
        A 3x1 array representing the current position base truth.
    phi: numpy.ndarray
        A 3x1 array representing the current orientation RPY base truth.
    v: numpy.ndarray
        A 3x1 array representing the current linear velocity base truth.
    """
    def __init__(self, filtering, sensors):
        super(Simulator, self).__init__()
        self.filtering = filtering
        self.sensors = sensors
        self.delta_time = min(sensor.delta_time for sensor in sensors)
        self.next_tick = min(
                s.next_measurement_time for s in sensors) + self.delta_time / 2.

        self.p = numpy.zeros([3, 1])
        self.phi = numpy.zeros([3, 1])
        self.v = numpy.zeros([3, 1])

    def tick(self, v, w):
        """Tick simulation by the set time interval.

        The simulation tick receives the current linear and angular velocity.

        Parameters
        ----------
        v: numpy.ndarray
            A 3x1 array representing the new linear velocity.
        w: numpy.ndarray
            A 3x1 array representing the new angular velocity.
        """

        v = numpy.asarray(v).reshape([3, 1])
        w = numpy.asarray(w).reshape([3, 1])
        a = (v - self.v) / self.delta_time
        rotation_matrix = localization.util.rpyToRotationMatrix(
                self.phi[0, 0], self.phi[1, 0], self.phi[2, 0])
        self.p += rotation_matrix.dot(self.v) * self.delta_time
        self.p += rotation_matrix.dot(a) * (self.delta_time**2 * 0.5)
        self.phi += w * self.delta_time
        self.v = v

        state = numpy.append(self.p, self.phi, axis=0)
        state = numpy.append(state, self.v, axis=0)
        state = numpy.append(state, w, axis=0)
        state = numpy.append(state, a, axis=0)

        for sensor in self.sensors:
            reading = sensor.getStateMeasurementsUntil(state, self.next_tick)
            if reading is None:
                continue
            self.filtering.enqueueMeasurement(reading)
        self.filtering.integrateMeasurements()
        self.next_tick += self.delta_time

    def getFrameBaseTruth(self):
        """Get the base truth of the current frame.

        Returns
        -------
        dict(numpy.ndarray)
            A dict with 3 3x1 arrays for 'position', 'rpy' and 'velocity'.
        """
        return {
            'position': self.p.copy(),
            'rpy': self.phi.copy(),
            'velocity': self.v.copy(),
        }

    def getFrameEstimate(self):
        """Get sensor fusion estimate of the current frame.

        Returns
        -------
        dict(numpy.ndarray)
            A dict with 3 3x1 arrays for 'position', 'rpy' and 'velocity'.
        """
        return {
            'position': self.filtering.state[0:3, 0:1],
            'rpy': self.filtering.state[3:6, 0:1],
            'velocity': self.filtering.state[6:9, 0:1],
        }
