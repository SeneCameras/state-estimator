import numpy

import localization.util
import localization.filters.ekf
import localization.filters.ukf
import localization.sensors.imu
import localization.sensors.vision


def getSimulationUsecase(filter_name, vision_freq, vision_covariance_diagonal):
    if filter_name is 'EKF':
        filtering = localization.filters.ekf.Ekf()
    elif filter_name is 'UKF':
        filtering = localization.filters.ukf.Ukf()
    else:
        raise ValueError('Filter name must be either "EKF" or "UKF"')
    vision_covariance = numpy.diag(vision_covariance_diagonal)
    sensors = [
        localization.sensors.imu.InvensenseMPU9250(0., 4000.),
        localization.sensors.vision.Vision(0., vision_freq, vision_covariance)]
    return Simulator(filtering, sensors)


class Simulator(object):
    """Simulator for the desired scenario."""
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
        return {
            'position': self.p.copy(),
            'rpy': self.phi.copy(),
            'velocity': self.v.copy(),
        }

    def getFrameEstimate(self):
        return {
            'position': self.filtering.state[0:3, 0:1],
            'rpy': self.filtering.state[3:6, 0:1],
            'velocity': self.filtering.state[6:9, 0:1],
        }
