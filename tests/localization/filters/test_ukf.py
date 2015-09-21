import numpy
import unittest

import localization.filters.ukf
import localization.util


class TestUkf(unittest.TestCase):
    def test_measurement_struct(self):
        ekf = localization.filters.ukf.Ukf()
        initial_covariance = numpy.identity(15) * 0.5
        ekf.estimate_error_covariance = initial_covariance.copy()
        measurement = numpy.array([[0.15 * i] for i in xrange(15)])
        measurement_covariance = numpy.identity(15) * 1e-9
        update_vector = range(15)
        meas = localization.util.Measurement(
                1000., measurement, measurement_covariance, update_vector)
        ekf.processMeasurement(meas)

        self.assertSequenceEqual((15, 1), measurement.shape)
        self.assertSequenceEqual(measurement.shape, ekf.state.shape)
        self.assertSequenceEqual(measurement[0], ekf.state[0])

        ekf.estimate_error_covariance = initial_covariance.copy()

        measurement *= 2.
        meas = localization.util.Measurement(
                1002., measurement, measurement_covariance, update_vector)
        self.assertSequenceEqual((15, 1), measurement.shape)
        self.assertSequenceEqual(measurement.shape, ekf.state.shape)
        for value, state in zip(measurement[0], ekf.state[0]):
            self.assertAlmostEqual(value, state, delta=0.001)


if __name__ == '__main__':
    unittest.main()
