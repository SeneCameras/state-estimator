import numpy
import unittest

import localization.sensors.base
import localization.util


class SensorDerived(localization.sensors.base.SensorBase):
    def __init__(self, test_instance, start_time):
        covariance = numpy.diag([0.] * 3 + [1.5, 2.] + [0.] * 10)
        super(SensorDerived, self).__init__(start_time, 1., covariance, [3, 4])
        self.test_instance = test_instance

    def generateMeasurement(self, real_state):
        return localization.util.Measurement(
                0., self.covariance.dot(real_state),
                self.covariance, self.update_vector)


class TestSensorBase(unittest.TestCase):
    def test_constructor(self):
        derived = SensorDerived(self, 10.)
        self.assertSequenceEqual(
                [False] * 3 + [True] * 2 + [False] * 10, derived.update_vector)
        self.assertAlmostEqual(10., derived.next_measurement_time)

    def test_get_state_measurements_until(self):
        derived = SensorDerived(self, 10.)
        self.assertAlmostEqual(10., derived.next_measurement_time)

        meas = derived.getStateMeasurementsUntil(
                numpy.arange(15.).reshape([15, 1]), 10.3)
        self.assertIsNotNone(meas)
        self.assertSequenceEqual([15, 1], meas.measurement.shape)
        self.assertAlmostEqual(4.5, meas.measurement[3, 0])
        self.assertAlmostEqual(8., meas.measurement[4, 0])
        self.assertAlmostEqual(11., derived.next_measurement_time)

        meas = derived.getStateMeasurementsUntil(
                numpy.arange(2., 17.).reshape([15, 1]), 10.6)
        self.assertIsNone(meas)
        self.assertAlmostEqual(11., derived.next_measurement_time)

        meas = derived.getStateMeasurementsUntil(
                numpy.arange(4., 19.).reshape([15, 1]), 11.2)
        self.assertIsNotNone(meas)
        self.assertSequenceEqual([15, 1], meas.measurement.shape)
        self.assertAlmostEqual(10.5, meas.measurement[3, 0])
        self.assertAlmostEqual(16., meas.measurement[4, 0])
        self.assertAlmostEqual(12., derived.next_measurement_time)


if __name__ == '__main__':
    unittest.main()
