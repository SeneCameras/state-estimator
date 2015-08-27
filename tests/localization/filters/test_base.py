import numpy
import unittest

import localization.filters.base


class FilterDerived(localization.filters.base.FilterBase):
    def __init__(self, test_instance):
        super(FilterDerived, self).__init__()
        self.val = 0.
        self.test_instance = test_instance

    def correct(self, measurement):
        self.test_instance.assertEqual(
                self.val, measurement.time - self.last_measurement_time)
        self.test_instance.assertEqual(15, len(measurement.update_vector))
        self.test_instance.assertSequenceEqual(
                [True] * 15, measurement.update_vector)

    def predict(self, delta):
        self.val = delta


class TestFilterBase(unittest.TestCase):
    def test_measurement_struct(self):
        meas1 = localization.filters.base.Measurement(0., None, None, None)
        meas2 = localization.filters.base.Measurement(0., None, None, None)
        self.assertEqual(0., meas1.time)
        self.assertEqual(0., meas2.time)
        self.assertFalse(meas1 > meas2)
        self.assertFalse(meas2 > meas1)
        meas1.time = 100
        meas2.time = 200
        self.assertFalse(meas1 > meas2)
        self.assertTrue(meas2 > meas1)

    def test_derived_filter_get_set(self):
        derived = FilterDerived(self)
        self.assertFalse(derived.isInitialized())

    def test_measurement_process(self):
        derived = FilterDerived(self)

        measurement = numpy.array([[0.1 * i] for i in xrange(15)])
        measurement_covariance = numpy.array(
                [[0.1 * i * j for j in xrange(15)] for i in xrange(15)])

        meas = localization.filters.base.Measurement(
                1000., measurement, measurement_covariance, [True] * 15)

        self.assertFalse(derived.isInitialized())

        derived.processMeasurement(meas)

        self.assertTrue(derived.isInitialized())
        self.assertSequenceEqual((15, 1), measurement.shape)
        self.assertSequenceEqual(measurement.shape, derived.state.shape)
        self.assertSequenceEqual(measurement[0], derived.state[0])

        meas.time = 1002.
        derived.processMeasurement(meas)

        self.assertEqual(1002., derived.last_measurement_time)


if __name__ == '__main__':
    unittest.main()
