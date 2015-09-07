import math
import numpy
import unittest

import localization.filters.ahrs


class TestAhrs(unittest.TestCase):
    def test_constructor(self):
        filt = localization.filters.ahrs.Ahrs(1000.)
        self.assertSequenceEqual((4, 1), filt.q.shape)
        self.assertSequenceEqual((1., 0., 0., 0.), filt.q.reshape(4).tolist())
        self.assertEqual(1000., filt.frequency)

    def test_get_rpy(self):
        filt = localization.filters.ahrs.Ahrs(1000.)
        rpy = filt.getRPY()
        self.assertSequenceEqual((3, 1), rpy.shape)
        for wanted, real in zip((0., 0., 0.), rpy.reshape(3).tolist()):
            self.assertAlmostEqual(wanted, real)

        sin_pi_quar = math.sin(math.pi / 8.)
        cos_pi_quar = math.cos(math.pi / 8.)

        filt.q = numpy.array([cos_pi_quar, sin_pi_quar, 0., 0.]).reshape(4, 1)
        rpy = filt.getRPY()
        self.assertSequenceEqual((3, 1), rpy.shape)
        angles = (math.pi / 4., 0., 0.)
        for wanted, real in zip(angles, rpy.reshape(3).tolist()):
            self.assertAlmostEqual(wanted, real)

        filt.q = numpy.array([cos_pi_quar, 0., sin_pi_quar, 0.]).reshape(4, 1)
        rpy = filt.getRPY()
        self.assertSequenceEqual((3, 1), rpy.shape)
        angles = (0., math.pi / 4., 0.)
        for wanted, real in zip(angles, rpy.reshape(3).tolist()):
            self.assertAlmostEqual(wanted, real)

        filt.q = numpy.array([cos_pi_quar, 0., 0., sin_pi_quar]).reshape(4, 1)
        rpy = filt.getRPY()
        self.assertSequenceEqual((3, 1), rpy.shape)
        angles = (0., 0., math.pi / 4.)
        for wanted, real in zip(angles, rpy.reshape(3).tolist()):
            self.assertAlmostEqual(wanted, real)

    def test_madgwick(self):
        madg = localization.filters.ahrs.Madgwick(1000.)

        for _ in xrange(10000):
            madg.update(numpy.zeros([3, 1]),
                        numpy.array([1., 0., 1.]).reshape(3, 1))
        roll, pitch, yaw = madg.getRPY().reshape(3).tolist()
        self.assertAlmostEqual(0., roll, delta=0.01)
        self.assertAlmostEqual(-math.pi / 4., pitch, delta=0.01)

        for _ in xrange(10000):
            madg.update(numpy.zeros([3, 1]),
                        numpy.array([0., 1., 1.]).reshape(3, 1))
        roll, pitch, yaw = madg.getRPY().reshape(3).tolist()
        self.assertAlmostEqual(0., pitch, delta=0.01)
        self.assertAlmostEqual(math.pi / 4., roll, delta=0.01)

if __name__ == '__main__':
    unittest.main()
