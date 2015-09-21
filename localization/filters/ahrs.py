import math
import numpy


class Ahrs(object):
    """Base class for AHRS.

    Parameters
    ----------
    frequency: float
        Number of performed updates per second.

    Attributes
    ----------
    frequency
    """
    def __init__(self, frequency):
        super(Ahrs, self).__init__()
        self.q = numpy.array([1., 0., 0., 0.]).reshape([4, 1])
        self.frequency = frequency

    def getRPY(self):
        """Get RPY angles based on the current quaternion.

        Returns
        -------
        numpy.ndarray
            The roll, pitch and yaw, respectively, in a 3x1 array.
        """
        w, x, y, z = self.q.reshape(4).tolist()
        # source: https://goo.gl/gvEZnt
        return numpy.array([[
                math.atan2(y * z + w * x, 0.5 - y**2 - x**2)
            ], [
                math.asin(2 * (y * w - x * z))
            ], [
                math.atan2(x * y + w * z, 0.5 - y**2 - z**2)
            ]])

    def update(self, g, a):
        """Update attitude and heading based on measurements.

        Not implemented in base class.

        Parameters
        ----------
        g: numpy.ndarray
            Angular velocity read by a gyroscope. Size 3x1.
        a: numpy.ndarray
            Linear acceleration read by an accelerometer. Size 3x1.
        """
        raise NotImplementedError("Please Implement this method")


class Madgwick(Ahrs):
    """Madgwick AHRS.

    Parameters
    ----------
    frequency: float
        Number of performed updates per second.

    Attributes
    ----------
    frequency
    beta: float
        Weight of correction caused by the gravity reading.
    """
    def __init__(self, frequency):
        super(Madgwick, self).__init__(frequency)
        self.beta = 0.1

    def update(self, g, a):
        """Update attitude and heading based on measurements.

        Use the Madgwick esimation to achieve this.

        Parameters
        ----------
        g: numpy.ndarray
            Angular velocity read by a gyroscope. Size 3x1.
        a: numpy.ndarray
            Linear acceleration read by an accelerometer. Size 3x1.
        """
        q_dot = numpy.array([[
                0., -g[0, 0], -g[1, 0], -g[2, 0]
            ], [
                g[0, 0], 0., g[2, 0], -g[1, 0]
            ], [
                g[1, 0], -g[2, 0], 0., g[0, 0]
            ], [
                g[2, 0], g[1, 0], -g[0, 0], 0.
            ]]).dot(self.q * 0.5)

        if numpy.count_nonzero(a) > 0:
            acc = a / numpy.linalg.norm(a, 2)

            f = numpy.array([[
                    self.q[1, 0] * self.q[3, 0] - self.q[0, 0] * self.q[2, 0]
                ], [
                    self.q[0, 0] * self.q[1, 0] + self.q[2, 0] * self.q[3, 0]
                ], [
                    0.5 - self.q[1, 0]**2 - self.q[2, 0]**2
                ]]) * 2. - acc
            j = numpy.array([[
                    -self.q[2, 0], self.q[3, 0], -self.q[0, 0], self.q[1, 0]
                ], [
                    self.q[1, 0], self.q[0, 0], self.q[3, 0], self.q[2, 0]
                ], [
                    0., -2. * self.q[1, 0], -2. * self.q[2, 0], 0.
                ]]) * 2.

            step = j.T.dot(f)

            q_dot -= step * (self.beta / numpy.linalg.norm(step, 2))

        self.q += q_dot / self.frequency
        self.q /= numpy.linalg.norm(self.q, 2)


class Mahony(Ahrs):
    """Mahony AHRS.

    Parameters
    ----------
    frequency: float
        Number of performed updates per second.

    Attributes
    ----------
    frequency
    int_fb : numpy.ndarray
        A 3x1 array storage for the integral component.
    ki2: float
        Integral regulator component, doubled.
    kp2: float
        Proportional regulator component, doubled.
    """
    def __init__(self, frequency):
        super(Madgwick, self).__init__(frequency)
        self.int_fb = numpy.zeros([3, 1])
        self.ki2 = 2. * 0.
        self.kp2 = 2. * 0.5

    def update(self, g, a):
        """Update attitude and heading based on measurements.

        Use the Mahony esimation to achieve this.

        Parameters
        ----------
        g: numpy.ndarray
            Angular velocity read by a gyroscope. Size 3x1.
        a: numpy.ndarray
            Linear acceleration read by an accelerometer. Size 3x1.
        """
        gyr = g.copy()
        if numpy.count_nonzero(a) > 0:
            acc = a / numpy.linalg.norm(a, 2)

            halfvx = self.q[1, 0] * self.q[3, 0] - self.q[0, 0] * self.q[2, 0]
            halfvy = self.q[0, 0] * self.q[1, 0] + self.q[2, 0] * self.q[3, 0]
            halfvz = (self.q[0, 0] * self.q[0, 0] + self.q[3, 0] * self.q[3, 0]
                      - 0.5)

            halfe = numpy.array([[
                    acc[1, 0] * halfvz - acc[2, 0] * halfvy
                ], [
                    acc[2, 0] * halfvx - acc[0, 0] * halfvz
                ], [
                    acc[0, 0] * halfvy - acc[1, 0] * halfvx
                ]])

            if self.ki2 > 0.:
                self.int_fb += (self.ki2 / self.frequency) * halfe
                gyr += self.int_fb
            else:
                self.int_fb = numpy.zeros([3, 1])

            gyr += self.kp2 * halfe
        gyr /= 2. * self.frequency

        qa = q0;
    	qb = q1;
    	qc = q2;
        gx, gy, gz = gyr.reshape(3).tolist()
    	self.q[0, 0] += -qb * gx - qc * gy - q3 * gz
    	self.q[1, 0] += qa * gx + qc * gz - q3 * gy
    	self.q[2, 0] += qa * gy - qb * gz + q3 * gx
    	self.q[3, 0] += qa * gz + qb * gy - qc * gx

        self.q /= numpy.linalg.norm(self.q, 2)
