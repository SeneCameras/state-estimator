import math
import numpy

class StateMember:
    x = 0
    y = 1
    z = 2
    roll = 3
    pitch = 4
    yaw = 5
    v_x = 6
    v_y = 7
    v_z = 8
    v_roll = 9
    v_pitch = 10
    v_yaw = 11
    a_x = 12
    a_y = 13
    a_z = 14


def clampRotation(angle):
    while (angle > math.pi):
        angle -= 2.0 * math.pi
    while (angle < -math.pi):
        angle += 2.0 * math.pi
    return angle


def rpyToRotationMatrix(roll, pitch, yaw):
    trig_funcs = (math.cos(roll), math.cos(pitch), math.cos(yaw),
                  math.sin(roll), math.sin(pitch), math.sin(yaw))
    return __sinCosToRotationMatrix(*trig_funcs)


def rpyToRotationMatrixAndDerivatives(roll, pitch, yaw):
    trig_funcs = (math.cos(roll), math.cos(pitch), math.cos(yaw),
                  math.sin(roll), math.sin(pitch), math.sin(yaw))
    return (__sinCosToRotationMatrix(*trig_funcs),
            __sinCosToRdr(*trig_funcs),
            __sinCosToRdp(*trig_funcs),
            __sinCosToRdy(*trig_funcs))


def __sinCosToRotationMatrix(cr, cp, cy, sr, sp, sy):
    return numpy.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]])


def __sinCosToRdr(cr, cp, cy, sr, sp, sy):
    return numpy.array([[
            0.,
            cy * sp * cr + sy * sr,
            -cy * sp * sr + sy * cr,
        ], [
            0.,
            sy * sp * cr - cy * sr,
            -sy * sp * sr - cy * cr,
        ], [
            0.,
            cp * cr,
            -cp * sr,
        ]])


def __sinCosToRdp(cr, cp, cy, sr, sp, sy):
    return numpy.array([[
            -cy * sp,
            cy * cp * sr,
            cy * cp * cr,
        ], [
            -sy * sp,
            sy * cp * sr,
            sy * cp * cr,
        ], [
            -cp,
            -sp * sr,
            -sp * cr,
        ]])


def __sinCosToRdy(cr, cp, cy, sr, sp, sy):
    return numpy.array([[
            -sy * cp,
            -sy * sp * sr - cy * cr,
            -sy * sp * cr + cy * sr,
        ], [
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
        ], [
            0.,
            0.,
            0.,
        ]])


class Measurement(object):
    """Structure for storing measurements"""
    def __init__(self, time, measurement, covariance, update_vector):
        super(Measurement, self).__init__()
        self.measurement = measurement
        self.covariance = covariance
        self.update_vector = update_vector
        self.time = time
        self.mahalanobis_threshold = float('inf')

    """Override of comparison operator for sorting purposes"""
    def __cmp__(self, other):
        return cmp(self.time, other.time)
