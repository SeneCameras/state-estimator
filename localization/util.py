import math

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
