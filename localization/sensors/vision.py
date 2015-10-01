import cv2
import numpy
import random

import localization.sensors.base
from localization.util import Measurement, StateMember, rpyToRotationMatrix


def projectToPlane(points):
    """Projects a list of 3x1 coordinates as in a pinhole camera.

    The first coordinate is scaled down to 1, and the rest are scaled down
    proportionally.

    Parameters
    ----------
    points: list(numpy.ndarray)
        A list of 3x1 arrays representing coordinates in the camera frame.

    Returns
    -------
    list(numpy.ndarray)
        A list of 2x1 arrays representing coordinates in the image plane.
    """
    return [p[1:3] / p[0] for p in points]


class FeatureGenerator(object):
    """Generates simulated points for features in the world.

    Stores random feature points in the world, that get randomly occluded and
    replaced, to give an approximation of how feature points are extracted.

    The generator is focused for problems no intended movement, focusing a lot
    of points around the world center. The distribution of the points is
    Gaussian.

    Parameters
    ----------
    distance_standard_deviation: float, optional
        Standard deviation of feature point poisitons around the world center.
        Default is 50.0.
    point_count: int, optional
        Number of points generated in the world.
        Default is 1000.

    Attributes
    ----------
    points: list(numpy.ndarray)
        A list of length 3 arrays containing coordinates of all features.
    """
    def __init__(self, distance_standard_deviation = None, point_count = None):
        super(FeatureGenerator, self).__init__()

        if distance_standard_deviation is None:
            distance_standard_deviation = 50.
        if point_count is None:
            point_count = 1000

        covariance = numpy.identity(3) * distance_standard_deviation
        mean = numpy.zeros(3)

        self.points = [numpy.random.multivariate_normal(mean, covariance)
                for _ in xrange(point_count)]

    def getVisiblePoints(self, position, rotation, fov):
        """Get a list of point positions, relative to given view.

        Get all feature points that are visible by looking in the given
        direction and standing in the given position. The view is conical, with
        the fov given.

        Parameters
        ----------
        position: numpy.ndarray
            A 3x1 array representing the camera position.
        rotation: numpy.ndarray
            A 3x3 array representing the camera orientation's rotation matrix.
        fov: float
            Half of the opening angle of the cone that we are viewing through.
            The angle is in radians, and represents the maximum angle offset
            of any point that we can see.

        Returns
        -------
        list(numpy.ndarray)
            A list of 3x1 arrays representing coordinates in the camera frame.
        """
        direction = rotation.dot(numpy.array([[1.],[0.],[0.]]))
        cos_fov = numpy.cos(fov)
        pos = position.flatten()

        retval = []

        for p in self.points:
            delta_p = p - pos
            delta_p_len = numpy.linalg.norm(delta_p)
            if delta_p.dot(direction) > delta_p_len * cos_fov:
                retval.append(rotation.T.dot(delta_p.reshape([3, 1])))

        return retval

    def getRandomVisiblePoints(self, position, rotation, fov, percentage):
        """Get a sample of the list of point positions, relative to given view.

        Get some of the feature points that are visible by looking in the given
        direction and standing in the given position. The view is conical, with
        the fov given.

        Parameters
        ----------
        position: numpy.ndarray
            A 3x1 array representing the camera position.
        rotation: numpy.ndarray
            A 3x3 array representing the camera orientation's rotation matrix.
        fov: float
            Half of the opening angle of the cone that we are viewing through.
            The angle is in radians, and represents the maximum angle offset
            of any point that we can see.
        percentage: float
            The percentage of points, with values inside [0, 1], that are
            chosen for viewing, from inside the given cone.

        Returns
        -------
        list(numpy.ndarray)
            A list of 3x1 arrays representing coordinates in the camera frame.
        """
        p = self.getVisiblePoints(position, rotation, fov)
        return random.sample(p, int(len(p) * percentage))


def matchFlann(points_a, points_b, translation, rotation):
    """Match two sets of points based on a transformation.

    Given a set of old points A, and a set of new points B, transform the
    old points with the given translation and rotation, and retrieve which
    points match with eachother.
    Matches are accepted if the closest point is 30 percent closer than the
    second closest point.

    Parameters
    ----------
    points_a: list(numpy.ndarray)
        A list of 3x1 arrays representing old feature coordinates in space.
    points_b: list(numpy.ndarray)
        A list of 3x1 arrays representing new feature coordinates in space.
    translation: numpy.ndarray
        A 3x1 array representing the translation between states.
    rotation: numpy.ndarray
        A 3x3 array representing the rotation between states.

    Returns
    -------
    list(tuple(int, int))
        List of point indices for all matches.
    """
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
    search_params = dict(checks=50)
    points_a = numpy.array([(p + translation).flatten() for p in points_a])
    points_b = numpy.array([p.flatten() for p in points_b], numpy.float32)
    points_a = rotation.dot(points_a.T).T.astype(numpy.float32)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(points_a, points_b, k=2)

    good_matches = {}
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            accepts = True
            if m.queryIdx in good_matches:
                accepts = False
                if m.distance < good_matches[m.queryIdx].distance:
                    accepts = True
            if accepts:
                good_matches[m.queryIdx] = m

    return [(m.queryIdx, m.trainIdx) for m in good_matches.values()]


class Vision(localization.sensors.base.SensorBase):
    """Implementation of the vision system.

    Determines velocity based on changes in pairs of low resolution images.
    The velocity coordinates are, in order: forward, left, up.

    The reading frequency can be chosen.

    Parameters
    ----------
    start_time: float
        Start time of the sensor usage, in seconds. Preferably set to 0.
    frequency: float
        Frequency of sensor responses, in Hz.
    """
    def __init__(self, start_time, frequency):
        super(Vision, self).__init__(
                start_time, 1. / frequency, numpy.asarray(covariance),
                [StateMember.v_x, StateMember.v_y, StateMember.v_z])

    def generateMeasurement(self, real_state):
        """Generate a vision measurement based on the given state.

        Parameters
        ----------
        real_state: numpy.ndarray
            A 15x1 array representing the actual state.

        Returns
        -------
        localization.util.Measurement
            Generate a measurement based on images and changes.
        """
        raise NotImplementedError("Please Implement this method")
