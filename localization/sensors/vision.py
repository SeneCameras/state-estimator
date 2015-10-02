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
        Default is 10.0.
    point_count: int, optional
        Number of points generated in the world.
        Default is 1000.
    minimum_distance: float, optional
        Minimum distance of feature point from the world center.
        Default is 1.5.

    Attributes
    ----------
    points: list(numpy.ndarray)
        A list of length 3 arrays containing coordinates of all features.
    """
    def __init__(self, distance_standard_deviation = None, point_count = None,
                 minimum_distance = None):
        super(FeatureGenerator, self).__init__()

        if distance_standard_deviation is None:
            distance_standard_deviation = 10.
        if point_count is None:
            point_count = 1000
        if minimum_distance is None:
            minimum_distance = 1.5
        distance_standard_deviation -= minimum_distance

        covariance = numpy.identity(3) * (distance_standard_deviation ** 2)
        mean = numpy.zeros(3)

        self.points = [self.__generatePoint(mean, covariance, minimum_distance)
                for _ in xrange(point_count)]


    def __generatePoint(self, mean, covariance, shift_distance):
        """Generate a point and shift it by the given distance."""
        p = numpy.random.multivariate_normal(mean, covariance)
        p_len = numpy.linalg.norm(p)
        if not p_len > 0.:
            p[0] = 1.
            p_len = 1.
        p += p * (shift_distance / p_len)
        return p


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
    cone_angle_reach: float
        Describes the field of view of the used cameras.
        Represents the maximum angle of every visible point, relative to the
        optical center of the camera.
    delta_x: float
        Distance between the left and right camera centers.
        Each camera is offset by delta_x / 2.
    """
    def __init__(self, start_time, frequency, cone_angle_reach, delta_x):
        super(Vision, self).__init__(
                start_time, 1. / frequency, numpy.asarray(covariance),
                [StateMember.v_x, StateMember.v_y, StateMember.v_z])
        self.feature_generator = FeatureGenerator()
        self.fov = cone_angle_reach
        self.delta_x = delta_x
        self.v_lin = numpy.zeros([3, 1])
        self.v_ang = numpy.zeros([3, 1])
        self.old_points = []

    def retrieveVelocities(self, linear, angular):
        """Retrieves linear and angular velocities from measuremenets.

        Needs to be called before every generateMeasurement call.

        Parameters
        ----------
        linear: numpy.ndarray
            A 3x1 array representing the linear velocity.
        angular: numpy.ndarray
            A 3x1 array representing the angular velocity.
        """
        self.v_lin = linear.copy()
        self.v_ang = angular.copy()

    def _generateFeatureProjections(self, real_state, percentage):
        """Generates projections of feature points on the camera planes.

        Parameters
        ----------
        real_state: numpy.ndarray
            A 15x1 array representing the actual state.
        percentage: float
            The percentage of points, with values inside [0, 1], that are
            chosen for viewing, from inside the given cone.

        Returns
        -------
        (list(numpy.ndarray), list(numpy.ndarray))
            Tuple of two lists of 2x1 arrays representing coordinates on the
            camera plane.
        """
        position = real_state[StateMember.x:StateMember.z+1]
        rotation = localization.util.rpyToRotationMatrix(
                *real_state[StateMember.roll:StateMember.yaw+1, 0])
        camera_offset = rotation.dot(
                numpy.array([[0.],[self.delta_x / 2.],[0.]]))
        view_left = projectToPlane(
                self.feature_generator.getRandomVisiblePoints(
                        position - camera_offset, rotation, self.fov,
                        percentage))
        view_right = projectToPlane(
                self.feature_generator.getRandomVisiblePoints(
                        position + camera_offset, rotation, self.fov,
                        percentage))
        return (view_left, view_right)

    def _generateVisibleSpatialPoints(self, real_state, percentage):
        """Generates visible feature points on the camera planes.

        Parameters
        ----------
        real_state: numpy.ndarray
            A 15x1 array representing the actual state.
        percentage: float
            The percentage of points, with values inside [0, 1], that are
            chosen for viewing, from inside the given cone.

        Returns
        -------
        list(numpy.ndarray)
            List of 3x1 arrays representing coordinates from the camera
            perspective.
        """
        position = real_state[StateMember.x:StateMember.z+1]
        rotation = localization.util.rpyToRotationMatrix(
                *real_state[StateMember.roll:StateMember.yaw+1, 0])
        return self.feature_generator.getRandomVisiblePoints(
                position, rotation, self.fov, percentage)

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
        None
            In case that a measurement fails to get generated.
        """
        view_points = self._generateVisibleSpatialPoints(real_state, 0.9)
        if len(self.view_points) == 0 or len(self.old_points) == 0:
            self.old_points = view_points
            return None

        # Transform all old points by our best transformation guess
        translation = self.v_lin * self.delta_time
        rotation = rpyToRotationMatrix(self.v_ang * self.delta_time)
        matches = matchFlann(
                view_points, self.old_points, translation, rotation)

        self.old_points = view_points
        raise NotImplementedError("Please Implement this method")
