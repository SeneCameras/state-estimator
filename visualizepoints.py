#!/usr/bin/env python

from localization.sensors.vision import FeatureGenerator, projectToPlane, matchFlann
import numpy as np
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D


def main():
    fg = FeatureGenerator()
    fig = pp.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = np.zeros([3, 1])
    r = np.identity(3)
    ps = []
    for st, color, marker in [(2., 'red', 'o'), (-2., 'blue', '^')]:
        p[1] = st
        mypoints = fg.getRandomVisiblePoints(p, r, np.pi / 4., 0.8)
        ps.append(mypoints)
        x = np.array([q[0] for q in mypoints])
        y = np.array([q[1] + st for q in mypoints])
        z = np.array([q[2] for q in mypoints])
        ax.scatter(x, y, z, color=color, marker=marker)
        mypoints = projectToPlane(mypoints)
        x = np.array([0. for q in mypoints])
        y = np.array([q[0] + st for q in mypoints])
        z = np.array([q[1] for q in mypoints])
        ax.scatter(x, y, z, color=color, marker=marker)
    p[1] = 4.
    matches = matchFlann(ps[0], ps[1], p, np.identity(3))
    offset = np.array([[0., 0.], [2., -2.], [0., 0.]])
    print len(matches), 'matches found!'
    for i1, i2 in matches:
        mypoints = np.array(
                [ps[0][i1].flatten(), ps[1][i2].flatten()]).T + offset
        ax.plot(mypoints[0, :], mypoints[1, :], mypoints[2, :], color='green')
        mypoints = mypoints.mean(1).reshape([3, 1])
        ax.scatter(mypoints[0, :], mypoints[1, :], mypoints[2, :],
                color='green', marker='x')
    pp.show()


if __name__ == '__main__':
    main()
