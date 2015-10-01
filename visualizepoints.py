#!/usr/bin/env python

from localization.sensors.vision import FeatureGenerator, projectToPlane
import numpy as np
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D


def main():
    fg = FeatureGenerator()
    fig = pp.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = np.zeros([3, 1])
    r = np.identity(3)
    for st, color, marker in [(2., 'red', 'o'), (-2., 'blue', '^')]:
        p[1] = st
        mypoints = fg.getRandomVisiblePoints(p, r, np.pi / 4., 0.8)
        x = np.array([q[0] for q in mypoints])
        y = np.array([q[1] + st for q in mypoints])
        z = np.array([q[2] for q in mypoints])
        ax.scatter(x, y, z, color=color, marker = marker)
        mypoints = projectToPlane(mypoints)
        x = np.array([0. for q in mypoints])
        y = np.array([q[0] + st for q in mypoints])
        z = np.array([q[1] for q in mypoints])
        ax.scatter(x, y, z, color=color, marker = marker)
    pp.show()


if __name__ == '__main__':
    main()
