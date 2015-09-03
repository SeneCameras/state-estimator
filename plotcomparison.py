#!/usr/bin/env python

import matplotlib.pyplot as pp
import numpy
import pickle

if __name__ == '__main__':
    with open('comparedata.data', 'r') as f:
        responses = pickle.load(f)

    cases = []

    for t in ('stationary', 'oscillating'):
        for filter_name in ('EKF', 'UKF'):
            for vision_freq in (1., 5., 10., 30., 100.):
                for vision_variance in (0.1, 0.01, 0.001, 0.0001):
                    cases.append([filter_name, vision_freq, vision_variance, t])

    print 'Red   - Stationary EKF'
    print 'Blue  - Stationary UKF'
    print 'Green - Oscillating EKF'
    print 'Cyan  - Oscillating UKF'

    step = len(responses) / 4

    # Compare individual cases between oscillating and stationary, EKF and UKF
    for case in xrange(step):
        pp.figure(case)
        pp.suptitle('Vision freq: {0} Hz; Variance multiplier: {1}'.format(
                cases[case][1], cases[case][2]))
        subcases = [responses[case + offs] for offs in numpy.arange(4) * step]
        for subcase, color in zip(subcases, ['red', 'blue', 'green', 'cyan']):
            # Average the datapoints
            p_error = 0.
            for datapoint in subcase:
                p_error += numpy.abs(datapoint['position']['real'] -
                                     datapoint['position']['estimate'])
            p_error /= len(subcase)

            pp.subplot(2, 2, 1)
            pp.plot(p_error[0, :], color=color)
            pp.subplot(2, 2, 2)
            pp.plot(p_error[1, :], color=color)
            pp.subplot(2, 2, 3)
            pp.plot(p_error[2, :], color=color)
            p_error_abs = numpy.sqrt(numpy.sum(p_error * p_error, axis=0))
            pp.subplot(2, 2, 4)
            pp.plot(p_error_abs, color=color)
        pp.subplot(2, 2, 1)
        pp.title('Error along X axis')
        pp.subplot(2, 2, 2)
        pp.title('Error along Y axis')
        pp.subplot(2, 2, 3)
        pp.title('Error along Z axis')
        pp.subplot(2, 2, 4)
        pp.title('Euclidean distance error')
        pp.show()
