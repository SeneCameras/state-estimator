#!/usr/bin/env python

import numpy
import matplotlib.pyplot as pp

import localization.simulator

def runCase(filter_name, vision_freq, vision_covar_diag, color):
    sim = localization.simulator.getSimulationUsecase(
            filter_name, vision_freq, vision_covar_diag)
    p_real = numpy.zeros([3, 0])
    p_est = numpy.zeros([3, 0])
    v_real = numpy.zeros([3, 0])
    v_est = numpy.zeros([3, 0])

    ticks = 10000

    t = numpy.arange(float(ticks)) * sim.delta_time

    for i in xrange(ticks):
        sim.tick([i * 0.01, 0.002, 0.], [0., 0., 1.0])
        frame_real = sim.getFrameBaseTruth()
        frame_est = sim.getFrameEstimate()

        p_real = numpy.append(p_real, frame_real['position'], axis=1)
        p_est = numpy.append(p_est, frame_est['position'], axis=1)
        v_real = numpy.append(v_real, frame_real['velocity'], axis=1)
        v_est = numpy.append(v_est, frame_est['velocity'], axis=1)
    p_err = p_real - p_est
    p_err = numpy.sqrt((p_err * p_err).T.dot(numpy.ones([3, 1])))
    v_err = v_real - v_est
    v_err = numpy.sqrt((v_err * v_err).T.dot(numpy.ones([3, 1])))
    pp.subplot(2, 1, 1)
    pp.plot(t, p_err, color=color)
    pp.subplot(2, 1, 2)
    pp.plot(t, v_err, color=color)

if __name__ == '__main__':
    runCase('UKF', 30., [0.05, 0.02, 0.02], 'blue')
    runCase('EKF', 30., [0.05, 0.02, 0.02], 'green')
    runCase('UKF', 10., [0.005, 0.002, 0.002], 'red')
    runCase('EKF', 10., [0.005, 0.002, 0.002], 'cyan')
    pp.show()
