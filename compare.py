#!/usr/bin/env python

import math
import numpy
import pickle

import localization.simulator


def runCase(filter_name, vision_freq, vision_covar_diag, velocities):
    sim = localization.simulator.getSimulationUsecase(
            filter_name, vision_freq, vision_covar_diag)
    p_real = numpy.zeros([3, 0])
    p_est = numpy.zeros([3, 0])
    v_real = numpy.zeros([3, 0])
    v_est = numpy.zeros([3, 0])

    t = numpy.arange(float(len(velocities))) * sim.delta_time

    for point in velocities:
        sim.tick(*point)
        frame_real = sim.getFrameBaseTruth()
        frame_est = sim.getFrameEstimate()

        p_real = numpy.append(p_real, frame_real['position'], axis=1)
        p_est = numpy.append(p_est, frame_est['position'], axis=1)
        v_real = numpy.append(v_real, frame_real['velocity'], axis=1)
        v_est = numpy.append(v_est, frame_est['velocity'], axis=1)

    return {
        'position': {
            'real': p_real,
            'estimate': p_est,
        },
        'velocity': {
            'real': v_real,
            'estimate': v_est,
        },
    }


def firstOrderStepResponse(A, T, dt, steps):
    t = numpy.arange(steps) * dt
    y = 1. - numpy.exp(t / -T) * A
    return y.tolist()


if __name__ == '__main__':
    rotation_velocity = 1.0
    rotation_steps = 2000
    rotation_T = 0.02
    oscillations = 5
    delta_time = 1. / 4000.
    case_attempts = 5

    traj_stationary = [[[0.] * 3] * 2] * (2 * rotation_steps * oscillations)

    response_right = firstOrderStepResponse(
            rotation_velocity, rotation_T, delta_time, rotation_steps)
    response_left = firstOrderStepResponse(
            -rotation_velocity, rotation_T, delta_time, rotation_steps)
    oscillation_speeds = (response_right + response_left) * oscillations
    traj_oscillation = [[[0.] * 3, [0., 0., v]] for v in oscillation_speeds]

    trajectories = {
        'stationary': traj_stationary,
        'oscillating': traj_oscillation,
    }

    cases = []

    for t in ('stationary', 'oscillating'):
        for filter_name in ('EKF', 'UKF'):
            for vision_freq in (1., 5., 10., 30., 100.):
                for vision_variance in (0.1, 0.01, 0.001, 0.0001):
                    cases.append([filter_name, vision_freq, vision_variance, t])

    responses = []

    for c in cases:
        print 'Running case...'
        print 'Filter: ' + c[0]
        print 'Vision frequency: ' + str(c[1])
        print 'Vision variance factor: ' + str(c[2])
        print 'Trajectory: ' + c[3]

        covar = [5. * c[2], 2. * c[2], 2. * c[2]]

        attempts = [runCase(c[0], c[1], covar, trajectories[c[3]])
                for _ in xrange(case_attempts)]

        responses.append(attempts)

    with open('comparedata.data', 'w') as f:
        pickle.dump(responses, f)
