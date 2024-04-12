import numpy as np
from config import *
from utilities import time_parameter, s_parameter

precomputed_positions = None


def get_positions(flo, precomputed_positions, pos, t):

    if (precomputed_positions[t, pos[0], pos[1]] != not_calculated_err).all():
        return precomputed_positions[t, pos[0], pos[1], 0], precomputed_positions[t, pos[0], pos[1], 1]

    # [-flow_range, -flow_range + 1, ..., -1, 1, ..., flow_range - 1, flow_range]
    precomputed_positions[t, pos[0], pos[1], 0] = np.full((flow_range, 2), out_of_range_err)
    precomputed_positions[t, pos[0], pos[1], 1] = np.full((flow_range, 2), out_of_range_err)
    cur_pos = list(pos)
    # Loop from 0 to flow_range
    for offset in range(min(flow_range, precomputed_positions.shape[0] - t - 1)):
        flow = flo[t + offset]
        dx, dy = flow[cur_pos[0], cur_pos[1]]
        cur_pos = [round(cur_pos[0] + dx), round(cur_pos[1] + dy)]
        if cur_pos[0] < 0 or cur_pos[0] >= flo.shape[1] or cur_pos[1] < 0 or cur_pos[1] >= flo.shape[2]:
            break
        precomputed_positions[t, pos[0], pos[1], 1, offset] = [cur_pos[0], cur_pos[1]]
        precomputed_positions[t + offset + 1, cur_pos[0], cur_pos[1], 0, offset] = [pos[0], pos[1]]

    cur_pos = list(pos)
    # Loop from 0 to -flow_range
    for offset in range(min(flow_range, t - 1)):
        flow = flo[t - offset]
        dx, dy = flow[cur_pos[0], cur_pos[1]]
        cur_pos = [round(cur_pos[0] - dx), round(cur_pos[1] - dy)]
        if cur_pos[0] < 0 or cur_pos[0] >= flo.shape[1] or cur_pos[1] < 0 or cur_pos[1] >= flo.shape[2]:
            break
        precomputed_positions[t, pos[0], pos[1], 0, offset] = [cur_pos[0], cur_pos[1]]
        precomputed_positions[t - offset - 1, cur_pos[0], cur_pos[1], 1, offset] = [pos[0], pos[1]]

    return precomputed_positions[t, pos[0], pos[1], 0], precomputed_positions[t, pos[0], pos[1], 1]


def energy(mask: np.ndarray, osvos_mask: np.ndarray, t, x, y, value):

    if precomputed_positions is None:
        raise ValueError("precomputed_positions must be initialized before calling this function.")

    idx = (t, x, y)
    e_u, e_t, e_s = 0, 0, 0

    e_u += -log_e_u_false_possibility if value != osvos_mask[idx] else -log_e_u_true_possibility

    # -1 -> -flow_range
    for dt in range(min(flow_range, t - 1)):
        if precomputed_positions[t, x, y, 0, dt, 0] == out_of_range_err:
            break
        # print(f'{t}, {dt}, {x}, {y}, {precomputed_positions[t, x, y, 0, dt, 0]}, {precomputed_positions[t, x, y, 0, dt, 1]}')
        e_t += time_parameter(t - dt - 1) * time_parameter(t) * (1 if value != mask[t - dt - 1, precomputed_positions[t, x, y, 0, dt, 0], precomputed_positions[t, x, y, 0, dt, 1]] else 0) ** 2

    # 1 -> flow_range
    for dt in range(min(flow_range, mask.shape[0] - t - 1)):
        if precomputed_positions[t, x, y, 1, dt, 0] == out_of_range_err:
            break
        e_t += time_parameter(t + dt + 1) * time_parameter(t) * (1 if value != mask[t + dt + 1, precomputed_positions[t, x, y, 1, dt, 0], precomputed_positions[t, x, y, 1, dt, 1]] else 0) ** 2

    return theta_u * e_u + theta_t * e_t + theta_s * s_parameter(t) * e_s


def init(flo, mask):

    global precomputed_positions
    shape = mask.shape

    precomputed_positions = np.empty((shape[0], shape[1], shape[2], 2, flow_range, 2), dtype=tuple)
    precomputed_positions.fill(not_calculated_err)

    for t in range(shape[0]):
        for x in range(shape[1]):
            for y in range(shape[2]):
                neg_ret, pos_ret = get_positions(flo, precomputed_positions, (x, y), t)
                precomputed_positions[t, x, y, 0] = neg_ret
                precomputed_positions[t, x, y, 1] = pos_ret
