import numpy as np

e_u_true_possibility = 0.99
log_e_u_true_possibility = np.log(e_u_true_possibility)
e_u_false_possibility = 1 - e_u_true_possibility
log_e_u_false_possibility = np.log(e_u_false_possibility)

theta_u = 1
theta_t = 1
theta_s = 1
flow_threshold = 100
flow_range = 3

flow_out_of_uncalculated_err = -20000
flow_out_of_range_err = -10000


def filter_unreliable_flow(flow):
    flow_magnitude = np.linalg.norm(flow, axis=2)
    reliable_flow = np.zeros_like(flow)
    reliable_flow[flow_magnitude < flow_threshold] = flow[flow_magnitude < flow_threshold]
    return reliable_flow


def get_positions(flo, precomputed_positions, pos, t):

    if (precomputed_positions[t, pos[0], pos[1]] != flow_out_of_uncalculated_err).all():
        return precomputed_positions[t, pos[0], pos[1], 0], precomputed_positions[t, pos[0], pos[1], 1]

    # [-flow_range, -flow_range + 1, ..., -1, 1, ..., flow_range - 1, flow_range]
    precomputed_positions[t, pos[0], pos[1], 0] = np.full((flow_range, 2), flow_out_of_range_err)
    precomputed_positions[t, pos[0], pos[1], 1] = np.full((flow_range, 2), flow_out_of_range_err)
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


def energy(mask: np.ndarray, osvos_mask: np.ndarray, idx):

    global precomputed_positions, diff_between_masks

    if precomputed_positions is None or diff_between_masks is None:
        raise ValueError("Global variables precomputed_positions and diff_between_masks must be initialized before calling this function.")

    t = idx[0]
    x, y = idx[1], idx[2]
    e_u, e_t, e_s = 0, 0, 0

    e_u += -log_e_u_false_possibility if mask[idx] != osvos_mask[idx] else -log_e_u_true_possibility

    # -1 -> -flow_range
    for dt in range(min(flow_range, t - 1)):
        e_t += time_parameter(t - dt - 1) * time_parameter(t) * diff_between_masks[t - dt - 1, t] * diff_between_masks[t - dt - 1, t]

    # 1 -> flow_range
    for dt in range(min(flow_range, mask.shape[0] - t - 1)):
        e_t += time_parameter(t + dt + 1) * time_parameter(t) * diff_between_masks[t, t + dt + 1] * diff_between_masks[t, t + dt + 1]

    return theta_u * e_u + theta_t * e_t + theta_s * e_s


def diff_update(t, pos, dv):
    x, y = pos[0], pos[1]

    global mask

    for dt in range(min(flow_range, t - 1)):
        if precomputed_positions[t, x, y, 0, dt, 0] == flow_out_of_range_err:
            break
        if mask[t - dt - 1, precomputed_positions[t, x, y, 0, dt, 0], precomputed_positions[t, x, y, 0, dt, 1]] != mask[t, x, y]:
            diff_between_masks[t - dt - 1, t] += dv

    for dt in range(min(flow_range, mask.shape[0] - t - 1)):
        if precomputed_positions[t, x, y, 1, dt, 0] == flow_out_of_range_err:
            break
        if mask[t + dt + 1, precomputed_positions[t, x, y, 1, dt, 0], precomputed_positions[t, x, y, 1, dt, 1]] != mask[t, x, y]:
            diff_between_masks[t, t + dt + 1] += dv


def init(flo):

    global precomputed_positions, diff_between_masks

    print('Start init...')

    precomputed_positions = np.empty((mask.shape[0], mask.shape[1], mask.shape[2], 2, flow_range, 2), dtype=tuple)
    precomputed_positions.fill(flow_out_of_uncalculated_err)

    for t in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            for y in range(mask.shape[2]):
                neg_ret, pos_ret = get_positions(flo, precomputed_positions, (x, y), t)
                precomputed_positions[t, x, y, 0] = neg_ret
                precomputed_positions[t, x, y, 1] = pos_ret

    diff_between_masks = np.zeros((mask.shape[0], mask.shape[0]))

    for t in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            for y in range(mask.shape[2]):
                diff_update(t, (x, y), 1)