from khrylib.utils import *


def get_traj_from_state_pred(state_pred, dt=1.0/30.0, init_pos=None, init_heading=None):
    init_pos = np.array([0., 0.]) if init_pos is None else init_pos
    init_heading = np.array([1., 0., 0., 0.]) if init_heading is None else init_heading
    nq = state_pred.shape[-1] - 4
    pos = init_pos.copy()
    heading = init_heading.copy()
    traj_pred = []
    for i in range(state_pred.shape[0]):
        state_pred[i, 1:5] /= np.linalg.norm(state_pred[i, 1:5])
        qpos = np.concatenate((pos, state_pred[i, :nq - 2]))
        qvel = state_pred[i, nq - 2:]
        qpos[3:7] = quaternion_multiply(heading, qpos[3:7])
        linv = quat_mul_vec(heading, qvel[:3])
        angv = quat_mul_vec(qpos[3:7], qvel[3:6])
        pos += linv[:2] * dt
        new_q = quaternion_multiply(quat_from_expmap(angv * dt), qpos[3:7])
        heading = get_heading_q(new_q)
        traj_pred.append(qpos)
    traj_qpos = np.vstack(traj_pred)
    return traj_qpos


def batch_get_traj(traj_arr, dt=1.0/30.0, init_pos=None, init_heading=None):
    traj_int = []
    for traj in traj_arr:
        traj_int.append(get_traj_from_state_pred(traj, dt, init_pos, init_heading))
    traj_int = np.stack(traj_int)
    return traj_int
