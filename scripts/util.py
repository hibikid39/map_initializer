import numpy as np

def vec2ss_matrix(vector):  # vector to skewsym. matrix
    ss_matrix = np.zeros((3,3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix

def RT2Twist(R, t):
    twist = np.zeros(6)

    E = np.identity(3)
    if np.linalg.norm(E - R) < 1e-12:
        twist[0:3] = 0
        twist[3:6] = t
    else:
        th = np.arccos(0.5 * (np.trace(R) - 1))  # theta
        w_skewsym = (R - R.transpose()) / (2.0 * np.sin(th))
        twist[0:3] = np.array([w_skewsym[2, 1],
                            w_skewsym[0, 2],
                            w_skewsym[1, 0]]) * th

        G_inv = (
            np.identity(3) / th
            - w_skewsym / 2.0
            + (1.0 / th - 1.0 / (2.0 * np.tan(th / 2.0)))
            * np.matmul(w_skewsym, w_skewsym)
        )
        twist[3:6] = G_inv @ t * th

    return twist

def Twist2RT(twist):
    th = np.linalg.norm(twist[0:3])

    if th < 1e-12:
        R = np.identity(3)
        t = twist[3:6]

        return R, t
    else:
        w = (twist[0:3] / th)
        v = (twist[3:6] / th)
        w_skewsym = vec2ss_matrix(w)

        R = np.identity(3) + np.sin(th) * w_skewsym + (1 - np.cos(th)) * (w_skewsym @ w_skewsym)

        G = (
            np.identity(3) * th
            + (1 - np.cos(th)) * w_skewsym
            + (th - np.sin(th)) * (w_skewsym @ w_skewsym)
        )
        t = G @ v

        return R, t