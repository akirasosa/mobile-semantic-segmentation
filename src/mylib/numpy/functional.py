import numpy as np


def rand_rotation_matrix(deflection=1.0, rand=None):
    if rand is None:
        rand = np.random.uniform(size=(3,))

    theta, phi, z = rand

    theta = theta * 2.0 * deflection * np.pi
    phi = phi * 2.0 * np.pi
    z = z * 2.0 * deflection

    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    M = (np.outer(V, V) - np.eye(3)).dot(R)

    return M
