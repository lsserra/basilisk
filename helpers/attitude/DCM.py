import numpy as np


def T1(theta):
    """Passive DCM for rotation about axis 1 (x-axis)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])

def T2(theta):
    """Passive DCM for rotation about axis 2 (y-axis)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])

def T3(theta):
    """Passive DCM for rotation about axis 3 (z-axis)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])