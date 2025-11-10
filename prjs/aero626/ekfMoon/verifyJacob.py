import numpy as np
import sys, os

# Add Basilisk root if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Example: if you have a Quaternion helper (optional)
# from helpers.attitude.Quaternion import Quaternion


def measurement_model(x_r_alpha):
    """
    Measurement model:
    Given the body position r_BM_M and small attitude error alpha,
    computes a landmark position in the body frame.

    Args:
        x_r_alpha: (6,) np.array
                   [r_BM_M (3x1 position), alpha (3x1 small attitude angle)]

    Returns:
        zk: (3,) np.array - predicted measurement (landmark in body frame)
    """

    # State: [position (3), attitude error (3)]
    r_BM_M = x_r_alpha[:3]
    alpha = x_r_alpha[3:]

    # Known (fixed) landmark position in map frame
    r_LM_M = [np.float64(1737.2899172540408), np.float64(17.511677096554322), np.float64(8.708878958578461)]


    TBMhat = np.array([[ 0.99585566,  0.08714867, -0.02601181],
       [-0.08664875,  0.99604282,  0.01976614],
       [ 0.02763147, -0.01743034,  0.9994662 ]])
    
    

    # Rotation matrix approximation (small-angle assumption)
    I3 = np.eye(3)
    alpha_skew = np.array([
        [0, -alpha[2], alpha[1]],
        [alpha[2], 0, -alpha[0]],
        [-alpha[1], alpha[0], 0]
    ])

    # Landmark in body frame (approximation)
    zk = (I3 - alpha_skew)@ TBMhat @ (r_LM_M - r_BM_M)

    return zk


def numeric_jacobian(f, x, eps=1e-6):
    """
    Compute numerical Jacobian of vector function f(x) via central difference.

    Args:
        f: function that takes x (1D array) and returns y (1D array)
        x: (n,) np.array
        eps: float - step size for finite differencing

    Returns:
        J: (m, n) np.array - numerical Jacobian
    """
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(f(x))
    m, n = y0.size, x.size
    J = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = eps[i]
        y_plus = np.asarray(f(x + dx))
        y_minus = np.asarray(f(x - dx))
        J[:, i] = (y_plus - y_minus) / (2 * eps[i])
    return J


def analytic_jacobian(x):
    """
    Analytic Jacobian of measurement model wrt state x = [r_BM_M, alpha].

    Returns:
        Hx: (3, 6) Jacobian
    """

    r_BM_M = x[:3]
    alpha = x[3:]
    
    r_LM_M = [np.float64(1737.2899172540408), np.float64(17.511677096554322), np.float64(8.708878958578461)]


    TBMhat = np.array([[ 0.99585566,  0.08714867, -0.02601181],
       [-0.08664875,  0.99604282,  0.01976614],
       [ 0.02763147, -0.01743034,  0.9994662 ]])

    I3 = np.eye(3)

    # Partial wrt position: -I3
    Hx_pos = -TBMhat

    # Partial wrt small attitude error alpha
    dr = TBMhat@(r_LM_M - r_BM_M)
    rx, ry, rz = dr
    dr_skew = np.array([
        [0, -rz, ry],
        [rz, 0, -rx],
        [-ry, rx, 0]
    ])
    Hx_alpha = dr_skew

    Hx = np.hstack((Hx_pos, Hx_alpha))
    return Hx


if __name__ == "__main__":

    # Example state: position and small rotation error
    x_r_alpha = np.array([1.73875355e+03,  1.67427882e+00, -4.67455976e+01,
                           np.deg2rad(0.01), np.deg2rad(0.01), np.deg2rad(0.01)])
    
    epsVec = np.array([
        1e-6,1e-6,1e-6,np.deg2rad(0.001),np.deg2rad(0.001),np.deg2rad(0.001)
    ])

    # Compute analytic and numeric Jacobians
    Hx_analytic = analytic_jacobian(x_r_alpha)
    Hx_numeric = numeric_jacobian(measurement_model, x_r_alpha, eps=epsVec)

    print("Analytic Jacobian:\n", Hx_analytic)
    print("\nNumeric Jacobian:\n", Hx_numeric)
    print("\nDifference:\n", Hx_numeric - Hx_analytic)





