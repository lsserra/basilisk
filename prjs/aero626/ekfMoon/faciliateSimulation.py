import os, sys
import numpy as np

import copy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import pickle

# from EkfPoseEstimator import EkfErrorState, EkfReferenceState, EkfPoseEstimator


# Add the basilisk root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# attitude helpers
from helpers.attitude import DCM
from helpers.attitude.Quaternion import Quaternion




### create landmark measurements along trajectory ###

import numpy as np
def generateLandmarks(
    radiusBodyKm,
    mapPos1sigma,
    nPoints, 
    randomSeed=None
):
    if randomSeed is not None:
        np.random.seed(randomSeed)

    landmarks_N_km = []

    rng = np.random.default_rng()


    # random vectors
    v = rng.normal(loc=0, scale=1, size=(nPoints,3))
    v_norm = v / np.linalg.norm(v, axis=1, keepdims=True)

    # true landmarks
    trueLandmarks = v_norm * radiusBodyKm

    # noisy map 
    mapLandmarks = rng.normal(loc=trueLandmarks, scale=mapPos1sigma, size=(nPoints,3))

    return trueLandmarks, mapLandmarks

def getLandmarkMeasurements(
    r_BM_M_truth,
    q_BM_truth,
    distanceThresholdKm,
    trueLandmarks,
    measurement1sigma,
    halfAngleDeg=20.0,      # cone half-angle (degrees)
    randomSeed=None
):
    """
    Generate landmark measurements within a given distance AND within a viewing cone.
    Cone Axis is assumed to be nadir pointing.

    Args:
        r_BM_M_truth (np.ndarray): True body position in M frame (3,)
        q_BM_truth (Quaternion): True quaternion body-to-M frame
        distanceThresholdKm (float): Max distance to consider [km]
        trueLandmarks (np.ndarray): Nx3 array of landmark positions [km]
        measurement1sigma (float): Std dev of measurement noise [km]
        halfAngleDeg (float): Half-angle of visibility cone [deg]
        randomSeed (int, optional): RNG seed

    Returns:
        outputZkMat (np.ndarray): [n_visible x 4] matrix [x_B, y_B, z_B, landmarkID]
    """

    if randomSeed is not None:
        np.random.seed(randomSeed)
    rng = np.random.default_rng()

    # Compute relative position vectors in M frame
    r_LB_M = trueLandmarks - r_BM_M_truth
    distances = np.linalg.norm(r_LB_M, axis=1)

    # Filter by distance
    withinDistance = distances < distanceThresholdKm
    r_LB_M = r_LB_M[withinDistance]
    landmarkIndices = np.where(withinDistance)[0]

    if r_LB_M.shape[0] == 0:
        return np.empty((0, 4))

    # Rotate relative vectors into the body frame
    r_LB_B = np.array([q_BM_truth.rotate(vec) for vec in r_LB_M])

    # Normalize and find cone angles
    r_hat_B = r_LB_B / np.linalg.norm(r_LB_B, axis=1, keepdims=True)
    coneAxis_M = -r_BM_M_truth 
    coneAxis_B = q_BM_truth.rotate(coneAxis_M)
    coneAxis_B = coneAxis_B / np.linalg.norm(coneAxis_B)
    cosAngles = r_hat_B @ coneAxis_B
    halfAngleRad = np.deg2rad(halfAngleDeg)

    # Filter by cone
    cosAngles = np.clip(cosAngles, -1.0, 1.0)
    withinCone = np.acos(cosAngles) < halfAngleRad

    r_LB_B_visible = r_LB_B[withinCone]
    visibleIndices = landmarkIndices[withinCone]

    # Add measurement noise
    noisyMeasurements = rng.normal(
        loc=r_LB_B_visible, scale=measurement1sigma, size=r_LB_B_visible.shape
    )

    # Append landmark indices
    outputZkMat = np.hstack((noisyMeasurements, visibleIndices.reshape(-1, 1)))

    return outputZkMat


### MCMF propagation ###
w_MN_M = np.array([0.0, 0.0, 2*np.pi/27.322/24/3600])
def dqdt_wrapper(t, q):
    return Quaternion.dqdt(t, w_MN_M, q)

def propagateMCMF(tkm,tk,q_MN_tkm):
    # --- attitude --- #
    sol = solve_ivp(
        fun=dqdt_wrapper,
        t_span=[tkm, tk],
        y0=q_MN_tkm,
        method='RK45',
        rtol=1e-9,
        atol=1e-9
    )

    q_MN_tk = sol.y[:, -1]

    # Normalize quaternion
    if q_MN_tk[-1] < 0:
        q_MN_tk = -q_MN_tk
    q_MN_tk /= np.linalg.norm(q_MN_tk)

    return q_MN_tk





if __name__ == "__main__":


    rng = np.random.seed(42)


    bodyRadius_km = 1737.4

    nLandmarks = 10000
    mapSigma = .01 #km
    
    trueLandmarks,mapLandmarks = generateLandmarks(
        randomSeed=rng,
        nPoints=nLandmarks,
        radiusBodyKm=bodyRadius_km,
        mapPos1sigma=mapSigma
    )

    # statistics
    errorMap = trueLandmarks-mapLandmarks
    std = np.std(errorMap,axis=0)

    # --- Plot ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # landmarks
    ax.scatter(trueLandmarks[:,0], trueLandmarks[:,1], trueLandmarks[:,2], 
            c='k', s=8, label='True Landmarks')
    ax.scatter(mapLandmarks[:,0], mapLandmarks[:,1], mapLandmarks[:,2], 
            c='r', s=8, label='Map Landmarks')

    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')
    ax.set_title('Generated Landmarks on Lunar Surface')
    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.show()

    # save to file
    landmarkPicklePath = os.path.join('data', "landmarks.pkl")
    with open(landmarkPicklePath, "wb") as f:
        pickle.dump({"trueLandmarks": trueLandmarks, "mapLandmarks": mapLandmarks}, f)

    print(f"Saved trueLandmarks and mapLandmarks to {landmarkPicklePath}")