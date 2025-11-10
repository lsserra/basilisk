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


# landmark feeding logic
def getLandmarkMeasurements(r_BM_M_truth, q_BM_truth, distanceThresholdKm, trueLandmarks):
    """Pass measurement as the relative position between the true landmark
      wrt true body position rotated into the true body frame"""
    # compute distances from the current position to each landmark
    relativePositionVectors = trueLandmarks - r_BM_M_truth
    distances = np.linalg.norm(relativePositionVectors, axis=1)

    # find landmarks within the distance threshold
    validLandmarks = trueLandmarks[distances < distanceThresholdKm]

    # ID the landmarks with the row indices
    landmarkIndices = np.where(distances < distanceThresholdKm)[0]

    outputZkMat = np.zeros((len(validLandmarks), 3))

    for i in range(validLandmarks.shape[0]):
        # rotate to body frame
        r_LB_M = validLandmarks[i] - r_BM_M_truth
        r_LB_B = q_BM_truth.rotate(r_LB_M).reshape((1, 3))

        outputZkMat[i,:] = r_LB_B

    # get the valid landmark positions
    outputZkMat = np.hstack((outputZkMat,landmarkIndices.reshape(-1,1)))

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

    nLandmarks = 2000
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