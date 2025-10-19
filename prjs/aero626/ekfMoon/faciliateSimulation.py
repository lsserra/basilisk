import os, sys
import numpy as np

import copy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import pickle

from EkfPoseEstimator import EkfErrorState, EkfReferenceState, EkfPoseEstimator

# attitude helpers
from helpers.attitude import DCM
from helpers.attitude.Quaternion import Quaternion

# Add the basilisk root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))





### create landmark measurements along trajectory ###

import numpy as np
def generateLandmarks(
    rN_km, 
    vN_kms, 
    mu, 
    bodyRadius_km, 
    landmarksDensity, 
    corridorWidth_km,
    randomSeed=None
):
    if randomSeed is not None:
        np.random.seed(randomSeed)

    landmarks_N_km = []

    rng = np.random.default_rng()


    r_hat = rN_km / np.linalg.norm(rN_km)
    v_hat = vN_kms / np.linalg.norm(vN_kms)

    # Orbital angular momentum direction (normal to orbit plane)
    z_hat = np.cross(r_hat, v_hat)
    z_hat /= np.linalg.norm(z_hat)

    # Remove z_hat component from r_hat to get in-plane reference
    r_inplane = r_hat - np.dot(r_hat, z_hat) * z_hat
    r_inplane /= np.linalg.norm(r_inplane)

    # y_hat completes the right-handed system
    y_hat = np.cross(z_hat, r_inplane)
    x_hat = r_inplane  # optional alias for clarity

    nLandmarksPerIter = 10
    thetaIncr = np.linspace(0,2*np.pi,10)
    maxThetaNoise = corridorWidth_km / bodyRadius_km 
    noisyTheta = rng.normal(loc=0, scale=maxThetaNoise/3,size=(len(thetaIncr),nLandmarksPerIter))

    for i in range(len(thetaIncr)):

        # rotate original position vecotr about orbit angular momentum
        theta_inc = thetaIncr[i]    
        r_i =  DCM.T3(theta_inc) * r_hat
        # project noise on it
        for j in range(nLandmarksPerIter):
            
            # project noise on unit vector
            noiseTheta_i = noisyTheta[i,j]
            rNoisy = r_i*noiseTheta_i
            rNoisy /= np.linalg.norm(rNoisy)
            landmarks_N_km.append(rNoisy)


    
    return landmarks_N_km




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

    # --- Example setup ---
    mu_moon = 4902.800066  # [km^3/s^2]
    bodyRadius_km = 1737.4

    # simple circular orbit
    r_mag = bodyRadius_km + 100.0  # 100 km altitude
    v_mag = np.sqrt(mu_moon / r_mag)

    # generate a simple trajectory (e.g., 1 orbit)
    n_points = 200
    theta = np.linspace(0, 2*np.pi, n_points)
    rN_km = np.column_stack((r_mag*np.cos(theta), r_mag*np.sin(theta), np.zeros_like(theta)))
    vN_kms = np.column_stack((-v_mag*np.sin(theta), v_mag*np.cos(theta), np.zeros_like(theta)))



    rN_km0 = rN_km[0,:]
    vN_km0 = vN_kms[0,:]
    # generate landmarks
    landmarks_N_km = generateLandmarks(
        rN_km0,
        vN_km0,
        mu_moon,
        bodyRadius_km,
        landmarksDensity=0.005,
        corridorWidth_km=50.0,
        randomSeed=42
    )


    landmarks_N_km = np.array(landmarks_N_km)
    # --- Plot ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # body (sphere)
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    x = bodyRadius_km * np.cos(u) * np.sin(v)
    y = bodyRadius_km * np.sin(u) * np.sin(v)
    z = bodyRadius_km * np.cos(v)
    ax.plot_surface(x, y, z, color='lightgray', alpha=0.5)

    # trajectory
    ax.plot(rN_km[:,0], rN_km[:,1], rN_km[:,2], 'b', label='Trajectory')

    # landmarks
    if len(landmarks_N_km) > 0:
        ax.scatter(landmarks_N_km[:,0], landmarks_N_km[:,1], landmarks_N_km[:,2], 
                c='r', s=8, label='Landmarks')

    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')
    ax.set_title('Generated Landmarks on Lunar Surface')
    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.show()