import os, sys
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import pickle


from EkfPoseEstimator import EkfErrorState, EkfReferenceState, EkfPoseEstimator

# attitude helpers
from helpers.attitude import DCM
from helpers.attitude.Quaternion import Quaternion

# Add the basilisk root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))








# # Example: 100 km altitude circular orbit
# MU_MOON = 4902.799 * 1e9 # m^3/s^3
# r_test = np.array([1737.4e3 + 100e3, 0, 0])
# v_circ = np.sqrt(MU_MOON / np.linalg.norm(r_test))
# x0 = np.hstack((r_test, [0, v_circ, 0]))

# xdot = EkfPoseEstimator.McmfPoseDynamics(0, x0)
# print(xdot[3:])







with open("data/MoonCentralBody_MoonGrav.pkl", "rb") as f:
    sim_data = pickle.load(f)

timeData = sim_data["time"] * 1e-9 # to seconds
sc_pos   = sim_data["sc_pos"]
sc_vel   = sim_data["sc_vel"]
moon_pos = sim_data["moon_pos"]
moon_vel = sim_data["moon_vel"]
earth_pos = sim_data["earth_pos"]
earth_vel = sim_data["earth_vel"]

print("Simulation data successfully unboxed.")



# DCM definition of MCMF frame
lunarObliquityToEcliptic = 1.54 # deg

zhat = np.array([0.,0.,1.])
xhat = np.array([1.,0.,0.])
mhat3 = DCM.T2(np.deg2rad(lunarObliquityToEcliptic)) @ zhat
xhat1_orthog = xhat - np.dot(xhat,mhat3) * xhat
rhat1_orthog = xhat1_orthog / np.linalg.norm(xhat1_orthog)
mhat2 = np.cross(mhat3,rhat1_orthog)
mhat1 = np.cross(mhat2,mhat3)
T_MN = np.column_stack([mhat1,mhat2,mhat3])

cosang = np.clip(np.dot(mhat3, zhat), -1.0, 1.0)
angle_deg = np.rad2deg(np.arccos(cosang))


q_MN_0 = Quaternion.from_DCM(T_MN)
q_MN_0.ensureScalarPos()
q_MN_0.normalize()



# let's propagate truth MCMF frame attitude
# w_RN_R = np.array([0.0, 0.0, 2*np.pi/29.530/24/3600])
w_MN_M = np.array([0.0, 0.0, 2*np.pi/27.322/24/3600])


def dqdt_wrapper(t, q):
    return Quaternion.dqdt(t, w_MN_M, q)

# lets set up a for loop over sim time

r_BM_M_TruthStore = []
Mdrdt_BM_M_M_TruthStore = [] # time derivative of position B wrt M, as seen from M, coordinatized in M
q_MN_store = []
q_MN_tkm =q_MN_0.as_array()


# take vector in M and map to N, then plot traj
r_PM_M = np.array([1737.4e3 + 1.e3 , 0.0, 0.0])
eclipticPlane_r_PM_M = None
r_PM_N_store = []
eclipticPlane_r_PM_M_store = []



## Initialize EKF ##
# state sizes
nx = 6
nz = 3
# ekf object
ekf = EkfPoseEstimator(nx=nx,nz=nz)
# Process Noise
psd_r = 0.01 # 
psd_Mdrdt = 0.001 # 
Qww = np.diag([psd_r,psd_r,psd_r,psd_Mdrdt,psd_Mdrdt,psd_Mdrdt])
ekf.Qww = Qww*Qww.T # squared
ekf.Fw = np.eye(nx)

### intial conditions
# time 
t0 = timeData[0]

# covariance 
sigma_r = 0.1 # m
sigma_Mdrdt = 0.01 # m/s

## error state
errState0 = EkfErrorState(nx=nx)
errState0.mx = np.zeros((nx,1))
Pxx0 = np.diag([sigma_r,sigma_r,sigma_r,sigma_Mdrdt,sigma_Mdrdt,sigma_Mdrdt])
Pxx0 = Pxx0 @ Pxx0.T
errState0.Pxx = Pxx0
errState0.t = t0

## reference state
xRef0 = EkfReferenceState()
    # is true state for now
r_BM_M0 = q_MN_0.rotate(sc_pos[0,:])
rdot_BM_M = q_MN_0.rotate(sc_vel[0,:])
Mdrdt_BM_M0 = rdot_BM_M - (np.cross(w_MN_M, r_BM_M0))
    # fill
xRef0.r_BM_M = r_BM_M0
xRef0.Mdrdt_BM_M = Mdrdt_BM_M0
xRef0.t = t0

# pass to ekf obj
ekf.initialize(mx_prior_tk_= errState0,
               xRef_tk_ = xRef0)


for i, tk in enumerate(timeData):
    if i == 0:
        q_MN_store.append(q_MN_0)
        continue

    tkm = timeData[i-1]
    ### MCMF TRUTH GENERATION ###
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

    qObj = Quaternion.from_array(q_MN_tk)
    q_MN_store.append(qObj)
    q_MN_tkm = q_MN_tk

    q_MN_tk_obj = qObj


    # --- position --- #
    r_BM_N = sc_pos[i,:]
    r_BM_M = q_MN_tk_obj.rotate(r_BM_N)
    r_BM_M_TruthStore.append(r_BM_M)

    # --- velocity --- #
    rdot_BM_M = q_MN_tk_obj.rotate(sc_vel[i,:])
    Mdrdt_BM_M = rdot_BM_M - (np.cross(w_MN_M, r_BM_M))
    Mdrdt_BM_M_M_TruthStore.append(Mdrdt_BM_M)


    ## foo position vector of point p
    Nhat3 = np.array([0., 0., 1.])
    Nhat3_M = q_MN_tk_obj.rotate(Nhat3)
    # remove portion of vector normal to eclliptic plane
    eclipticPlane_r_PM_M = r_PM_M - np.dot(r_PM_M,Nhat3_M)*Nhat3_M
    eclipticPlane_r_PM_M_store.append(eclipticPlane_r_PM_M)
    q_NM_tk_obj = q_MN_tk_obj.inverse()
    q_NM_tk_obj.normalize()
    r_PM_N_store.append(q_NM_tk_obj.rotate(eclipticPlane_r_PM_M))


    ### EKF Progpagation ###
    ekf.propagate(toTime=tk)

    # manually get ready for next time
    ekf.xRef_tk_ = ekf.xRef_tk
    ekf.mx_prior_tk_ = ekf.mx_prior_tk






# grab ekf error state and reference state and make plots
errorStateList = copy.deepcopy(ekf.errorState_log)
referenceStateList = copy.deepcopy(ekf.refState_log)



# Extract times and covariance diagonals
t_filt = np.array([s.t for s in errorStateList])
Pxx_list = [s.Pxx for s in errorStateList]
P_diag = np.array([np.diag(P) for P in Pxx_list])
sigma3 = 3 * np.sqrt(P_diag)

# Extract reference state
r_filt = np.array([xref.r_BM_M for xref in referenceStateList])
v_filt = np.array([xref.Mdrdt_BM_M for xref in referenceStateList])

# Extract true state
r_truth = np.array(r_BM_M_TruthStore)
v_truth = np.array(Mdrdt_BM_M_M_TruthStore)

# Compute estimation error
positionError = r_truth - r_filt
velocityError = v_truth - v_filt
nSolutions = len(r_filt)

############################################
# Plot position and velocity estimation errors
############################################
fig, axs = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
pos_labels = ['X', 'Y', 'Z']
vel_labels = ['X', 'Y', 'Z']

# Position error plots
for i in range(3):
    axs[i, 0].plot(t_filt, positionError[:, i], 'k-', linewidth=1.8, label=f'{pos_labels[i]}')
    axs[i, 0].plot(t_filt, sigma3[:, i], 'r--', linewidth=1)
    axs[i, 0].plot(t_filt, -sigma3[:, i], 'r--', linewidth=1)
    axs[i, 0].set_ylabel(f'{pos_labels[i]} [m]')
    axs[i, 0].grid(True)
    axs[i, 0].legend(loc='upper right')
    axs[i,0].title("r_BM_M Estimation Error")

# Velocity error plots
for i in range(3):
    axs[i, 1].plot(t_filt, velocityError[:,i], 'k-', linewidth=1.8, label=f'{vel_labels[i]}')
    axs[i, 1].plot(t_filt, sigma3[:, 3 + i], 'r--', linewidth=1)
    axs[i, 1].plot(t_filt, -sigma3[:, 3 + i], 'r--', linewidth=1)
    axs[i, 1].set_ylabel(f'{vel_labels[i]} [m/s]')
    axs[i, 1].grid(True)
    axs[i, 1].legend(loc='upper right')
    axs[i,0].title("Md(r_BM_M)/dt Estimation Error")

axs[-1, 0].set_xlabel('Time [s]')
axs[-1, 1].set_xlabel('Time [s]')
fig.suptitle(f"MCMF r_BM_M & Md(.)dt Estimation Errors ±3σ\n{nSolutions} EKF Steps", fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



