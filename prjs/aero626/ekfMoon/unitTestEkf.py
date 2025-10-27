import os, sys
import numpy as np

import copy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.linalg import block_diag

import pickle
from EkfPoseEstimator import EkfPosVelState, MekfState, EkfPoseEstimator

# attitude helpers
from helpers.attitude import DCM
from helpers.attitude.Quaternion import Quaternion

from faciliateSimulation import generateLandmarks, propagateMCMF

# Add the basilisk root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))




with open("data/MoonCentralBody_MoonGrav.pkl", "rb") as f:
# with open("data/MoonCentralBody_MoonEarthGrav.pkl", "rb") as f:
    sim_data = pickle.load(f)

timeData = sim_data["time"] * 1e-9 # to seconds
sc_pos   = sim_data["sc_pos"] / 1000 # to km
sc_vel   = sim_data["sc_vel"] / 1000
moon_pos = sim_data["moon_pos"] / 1000
moon_vel = sim_data["moon_vel"] / 1000
# earth_pos = sim_data["earth_pos"] / 1000
# earth_vel = sim_data["earth_vel"] / 1000
gyro_meas = sim_data["gryoAngVel"] # rad/s
gryo_time = sim_data["timeGyro"] * 1e-9 # to seconds

q_BN_truth = sim_data["q_BN_truth"]

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



# lets set up a for loop over sim time

r_BM_M_TruthStore = []
Mdrdt_BM_M_M_TruthStore = [] # time derivative of position B wrt M, as seen from M, coordinatized in M
q_MN_store = [] # MCMF defintion
q_MN_tkm =q_MN_0.as_array()

# list for MCMF to body truth attitude
q_BM_store = []


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
ekf = EkfPoseEstimator()
# Process Noise
psd_r = 0.01 # 
psd_Mdrdt = 0.001 # 
Qww_posVel = np.diag([psd_r,psd_r,psd_r,psd_Mdrdt,psd_Mdrdt,psd_Mdrdt])


### intial conditions
# time 
t0 = timeData[0]

# covariance 
sigma_r = 0.1 # m
sigma_Mdrdt = 0.01 # m/s

## Initial Pos Vel state obj ##
posVelState0 = EkfPosVelState(nx=nx)

Pxx0 = np.diag([sigma_r,sigma_r,sigma_r,sigma_Mdrdt,sigma_Mdrdt,sigma_Mdrdt])
Pxx0 = Pxx0 @ Pxx0.T
posVelState0.Pxx = Pxx0
posVelState0.t = t0

# MCMF position and velocity
r_BM_M0 = q_MN_0.rotate(sc_pos[0,:])
rdot_BM_M = q_MN_0.rotate(sc_vel[0,:])
Mdrdt_BM_M0 = rdot_BM_M - (np.cross(w_MN_M, r_BM_M0))
# fill
posVelState0.r_BM_M_mean = r_BM_M0
posVelState0.Mdrdt_BM_M_mean = Mdrdt_BM_M0

## Initial MEKF state obj ##
mekfState0 = MekfState(nx=nx)
# init sc body attitude
q_NM_0 = q_MN_0.inverse()
q_BN_0 = q_BN_truth[0]
q_BM_true_0 = q_BN_0*q_NM_0
mekfState0.q_BMref = q_BM_true_0

# covariance
sigmaAtt = np.deg2rad(5.)
sigmaGyroBias = np.deg2rad(1.)
Pxx0 = block_diag(sigmaAtt*np.eye(3),sigmaGyroBias*np.eye(3))
Pxx0 = Pxx0@Pxx0.T
mekfState0.Pxx = Pxx0

# time
mekfState0.t = t0

# process noise
Qmekf = 0.00000 * np.eye(6) # TODO check!



# pass IC's, process noise PSD to filter
ekf.initialize(
    initPosVelState=posVelState0,
    initMekfState=mekfState0,
    QPosVel=Qww_posVel,
    Qmekf=Qmekf) 




# generate some landmarks

eqRadMoon = 1737.4e3 # km

for i, tk in enumerate(timeData):
    if i == 0:
        
        # MCMF and sc body attitude 
        q_MN_store.append(q_MN_0)
        q_NM_0 = q_MN_0.inverse()
        q_BN_0 = q_BN_truth[i]
        q_BM_true_array = (q_BN_0*q_NM_0).as_array()
        q_BM_store.append(q_BM_true_array)

        # initial r_BM_M and Mdrdt_BM_M
        r_BM_M0 = q_MN_0.rotate(sc_pos[0,:])
        rdot_BM_M = q_MN_0.rotate(sc_vel[0,:])
        Mdrdt_BM_M0 = rdot_BM_M - (np.cross(w_MN_M, r_BM_M0))
        
        r_BM_M_TruthStore.append(r_BM_M0)
        Mdrdt_BM_M_M_TruthStore.append(Mdrdt_BM_M0)
        continue

    tkm = timeData[i-1]

    ### MCMF TRUTH GENERATION ###
    # --- MCMF Coordinate Frame --- #
    q_MN_tk = propagateMCMF(
        tkm=tkm,
        tk=tk,
        q_MN_tkm=q_MN_tkm)
    
    q_MN_tk_obj = Quaternion.from_array(q_MN_tk)
    q_MN_store.append(q_MN_tk_obj)
    q_MN_tkm = q_MN_tk

    # --- position --- #
    r_BM_N = sc_pos[i,:]
    r_BM_M = q_MN_tk_obj.rotate(r_BM_N)
    r_BM_M_TruthStore.append(r_BM_M)

    # --- velocity --- #
    rdot_BM_M = q_MN_tk_obj.rotate(sc_vel[i,:])
    Mdrdt_BM_M = rdot_BM_M - (np.cross(w_MN_M, r_BM_M))
    Mdrdt_BM_M_M_TruthStore.append(Mdrdt_BM_M)

    # --- Attitude --- #
    q_NM_i = q_MN_tk_obj.inverse()
    q_BN_i = q_BN_truth[i]
    q_BM_true_array = (q_BN_i*q_NM_i).as_array()
    q_BM_store.append(q_BM_true_array)


    



    ### EKF Progpagation ###
    # grab gyro meas
    w_BN_B = gyro_meas[i,:]
    ekf.propagate(toTime=tk, w_BM_B_meas=w_BN_B)

    # manually get ready for next time
    ekf.mx_posVel_prior_tk_ = ekf.mx_posVel_prior_tk
    ekf.mx_mekf_prior_tk_ = ekf.mx_mekf_prior_tk


    ## running error check

    r_error = r_BM_M - ekf.mx_posVel_post_tk.r_BM_M_mean
    v_error = Mdrdt_BM_M - ekf.mx_posVel_post_tk.Mdrdt_BM_M_mean
    angleDiff = Quaternion.computeEulerVecAttErrorFromQuats(q_ref=q_MN_store[i-1],q_est=q_MN_store[i])

    foo=1






# grab ekf error state and reference state and make plots
posVelStateList = copy.deepcopy(ekf.posVelState_log)
mekfStateList = copy.deepcopy(ekf.mekfState_log)


############################################
# Position and Velocity filter state
# extraction, interpolation of truth, and error calculation
############################################
t_filt = np.array([s.t for s in posVelStateList])
Pxx_list = [s.Pxx for s in posVelStateList]
P_diag_posVel = np.array([np.diag(P) for P in Pxx_list])
sigma3_posVel = 3 * np.sqrt(P_diag_posVel)

# Extract reference state
r_filt = np.array([xref.r_BM_M_mean for xref in posVelStateList])
v_filt = np.array([xref.Mdrdt_BM_M_mean for xref in posVelStateList])

# Extract true state
r_truth = np.array(r_BM_M_TruthStore)
v_truth = np.array(Mdrdt_BM_M_M_TruthStore)

r_interp_truth = interp1d(timeData, r_truth, axis=0)
r_true_interp = r_interp_truth(t_filt)

v_interp_truth = interp1d(timeData, v_truth, axis=0)
v_true_interp = v_interp_truth(t_filt)


# Compute estimation error
positionError = r_true_interp - r_filt
velocityError = v_true_interp - v_filt
nSolutions = len(r_filt)

print(f"Final r Error = {positionError[-1,:]} [km/s]")
print(f"Final v Error = {velocityError[-1,:]} [km/s]")







############################################
# MEKF filter state
# extraction, interpolation of truth, and error calculation
############################################
t_filt = np.array([s.t for s in mekfStateList])
Pxx_list = [s.Pxx for s in mekfStateList]
P_diag_mekf = np.array([np.diag(P) for P in Pxx_list])
sigma3_mekf = 3 * np.sqrt(P_diag_mekf)

# Extract reference state
q_BM_filt_list = np.array([mx.q_BMref for mx in mekfStateList])
gyroBias_filt_array = np.array([mx.gyroBiasRef for mx in mekfStateList])

# interpolate truth solution
q_BM_truth_array = np.array(q_BM_store)
q_BM_interp1dObj_truth = interp1d(timeData, q_BM_truth_array, axis=0)
q_BM_true_interp = q_BM_interp1dObj_truth(t_filt)
gryoBiasTruth = np.zeros(gyroBias_filt_array.shape) # TODO grab from basilisk

# compute attitude error as principle rotation vector
# body attitude error list
PRV_BprimeB_list = []
for i in range(q_BM_filt_list.shape[0]):
    # compute attitude error and store
    q_BM_true = Quaternion.from_array(q_BM_true_interp[i,:]).normalize()
    q_BM_filt = q_BM_filt_list[i]
    prv_BprimeB = Quaternion.computeEulerVecAttErrorFromQuats(
        q_ref=q_BM_true,
        q_est=q_BM_filt
    )
    PRV_BprimeB_list.append(prv_BprimeB)

PRV_BprimeB_array = np.array(PRV_BprimeB_list)

# gryo bias error 
gyroBiasError_array = gryoBiasTruth - gyroBias_filt_array



############################################
# Plot position and velocity estimation errors
############################################
fig, axs = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
pos_labels = ['X', 'Y', 'Z']
vel_labels = ['X', 'Y', 'Z']

# Position error plots
for i in range(3):
    axs[i, 0].plot(t_filt, positionError[:, i], 'k-', linewidth=1.8, label=f'{pos_labels[i]}')
    axs[i, 0].plot(t_filt, sigma3_posVel[:, i], 'r--', linewidth=1)
    axs[i, 0].plot(t_filt, -sigma3_posVel[:, i], 'r--', linewidth=1, label='±3σ confidence')
    axs[i, 0].set_ylabel(f'{pos_labels[i]} [km]')
    axs[i, 0].grid(True)
    axs[i, 0].legend(loc='upper right')
    axs[0,0].set_title("r_BM_M Estimation Error")

# Velocity error plots
for i in range(3):
    axs[i, 1].plot(t_filt, velocityError[:,i], 'k-', linewidth=1.8, label=f'{vel_labels[i]}')
    axs[i, 1].plot(t_filt, sigma3_posVel[:, 3 + i], 'r--', linewidth=1)
    axs[i, 1].plot(t_filt, -sigma3_posVel[:, 3 + i], 'r--', linewidth=1,label='±3σ confidence')
    axs[i, 1].set_ylabel(f'{vel_labels[i]} [km/s]')
    axs[i, 1].grid(True)
    axs[i, 1].legend(loc='upper right')
    axs[0,1].set_title("Md(r_BM_M)/dt Estimation Error")

axs[-1, 0].set_xlabel('Time [s]')
axs[-1, 1].set_xlabel('Time [s]')
fig.suptitle(f"MCMF r_BM_M & Md(.)dt Estimation Errors ±3σ\n{nSolutions} EKF Steps", fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])





############################################
# Plot attitude and gyro bias estimation errors
############################################
fig, axs = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
body_labels = ['X', 'Y', 'Z']
gryo_labels = ['X', 'Y', 'Z']

# Position error plots
for i in range(3):
    axs[i, 0].plot(t_filt, np.rad2deg(PRV_BprimeB_array[:, i]), 'k-', linewidth=1.8, label=f'{body_labels[i]}')
    axs[i, 0].plot(t_filt, np.rad2deg(sigma3_mekf[:, i]), 'r--', linewidth=1)
    axs[i, 0].plot(t_filt, np.rad2deg(-sigma3_mekf[:, i]), 'r--', linewidth=1, label='±3σ confidence')
    axs[i, 0].set_ylabel(f'Body Frame {body_labels[i]} Error [deg]')
    axs[i, 0].grid(True)
    axs[i, 0].legend(loc='upper right')
    axs[0,0].set_title("Body Frame MCMF Attitude Error as PRV")

# Velocity error plots
for i in range(3):
    axs[i, 1].plot(t_filt, np.rad2deg(gyroBiasError_array[:,i]), 'k-', linewidth=1.8, label=f'{body_labels[i]}')
    axs[i, 1].plot(t_filt, np.rad2deg(sigma3_mekf[:, 3 + i]), 'r--', linewidth=1)
    axs[i, 1].plot(t_filt, np.rad2deg(-sigma3_mekf[:, 3 + i]), 'r--', linewidth=1,label='±3σ confidence')
    axs[i, 1].set_ylabel(f'Gyro Frame {body_labels[i]} Bias Error [deg/s]')
    axs[i, 1].grid(True)
    axs[i, 1].legend(loc='upper right')
    axs[0,1].set_title("Gryo Bias Error")

axs[-1, 0].set_xlabel('Time [s]')
axs[-1, 1].set_xlabel('Time [s]')
fig.suptitle(f"MEKF Estimation Errors ±3σ\n{nSolutions} EKF Steps", fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


