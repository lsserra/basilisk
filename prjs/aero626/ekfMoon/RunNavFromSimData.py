import numpy as np
import os, sys, copy
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import pickle


# Add the basilisk root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# EKF
from EkfPoseEstimator import EkfPosVelState, MekfState, EkfPoseEstimator

# attitude helpers
from helpers.attitude import DCM
from helpers.attitude.Quaternion import Quaternion

# simululation assistance
from faciliateSimulation import generateLandmarks, propagateMCMF, getLandmarkMeasurements
from PlottingAnalysisTools import plot_landmark_innovations, plotPosVelStateErrorAnd3sigma, plotMekfAttitudeErrorAnd3Sigma


# initalize random seed 
random_seed = 42
# random_seed = 41
# random_seed = 40
rng = np.random.default_rng(random_seed)



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
gyro_time = sim_data["timeGyro"] * 1e-9 # to seconds

q_BN_truth = sim_data["q_BN_truth"]

print("Simulation data successfully unboxed.")

## limit sim time for testing ##
# idxCap = 25
# idxCap = 700
# idxCap = 1500
idxCap = None
if idxCap is not None:
    timeData = timeData[:idxCap]
    sc_pos = sc_pos[:idxCap,:]
    sc_vel = sc_vel[:idxCap,:]
    moon_pos = moon_pos[:idxCap,:]
    moon_vel = moon_vel[:idxCap,:]
    gyro_meas = gyro_meas[:idxCap,:]
    gyro_time = gyro_time[:idxCap]
    q_BN_truth = q_BN_truth[:idxCap]
    print(f"Simulation data truncated to {idxCap} steps for testing.")


## set measurement update at _ Hz ##
measFreq = 1.0 # Hz
simdt = timeData[1] - timeData[0]
measDt = 1.0 / measFreq
simIterPublishMeasBound = int(np.round(measDt / simdt))
measCounter = 0






with open("data/landmarks.pkl", "rb") as f:
    landmark_data = pickle.load(f)
trueLandmarks = landmark_data["trueLandmarks"]
mapLandmarks = landmark_data["mapLandmarks"]
print("Landmark data successfully unboxed.")


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
plotSimTime = []







## Initialize EKF ##
# state sizes
nx = 6
nz = 3
# ekf object
ekf = EkfPoseEstimator()
# Process Noise
psd_r = .1 # 
psd_Mdrdt = .01 # 
Qww_posVel = np.diag([psd_r,psd_r,psd_r,psd_Mdrdt,psd_Mdrdt,psd_Mdrdt])


### intial conditions
# time 
t0 = timeData[0]



# covariance 
sigma_r = 100. # km
sigma_Mdrdt = 1.1 # km/s

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

# fill Gaussian corrupted inital states
posVelState0.r_BM_M_mean = rng.normal(
        loc=r_BM_M0, scale=sigma_r/200, size=r_BM_M0.shape
    )
posVelState0.Mdrdt_BM_M_mean = rng.normal(
        loc=Mdrdt_BM_M0, scale=sigma_Mdrdt/200, size=Mdrdt_BM_M0.shape
    )


## Initial MEKF state obj ##
mekfState0 = MekfState(nx=nx)

# covariance
sigmaAtt = np.array((np.deg2rad(10),np.deg2rad(10),np.deg2rad(10))) # deg -> rad
sigmaGyroBias = np.deg2rad(.2)/3600 # deg/hr -> rad/s
Pxx0 = block_diag(np.diag(sigmaAtt),sigmaGyroBias*np.eye(3))
Pxx0 = Pxx0@Pxx0.T

# init sc body attitude
q_NM_0 = q_MN_0.inverse()
q_BN_0 = q_BN_truth[0]
q_BM_true_0 = q_BN_0*q_NM_0

# Gaussian currupted attitude
bodyErrorEulerVector= rng.normal(
        loc=np.zeros((3,1)), scale=np.deg2rad(0.1), size=np.zeros((3,1)).shape
    )
phi = np.linalg.norm(bodyErrorEulerVector)
ehat = bodyErrorEulerVector/phi
qbodyErrorEulerVector = Quaternion.from_axis_angle(axis=ehat.flatten(),angle=phi)
mekfState0.q_BMref = q_BM_true_0 * qbodyErrorEulerVector



# sigmaAtt = 9.4e-6 # rad^2
# sigmaGyroBias = 9.4e-13 # rad^2/s^2
# Pxx0 = block_diag(sigmaAtt*np.eye(3),sigmaGyroBias*np.eye(3))
mekfState0.Pxx = Pxx0
# time
mekfState0.t = t0
# process noise
Qmekf = 1e-12 * np.eye(6) # TODO check!
# pass IC's, process noise PSD to filter
ekf.initialize(
    initPosVelState=posVelState0,
    initMekfState=mekfState0,
    QPosVel=Qww_posVel,
    Qmekf=Qmekf) 



## landmark measurement initialization ##
# measurement noise
oneSigmaLandmarkMeas_eachAxis = 50.0 # km

# EKF measurement noise
PvvLM = oneSigmaLandmarkMeas_eachAxis**2 * np.eye(3)
ekf.Pvv = PvvLM
# load map 
ekf.loadLandmarkMap(trueLandmarks) 

# parameters for 'optical sensor suite'
radiusMoonkm = 1737.4 # km
relativeDistanceThresholdKm = np.linalg.norm(r_BM_M0) - radiusMoonkm + 100 # km
halfAngleConeFOVdeg = 85.



# main sim loop
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

        plotSimTime.append(tk)
        continue

    tkm = timeData[i-1]
    plotSimTime.append(tk)

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
    q_BM_true = q_BN_i * q_NM_i 
    q_BM_true.ensureScalarPos()
    q_BM_store.append(q_BM_true.as_array())


    



    ### EKF Progpagation ###
    # grab gyro meas
    w_BN_B = gyro_meas[i,:]
    ekf.propagate(toTime=tk, w_BN_B_meas=w_BN_B)

    ### EKF Measurement Update ###
    measCounter += 1
    didUpdate = False
    if measCounter >= simIterPublishMeasBound:
        measCounter = 0
        visibleLandmarks, PvvBodyFrame = getLandmarkMeasurements(
            r_BM_M_truth=r_BM_M,
            v_BM_M_truth=Mdrdt_BM_M,
            q_BM_truth=q_BM_true,
        distanceThresholdKm=relativeDistanceThresholdKm,  #  km
            trueLandmarks=trueLandmarks,
            measurement1sigma=oneSigmaLandmarkMeas_eachAxis,
            randomSeed=random_seed,
            halfAngleDeg=halfAngleConeFOVdeg,
            debugPlot = False
        )
        if visibleLandmarks.shape[0] > 0:
            ekf.updateWithLandmarks(
                z_meas_matrix = visibleLandmarks, 
                PvvBodyFrame=PvvBodyFrame,
                measTime=tk)
            didUpdate = True
            stopTimeLimit = 16.0
            runningPlots = True
            pauseTime = 1.
            if runningPlots and (tk>stopTimeLimit):
                tSim = np.array(copy.deepcopy(plotSimTime))
                ## plot data to analyze
                plotPosVelStateErrorAnd3sigma(r_TruthList=copy.deepcopy(r_BM_M_TruthStore),
                                            v_TruthList=copy.deepcopy(Mdrdt_BM_M_M_TruthStore),
                                            t_TruthNpArray=tSim,
                                            posVelEstListLog=copy.deepcopy(ekf.posVelState_log)
                                            )
                # plt.show(block=False)   # display immediately
                # plt.pause(pauseTime)          # keep open 5 seconds
                # plt.close('all') 

                plotMekfAttitudeErrorAnd3Sigma(mekfStateList=copy.deepcopy(ekf.mekfState_log),
                                            q_BM_TruthList=copy.deepcopy(q_BM_store),
                                            t_TruthNpArray=tSim)
                # plt.show(block=False)   # display immediately
                # plt.pause(pauseTime)          # keep open 5 seconds
                # plt.close('all') 

                plot_landmark_innovations(
                    copy.deepcopy(ekf.innovation_log),
                    xLabel="Time [s]",
                    title="EKF Landmark Innovations",
                    # measurementNoiseSigma=oneSigmaLandmarkMeas_eachAxis
                )
                # plt.show(block=False)   # display immediately
                # plt.pause(pauseTime)          # keep open 5 seconds
                # plt.close('all') 
                plt.show()
                
    if not didUpdate:
        # update solution timing 
        ekf.mx_mekf_post_tk = copy.deepcopy(ekf.mx_mekf_prior_tk)
        ekf.mx_posVel_post_tk = copy.deepcopy(ekf.mx_posVel_prior_tk)
        ekf.mx_full.t = tk

        # log data
        ekf.posVelState_log.append(copy.deepcopy(ekf.mx_posVel_post_tk))
        ekf.mekfState_log.append(copy.deepcopy(ekf.mx_mekf_post_tk))


    # manually get ready for next time
    ekf.mx_posVel_prior_tk_ = ekf.mx_posVel_post_tk
    ekf.mx_mekf_prior_tk_ = ekf.mx_mekf_post_tk

    if ekf.mx_posVel_prior_tk_.Pxx[0,0] > 200**2:
        break




    ## running error check
    r_error = r_BM_M.reshape(-1,1) - ekf.mx_posVel_post_tk.r_BM_M_mean
    v_error = Mdrdt_BM_M - ekf.mx_posVel_post_tk.Mdrdt_BM_M_mean
    angleDiff = Quaternion.computeEulerVecAttErrorFromQuats(q_ref=q_MN_store[i-1],q_est=q_MN_store[i])
    runningAttError = Quaternion.computeEulerVecAttErrorFromQuats(
        q_ref=Quaternion.from_array(q_BM_store[i]),
        q_est=ekf.mx_mekf_post_tk.q_BMref
    )

    foo=1






## grab ekf error state and reference state and make plots
posVelStateList = copy.deepcopy(ekf.posVelState_log)
mekfStateList = copy.deepcopy(ekf.mekfState_log)

# plot sim time
plotSimTime = np.array(plotSimTime)

## plot data to analyze
plotPosVelStateErrorAnd3sigma(r_TruthList=r_BM_M_TruthStore,
                              v_TruthList=Mdrdt_BM_M_M_TruthStore,
                              t_TruthNpArray=plotSimTime,
                              posVelEstListLog=posVelStateList
                              )

plotMekfAttitudeErrorAnd3Sigma(mekfStateList=mekfStateList,
                               q_BM_TruthList=q_BM_store,
                               t_TruthNpArray=plotSimTime)

plot_landmark_innovations(
    ekf.innovation_log,
    xLabel="Time [s]",
    title="EKF Landmark Innovations",
    measurementNoiseSigma=oneSigmaLandmarkMeas_eachAxis
)


plt.show()


