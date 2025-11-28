import pickle, sys, os
from matplotlib import pyplot as plt

# Add the basilisk root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from prjs.aero626.analysis.PlottingAnalysisTools import(   
 plot_landmark_innovations,
 plotPosVelStateErrorAnd3sigma,
 plotMekfAttitudeErrorAnd3Sigma,
 plot_landmark_innovations_lvlh)

from prjs.aero626.constants import PATH2SIMDATADIR


# --- Add compatibility alias for pickle ---
import prjs.aero626.ekfMoon.EkfPoseEstimator as ekf_mod
sys.modules['EkfPoseEstimator'] = ekf_mod

# Now normal imports work fine
from prjs.aero626.ekfMoon.EkfPoseEstimator import (
    EkfPosVelState, MekfState, EkfPoseEstimator, LandMarkInnovation
)
# attitude helpers
from helpers.attitude import DCM
from helpers.attitude.Quaternion import Quaternion
from prjs.aero626.constants import (
        PKL_SIM_TIME_KEY,
        PKL_TRUTH_ATT_KEY,
        PKL_TRUTH_POS_KEY,
        PKL_TRUTH_VEL_KEY,
        PKL_EKF_MEKF_STATE_KEY,
        PKL_TRUTH_GRYOBIAS_KEY,
        PKL_INNOVATION_LIST_KEY,
        PKL_EKF_POSVEL_STATE_KEY,
        PKL_TRUTH_TBODY2LVLH_KEY,
        PKL_TRUTH_LVLH_ONESIG_MEAS_NOISE
    )

def load_filter_solution(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    return {
        "posVelStateList":   data[PKL_EKF_POSVEL_STATE_KEY],
        "mekfStateList":     data[PKL_EKF_MEKF_STATE_KEY],
        "innObjList":        data[PKL_INNOVATION_LIST_KEY],

        "truth_pos":         data[PKL_TRUTH_POS_KEY],
        "truth_att":         data[PKL_TRUTH_ATT_KEY],
        "truth_vel":       data[PKL_TRUTH_VEL_KEY],
        "truth_gryoBias":       data[PKL_TRUTH_GRYOBIAS_KEY],
        "truth_TBodyToLVLH": data[PKL_TRUTH_TBODY2LVLH_KEY],
        "truth_lvlh_oneSigMeasNoise": data[PKL_TRUTH_LVLH_ONESIG_MEAS_NOISE],

        "SimTime":           data[PKL_SIM_TIME_KEY],
    }






pth2data= os.path.join(PATH2SIMDATADIR,"sim-0002/FILTER_SOL_AND_SIM_TRUTH.pkl")
pkg = load_filter_solution(pth2data)

posVelState   = pkg["posVelStateList"]
mekfState     = pkg["mekfStateList"]
innovation    = pkg["innObjList"]

truth_TBodyToLvlh = pkg["truth_TBodyToLVLH"]
lvlhOneSigMeasNoise = pkg["truth_lvlh_oneSigMeasNoise"]


truth_r       = pkg["truth_pos"]
truth_q       = pkg["truth_att"]
truth_vel   = pkg["truth_vel"]
truth_gyroBias = pkg["truth_gryoBias"]

t             = pkg["SimTime"]




## plot data to analyze
plotPosVelStateErrorAnd3sigma(r_TruthList=truth_r,
                              v_TruthList=truth_vel,
                              t_TruthNpArray=t,
                              posVelEstListLog=posVelState
                              )

plotMekfAttitudeErrorAnd3Sigma(mekfStateList=mekfState,
                               q_BM_TruthList=truth_q,
                               gryoBiasTruthList=truth_gyroBias,
                               t_TruthNpArray=t)

plot_landmark_innovations(
    innovation,
    xLabel="Time [s]",
    title="EKF Landmark Innovations",
    # measurementNoiseSigma=oneSigmaLandmarkMeas_eachAxis
)

plot_landmark_innovations_lvlh(
                        innovations_log=(innovation),
                        T_BodyToLvlh_List=(truth_TBodyToLvlh),
                        lvlh_oneSigmaArrayInput=(lvlhOneSigMeasNoise)
                    )


plt.show()