# constants.py
import os

# Root dir for sim data (you already have this)
PATH2SIMDATADIR = "/Users/lukeserrano/repos/personal/basilisk/prjs/aero626/data"

SIM_DIR_PREFIX = "sim-"               # do NOT put numbers here
FILTER_OUTDIR_STRING = "FILTER_OUTPUT"
DATA_DIR_STRING   = "data"

SIM_CONFIG_STRING = "simulation_config.txt"
EKF_CONFIG_STRING = "EKF_config.txt"


# PKL output file name
EKF_PKL_FILENAME = "FILTER_SOL_AND_SIM_TRUTH.pkl"

# Keys inside the pickle dict
PKL_EKF_POSVEL_STATE_KEY   = "posVelStateList"
PKL_EKF_MEKF_STATE_KEY     = "mekfStateList"
PKL_INNOVATION_LIST_KEY    = "innObjList"

PKL_TRUTH_POS_KEY          = "r_BM_M_TruthStoreList"
PKL_TRUTH_ATT_KEY          = "q_BM_TruthStoreList"
PKL_TRUTH_VEL_KEY        = "Mdrdt_BM_M_M_TruthStoreList"
PKL_TRUTH_TBODY2LVLH_KEY = "TBodyToLVLH_TruthStoreList"
PKL_TRUTH_LVLH_ONESIG_MEAS_NOISE = "lvlh_oneSigmaArrayInput_TruthStoreList"
PKL_TRUTH_GRYOBIAS_KEY   = "trueGyroBias"


PKL_SIM_TIME_KEY           = "SimTimeStore"
