import sys, os
import matplotlib.pyplot as plt
import pickle

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
PATH2SIMDATADIR = "/Users/lukeserrano/repos/personal/basilisk/prjs/aero626/data"


runNum = 32
SAVE_FIGURES = True

pth2data = os.path.join(PATH2SIMDATADIR, f"sim-{runNum:04d}")


from prjs.aero626.analysis.AnalysisTools import PoseAnalyzer


## EGMF
egmfPA = PoseAnalyzer()
pklPath = os.path.join(pth2data,"EGMF.pkl")

if os.path.exists(pklPath):

    # manually pull gyro data
    with open(pklPath, "rb") as f:
                    data = pickle.load(f)

    truth_gyroBias = data.get('truth_gyroBias', None)
    egmf_est_gyroBias = data.get('est_gyroBias', None)

    egmfPA._solutionName = "EGMF"
    egmfPA._EXPORT_FIGURES_FLAG = SAVE_FIGURES
    egmfPA._dataDir = pth2data
    egmfPA.LoadFromPklFile(pkl_path=pklPath)
    egmfPA.plotTransError3sigma()
    egmfPA.plotAttError3Sigma()

    # gyro bias plot 
    # rad to deg
    RAD2DEG2_PER_S2_TO_DEG2_PER_HR2 = (180/np.pi)**2 * (3600**2)
    est_Pxx =egmfPA._est_Pxx[:,9:,9:] * RAD2DEG2_PER_S2_TO_DEG2_PER_HR2
    truth_gyroBias = np.rad2deg(truth_gyroBias)*3600
    egmf_est_gyroBias = np.rad2deg(egmf_est_gyroBias)*3600

    egmfPA.plot3DimVecError3Sigma(
                                truth_v=truth_gyroBias,
                                truth_t=egmfPA._truth_t,
                                est_v=egmf_est_gyroBias,
                                est_Pxx=est_Pxx,
                                est_t=egmfPA._est_t,
                                strFigureTitle='EGMF Gryo Bias Error',
                                strXlabel='Time [s]',
                                strYunits='[Deg/Hr]'
            
    )


# # compute EGMF stats
# errors = egmfPA.compute_errors()
# print("EGMF Statistics:")
# print("Position RMSE:", errors["position"]["RMSE"], errors["position"]["units"])
# print("Position MAE:", errors["position"]["MAE"], errors["position"]["units"])
# print("Velocity RMSE:", errors["velocity"]["RMSE"], errors["velocity"]["units"])
# print("Velocity MAE:", errors["velocity"]["MAE"], errors["velocity"]["units"])
# print("Attitude RMSE:", errors["attitude"]["RMSE"], errors["attitude"]["units"])


## EKF
ekfPA = PoseAnalyzer()
pklPath = os.path.join(pth2data,"EKF.pkl")

if os.path.exists(pklPath):
    # manually pull gyro data
    with open(pklPath, "rb") as f:
                data = pickle.load(f)

    truth_gyroBias = data.get('truth_gyroBias', None)
    ekf_est_gyroBias = data.get('est_gyroBias', None)

    ekfPA._solutionName = "EKF"
    # ekfPA._NO_TITLE_FLAG = True
    ekfPA._EXPORT_FIGURES_FLAG = SAVE_FIGURES
    ekfPA._dataDir = pth2data
    ekfPA.LoadFromPklFile(pkl_path=pklPath)
    ekfPA.plotTransError3sigma()
    ekfPA.plotAttError3Sigma()


    # gyro bias plot 
    # rad to deg
    RAD2DEG2_PER_S2_TO_DEG2_PER_HR2 = (180/np.pi)**2 * (3600**2)
    est_Pxx =ekfPA._est_Pxx[:,9:,9:] * RAD2DEG2_PER_S2_TO_DEG2_PER_HR2

    truth_gyroBias = np.rad2deg(truth_gyroBias)*3600
    ekf_est_gyroBias = np.rad2deg(ekf_est_gyroBias)*3600

    ekfPA.plot3DimVecError3Sigma(
                                truth_v=truth_gyroBias,
                                truth_t=ekfPA._truth_t,
                                est_v=ekf_est_gyroBias,
                                est_Pxx=est_Pxx,
                                est_t=ekfPA._est_t,
                                strFigureTitle='EKF Gryo Bias Error',
                                strXlabel='Time [s]',
                                strYunits='Deg/Hr'
            
    )


    ## analyze ekf innovations
    # pull data

    inn_array = data.get('inn_array', None)
    inn_t = data.get('inn_t', None)
    inn_cov = data.get('inn_cov', None)
    inn_lm_id = data.get('inn_lm_id', None)
    meas_noise_one_sigma_lvlh = data.get('meas_noise_one_sigma_lvlh',None)

    ekfPA.plot_landmark_innovations_lvlh(
        innArray = inn_array,
        innTime_array = inn_t,
        innCov = inn_cov,
        landmark_ids = inn_lm_id,
        est_r = ekfPA._est_r,
        est_v = ekfPA._est_v,
        est_q_array = ekfPA._est_q,
        lvlh_oneSigmaArrayInput = meas_noise_one_sigma_lvlh,
        xLabel="Time [s]",
        title="Landmark Innovations LVLH Frame",
        unitString = "km",
        show_measurement_noise=True,
        show_confidence=True,
        figsize=(8,5),
        _PLOT_BODY = False
        )

# compute EGMF stats
# errors = ekfPA.compute_errors()
# print("EKF Statistics:")
# print("Position RMSE:", errors["position"]["RMSE"], errors["position"]["units"])
# print("Position MAE:", errors["position"]["MAE"], errors["position"]["units"])
# print("Velocity RMSE:", errors["velocity"]["RMSE"], errors["velocity"]["units"])
# print("Velocity MAE:", errors["velocity"]["MAE"], errors["velocity"]["units"])


plt.show()

