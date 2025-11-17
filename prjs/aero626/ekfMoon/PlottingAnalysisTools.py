import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Add the basilisk root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
# attitude helpers
from helpers.attitude import DCM
from helpers.attitude.Quaternion import Quaternion


def plot_landmark_innovations(
    innovations_log,
    xLabel="Time [s]",
    title="Landmark Innovations, Body Frame Axis",
    unitString = "km",
    show_measurement_noise=True,
    measurementNoiseSigma=None,
    show_confidence=True,
    figsize=(8,5)
):
    """
    Plot landmark innovations with optional ±3σ confidence bounds.

    Args:
        innovations_log (list[LandMarkInnovation]): list of innovation objects
            with attributes:
                - t : timestamp (float)
                - innovation : (3,1) or (3,) numpy array
                - innovationCov : (3,3) numpy array
                - landmarkId : int
        xLabel (str): x-axis label
        title (str): plot title
        show_measurement_noise (bool): plot gray ±3σ noise lines if True
        measurementNoiseSigma (float|None): noise std deviation (km or m)
        show_confidence (bool): plot ±3σ confidence lines from innovationCov
        figsize (tuple): matplotlib figure size
    """

    # --- Parse logs ---
    innTime_array = np.array([entry.t for entry in innovations_log])
    inn_array = np.array([entry.innovation.flatten() for entry in innovations_log])  # shape (N, 3)
    innSigma3_array = np.array([
        3.0 * np.sqrt(np.diag(entry.innovationCov))
        for entry in innovations_log
    ])  # shape (N, 3)

    # --- Plot ---
    fig, axs = plt.subplots(4, 1, figsize=(figsize[0], figsize[1]+2), sharex=True)
    labels = [f"x {unitString}", f"y {unitString}", f"z {unitString}"]

    for i, ax in enumerate(axs):
        if i ==3:
            continue
        ax.scatter(innTime_array, inn_array[:, i], marker='x', color='k', label=f'Innovation {labels[i]}')

        # Optional measurement noise bounds
        if show_measurement_noise and measurementNoiseSigma is not None:
            ax.axhline(y=3 * measurementNoiseSigma, color='gray', linestyle='--', label='Measurement Noise ±3σ')
            ax.axhline(y=-3 * measurementNoiseSigma, color='gray', linestyle='--')

        # Optional ±3σ filter confidence bounds
        if show_confidence:
            ax.plot(innTime_array, innSigma3_array[:, i], '-r', label='Innovation ±3σ confidence')
            ax.plot(innTime_array, -innSigma3_array[:, i], '-r')

        ax.grid(True)
        ax.set_ylabel(f'{labels[i]}')
        if i == 0:
            ax.legend(loc='upper right')
        if i == len(axs)-1:
            ax.set_xlabel(xLabel)

    # --- 4th row: Landmark ID vs time ---
    ax_id = axs[3]
    landmark_ids = np.array([entry.landmarkId for entry in innovations_log])

    ax_id.scatter(innTime_array, landmark_ids, marker='o', s=12, color='b')
    ax_id.set_ylabel("ID")
    ax_id.grid(True)
    ax_id.set_xlabel(xLabel)



    fig.suptitle(title)
    plt.tight_layout()
    #plt.show()


def plotPosVelStateErrorAnd3sigma(r_TruthList,
                                  v_TruthList,
                                  t_TruthNpArray,
                                  posVelEstListLog,
                                xLabel="Time [s]",
                                title="Landmark Innovations, Body Frame Axis",
                                unitString = "km",):


    ############################################
    # Position and Velocity filter state
    # extraction, interpolation of truth, and error calculation
    ############################################
    t_filt = np.array([s.t for s in posVelEstListLog])
    Pxx_list = [s.Pxx for s in posVelEstListLog]
    P_diag_posVel = np.array([np.diag(P) for P in Pxx_list])
    sigma3_posVel = 3 * np.sqrt(P_diag_posVel)



    # Extract reference state
    r_filt = np.vstack([
        np.array(xref.r_BM_M_mean).reshape(1, -1)
        for xref in posVelEstListLog
    ])
    v_filt = np.vstack([
        np.array(xref.Mdrdt_BM_M_mean).reshape(1, -1)
        for xref in posVelEstListLog
    ])

    # Extract true state
    r_truth = np.array(r_TruthList)
    v_truth = np.array(v_TruthList)

    r_interp_truth = interp1d(t_TruthNpArray, r_truth, axis=0)
    r_true_interp = r_interp_truth(t_filt)

    v_interp_truth = interp1d(t_TruthNpArray, v_truth, axis=0)
    v_true_interp = v_interp_truth(t_filt)

    mask = np.any(P_diag_posVel < 0., axis=1)
    bad_times = t_filt[mask]



    # Compute estimation error
    positionError = r_true_interp - r_filt
    velocityError = v_true_interp - v_filt
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



def plotMekfAttitudeErrorAnd3Sigma(mekfStateList,q_BM_TruthList,t_TruthNpArray):

    ############################################
    # MEKF filter state
    # extraction, interpolation of truth, and error calculation
    ############################################
    t_filt = np.array([s.t for s in mekfStateList])
    Pxx_list = [s.Pxx for s in mekfStateList]
    P_diag_mekf = np.array([np.diag(P) for P in Pxx_list])
    # convert to deg
    P_diag_mekf = np.rad2deg(np.rad2deg(P_diag_mekf))
    sigma3_mekf = 3 * np.sqrt(P_diag_mekf)


    # Extract reference state
    q_BM_filt_list = np.array([mx.q_BMref for mx in mekfStateList])
    gyroBias_filt_array = np.vstack([
        np.array(xref.gyroBiasRef).reshape(1, -1)
        for xref in mekfStateList
    ])
    # interpolate truth solution
    q_BM_truth_array = np.array(q_BM_TruthList)
    q_BM_interp1dObj_truth = interp1d(t_TruthNpArray, q_BM_truth_array, axis=0)
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

    nSolutions= len(q_BM_filt_list)

    ############################################
    # Plot attitude and gyro bias estimation errors
    ############################################
    fig, axs = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    body_labels = ['X', 'Y', 'Z']
    gryo_labels = ['X', 'Y', 'Z']

    # Position error plots
    for i in range(3):
        axs[i, 0].plot(t_filt, np.rad2deg(PRV_BprimeB_array[:, i]), 'k-', linewidth=1.8, label=f'{body_labels[i]}')
        axs[i, 0].plot(t_filt, (sigma3_mekf[:, i]), 'r--', linewidth=1)
        axs[i, 0].plot(t_filt, (-sigma3_mekf[:, i]), 'r--', linewidth=1, label='±3σ confidence')
        axs[i, 0].set_ylabel(f'Body Frame {body_labels[i]} Error [deg]')
        axs[i, 0].grid(True)
        axs[i, 0].legend(loc='upper right')
        axs[0,0].set_title("Body Frame MCMF Attitude Error as PRV")

    # Velocity error plots
    for i in range(3):
        axs[i, 1].plot(t_filt, np.rad2deg(gyroBiasError_array[:,i]), 'k-', linewidth=1.8, label=f'{body_labels[i]}')
        axs[i, 1].plot(t_filt, (sigma3_mekf[:, 3 + i]), 'r--', linewidth=1)
        axs[i, 1].plot(t_filt, (-sigma3_mekf[:, 3 + i]), 'r--', linewidth=1,label='±3σ confidence')
        axs[i, 1].set_ylabel(f'Gyro Frame {body_labels[i]} Bias Error [deg/s]')
        axs[i, 1].grid(True)
        axs[i, 1].legend(loc='upper right')
        axs[0,1].set_title("Gryo Bias Error")

    axs[-1, 0].set_xlabel('Time [s]')
    axs[-1, 1].set_xlabel('Time [s]')
    fig.suptitle(f"MEKF Estimation Errors ±3σ\n{nSolutions} EKF Steps", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
