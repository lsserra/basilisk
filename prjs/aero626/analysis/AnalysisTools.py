import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class PoseAnalyzer():
    ## define pkl accessing keys
    PKL_TRUTH_POS_KEY          = "truth_r"
    PKL_TRUTH_VEL_KEY          = "truth_v"
    PKL_TRUTH_ATT_KEY          = "truth_q"
    PKL_TRUTH_TIME_KEY          = "truth_t"

    PKL_EST_POS_KEY          = "est_r"
    PKL_EST_VEL_KEY          = "est_v"
    PKL_EST_ATT_KEY          = "est_q"
    PKL_EST_PXX_KEY            = "est_Pxx"
    PKL_EST_TIME_KEY          = "est_t"
    
    PKL_REF_FRAME_KEY          = "ref_frame"
    PKL_TGT_FRAME_KEY          = "tgt_frame"
    PKL_RESOLVED_FRAME_KEY     = "resolved_frame"

    def __init__(self):
        self._truth_r = None
        self._truth_v = None
        self._truth_q = None
        self._truth_t = None

        self._est_r = None
        self._est_v = None
        self._est_q = None
        self._est_Pxx = None
        self._est_t = None

        self._units_r = "km"
        self._units_v = "km/s"
        self._units_t = "s"

        self._ref_frame = ""
        self._tgt_frame = ""
        self._resolved_frame = ""

        self._dataDir = ""

        self._pklName = "FILTER_SOL_AND_SIM_TRUTH.pkl"

        

    
    def LoadFromPklFile(self,pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # ---- Truth ----
        self._truth_r = data.get(PoseAnalyzer.PKL_TRUTH_POS_KEY, None)
        self._truth_v = data.get(PoseAnalyzer.PKL_TRUTH_VEL_KEY, None)
        self._truth_q = data.get(PoseAnalyzer.PKL_TRUTH_ATT_KEY, None)
        self._truth_t = data.get(PoseAnalyzer.PKL_TRUTH_TIME_KEY, None)

        # ---- Estimation ----
        self._est_r = data.get(PoseAnalyzer.PKL_EST_POS_KEY, None)
        self._est_v = data.get(PoseAnalyzer.PKL_EST_VEL_KEY, None)
        self._est_q = data.get(PoseAnalyzer.PKL_EST_ATT_KEY, None)
        self._est_Pxx = data.get(PoseAnalyzer.PKL_EST_PXX_KEY, None)
        self._est_t = data.get(PoseAnalyzer.PKL_EST_TIME_KEY, None)

        # ---- Frames ----
        self._ref_frame      = data.get(PoseAnalyzer.PKL_REF_FRAME_KEY, "")
        self._tgt_frame      = data.get(PoseAnalyzer.PKL_TGT_FRAME_KEY, "")
        self._resolved_frame = data.get(PoseAnalyzer.PKL_RESOLVED_FRAME_KEY, "")
    
    def plotTransError3sigma(self):

        ############################################
        # Position and Velocity filter state
        # extraction, interpolation of truth, and error calculation
        ############################################
        t_filt = self._est_t
        P_diag_posVel = np.array([np.diag(P) for P in self._est_Pxx])
        sigma3_posVel = 3 * np.sqrt(P_diag_posVel)

        # Extract reference state
        r_filt = self._est_r
        v_filt = self._est_v

        # Extract true state
        r_truth = self._truth_r
        v_truth = self._truth_v

        r_interp_truth = interp1d(self._truth_t, r_truth, axis=0)
        r_true_interp = r_interp_truth(t_filt)

        v_interp_truth = interp1d(self._truth_t, v_truth, axis=0)
        v_true_interp = v_interp_truth(t_filt)


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
        positionTitle = (f"Position of {self._tgt_frame} w.r.t. {self._ref_frame} resolved in {self._resolved_frame} Estimation Error")
        for i in range(3):
            axs[i, 0].plot(t_filt, positionError[:, i], 'k-', linewidth=1.8, label=f'{pos_labels[i]}')
            axs[i, 0].plot(t_filt, sigma3_posVel[:, i], 'r--', linewidth=1)
            axs[i, 0].plot(t_filt, -sigma3_posVel[:, i], 'r--', linewidth=1, label='±3σ confidence')
            axs[i, 0].set_ylabel(f'{pos_labels[i]} [{self._units_r}]')
            axs[i, 0].grid(True)
            axs[i, 0].legend(loc='upper right')
            axs[0,0].set_title(positionTitle)

        # Velocity error plots
        velocityTitle = (f"Velocity of {self._tgt_frame} w.r.t. {self._ref_frame} resolved in {self._resolved_frame} Estimation Error")
        for i in range(3):
            axs[i, 1].plot(t_filt, velocityError[:,i], 'k-', linewidth=1.8, label=f'{vel_labels[i]}')
            axs[i, 1].plot(t_filt, sigma3_posVel[:, 3 + i], 'r--', linewidth=1)
            axs[i, 1].plot(t_filt, -sigma3_posVel[:, 3 + i], 'r--', linewidth=1,label='±3σ confidence')
            axs[i, 1].set_ylabel(f'{vel_labels[i]} [{self._units_v}]')
            axs[i, 1].grid(True)
            axs[i, 1].legend(loc='upper right')
            axs[0,1].set_title(velocityTitle)

        axs[-1, 0].set_xlabel(f'Time [{self._units_t}]')
        axs[-1, 1].set_xlabel(f'Time [{self._units_t}]')
        fig.suptitle(f"MCMF r_BM_M & Md(.)dt Estimation Errors ±3σ\n{nSolutions} EKF Steps", fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return

    
    @staticmethod
    def ComputeResolvedToLVLH_DCM():
        T_LR = None
        return T_LR
    
    @staticmethod
    def CreatePklFileDataDict(
                    input_truth_r = None,
                    input_truth_v = None,
                    input_truth_q = None,
                    input_truth_t = None,

                    input_est_r = None,
                    input_est_v = None,
                    input_est_q = None,
                    input_est_Pxx = None,
                    input_est_t = None,

                    input_ref_frame = "",
                    input_tgt_frame = "",
                    input_resolved_frame = ""
                    ):
        "input all matrices as (N solutions x __)"
        data_dict = {
            # Truth
            PoseAnalyzer.PKL_TRUTH_POS_KEY:          input_truth_r,
            PoseAnalyzer.PKL_TRUTH_VEL_KEY:          input_truth_v,
            PoseAnalyzer.PKL_TRUTH_ATT_KEY:          input_truth_q,
            PoseAnalyzer.PKL_TRUTH_TIME_KEY:           input_truth_t,

            # Estimation
            PoseAnalyzer.PKL_EST_POS_KEY:            input_est_r,
            PoseAnalyzer.PKL_EST_VEL_KEY:            input_est_v,
            PoseAnalyzer.PKL_EST_ATT_KEY:            input_est_q,
            PoseAnalyzer.PKL_EST_PXX_KEY:           input_est_Pxx,
            PoseAnalyzer.PKL_EST_TIME_KEY:           input_est_t,

            # Frames
            PoseAnalyzer.PKL_REF_FRAME_KEY:          input_ref_frame,
            PoseAnalyzer.PKL_TGT_FRAME_KEY:          input_tgt_frame,
            PoseAnalyzer.PKL_RESOLVED_FRAME_KEY:     input_resolved_frame,

        }
        return data_dict
    
    
    
    

