import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
# attitude helpers
from helpers.attitude import DCM
from helpers.attitude.Quaternion import Quaternion

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
        self._units_att = "deg"

        self._ref_frame = ""
        self._tgt_frame = ""
        self._resolved_frame = ""

        self._dataDir = ""

        self._solutionName = ""

        ## optionally remove titles for clean figures
        self._NO_TITLE_FLAG = False

        self._EXPORT_FIGURES_FLAG = False

        

    
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
        legend_label = 'Est. Error'

        # titles
        figureTitleOpt = f"{self._solutionName} {self._resolved_frame} Frame Translational Estimation Error"
        if self._NO_TITLE_FLAG:
            positionTitle = ''
            velocityTitle = ''
            figureTitle = ''
        else:
            positionTitle = (f"Position of {self._tgt_frame} w.r.t. {self._ref_frame} resolved in {self._resolved_frame} Estimation Error")
            velocityTitle = (f"Velocity of {self._tgt_frame} w.r.t. {self._ref_frame} resolved in {self._resolved_frame} Estimation Error")
            figureTitle = figureTitleOpt 
        
        # Position error plots
        for i in range(3):
            
            axs[i, 0].plot(t_filt, positionError[:, i], 'k-', linewidth=1.8, label=f'{legend_label}')
            axs[i, 0].plot(t_filt, sigma3_posVel[:, i], 'r--', linewidth=1)
            axs[i, 0].plot(t_filt, -sigma3_posVel[:, i], 'r--', linewidth=1, label='±3σ confidence')
            axs[i, 0].set_ylabel(f'{pos_labels[i]} [{self._units_r}]')
            axs[i, 0].grid(True)
            
        axs[0, 0].legend(loc='upper right')
        axs[0,0].set_title(positionTitle)

        # Velocity error plots
        for i in range(3):
            axs[i, 1].plot(t_filt, velocityError[:,i], 'k-', linewidth=1.8, label=f'{legend_label}')
            axs[i, 1].plot(t_filt, sigma3_posVel[:, 3 + i], 'r--', linewidth=1)
            axs[i, 1].plot(t_filt, -sigma3_posVel[:, 3 + i], 'r--', linewidth=1,label='±3σ confidence')
            axs[i, 1].set_ylabel(f'{vel_labels[i]} [{self._units_v}]')
            axs[i, 1].grid(True)
            
        axs[0, 1].legend(loc='upper right')
        axs[0,1].set_title(velocityTitle)

        axs[-1, 0].set_xlabel(f'Time [{self._units_t}]')
        axs[-1, 1].set_xlabel(f'Time [{self._units_t}]')
        fig.suptitle(figureTitle, fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # opt save
        if self._EXPORT_FIGURES_FLAG:
            fig_path = os.path.join(self._dataDir,'figures')
            out_path = os.path.join(fig_path,f"{figureTitleOpt.replace(' ', '_')}.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {out_path}")
        return
    
    def plotAttError3Sigma(self):

        ############################################
        # MEKF filter state
        # extraction, interpolation of truth, and error calculation
        ############################################
        t_filt = self._est_t
        P_diag_att = np.array([np.diag(P[6:,6:]) for P in self._est_Pxx])
        # convert to deg
        P_diag_att = np.rad2deg(np.rad2deg(P_diag_att))
        sigma3_mekf = 3 * np.sqrt(P_diag_att)


        # interpolate truth solution
        q_BM_truth_array = self._truth_q
        q_BM_interp1dObj_truth = interp1d(self._truth_t, q_BM_truth_array, axis=0)
        q_BM_true_interp = q_BM_interp1dObj_truth(t_filt)
        
        # compute attitude error as principle rotation vector
        # body attitude error list
        PRV_BprimeB_list = []
        for i in range(self._est_q.shape[0]):
            # compute attitude error and store
            q_BM_true = Quaternion.from_array(q_BM_true_interp[i,:]).normalize()
            q_BM_filt = Quaternion.from_array(self._est_q[i,:])
            prv_BprimeB = Quaternion.computeEulerVecAttErrorFromQuats(
                q_ref=q_BM_true,
                q_est=q_BM_filt
            )
            PRV_BprimeB_list.append(prv_BprimeB)

        PRV_BprimeB_array = np.array(PRV_BprimeB_list)


        nSolutions= len(self._est_q)

        ############################################
        # Plot attitude estimation errors
        ############################################
        fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        body_labels = ['X', 'Y', 'Z']
        legend_label = 'Est. Error'

        # titles
        figureTitleOpt = f"{self._solutionName} Frame {self._tgt_frame} w.r.t. {self._ref_frame} Estimation Error"
        if self._NO_TITLE_FLAG:
            attTitle = ''
            figureTitle = ''
        else:
            attTitle = f"{self._tgt_frame} frame attitude error as principle rotation vector"
            figureTitle = figureTitleOpt 
            
            
        
        # Position error plots
        for i in range(3):
            axs[i].plot(t_filt, np.rad2deg(PRV_BprimeB_array[:, i]), 'k-', linewidth=1.8, label=f'{legend_label}')
            axs[i].plot(t_filt, (sigma3_mekf[:, i]), 'r--', linewidth=1)
            axs[i].plot(t_filt, (-sigma3_mekf[:, i]), 'r--', linewidth=1, label='±3σ confidence')
            axs[i].set_ylabel(f'{body_labels[i]} [{self._units_att}]')
            axs[i].grid(True)
        axs[0].legend(loc='upper right')
        axs[0].set_title(attTitle)

       

        axs[-1].set_xlabel(f'Time [{self._units_t}]')
        fig.suptitle(figureTitle, fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

    # opt save
        if self._EXPORT_FIGURES_FLAG:
            fig_path = os.path.join(self._dataDir,'figures')
            out_path = os.path.join(fig_path,f"{figureTitleOpt.replace(' ', '_')}.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {out_path}")


    def plot3DimVecError3Sigma(self,truth_v, truth_t, est_v, est_Pxx, est_t,
                            strFigureTitle='',
                            strXlabel='',
                            strYunits=''
                               ):

        ############################################
        # MEKF filter state
        # extraction, interpolation of truth, and error calculation
        ############################################
        
        P_diag = np.array([np.diag(P) for P in est_Pxx])
        # convert to deg
        P_diag = (P_diag)
        sigma3 = 3 * np.sqrt(P_diag)


        # interpolate truth solution
        truth_inter_obj = interp1d(truth_t, truth_v, axis=0)
        truth_interp_to_sol = truth_inter_obj(est_t)
        
        error = truth_interp_to_sol - est_v

        ############################################
        # Plot attitude estimation errors
        ############################################
        fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axis_labels = ['X', 'Y', 'Z']
        legend_label = 'Est. Error'

        # titles
        if self._NO_TITLE_FLAG:
            figureTitle = ''
        else:
            figureTitle = strFigureTitle
            
            
        
        # Position error plots
        for i in range(3):
            axs[i].plot(est_t, error[:,i], 'k-', linewidth=1.8, label=f'{legend_label}')
            axs[i].plot(est_t, (sigma3[:, i]), 'r--', linewidth=1)
            axs[i].plot(est_t, (-sigma3[:, i]), 'r--', linewidth=1, label='±3σ confidence')
            axs[i].set_ylabel(f'{axis_labels[i]} [{strYunits}]')
            axs[i].grid(True)
        axs[0].legend(loc='upper right')

        axs[-1].set_xlabel(strXlabel)
        fig.suptitle(figureTitle, fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # opt save
        if self._EXPORT_FIGURES_FLAG:
            fig_path = os.path.join(self._dataDir,'figures')
            out_path = os.path.join(fig_path,f"{self._solutionName } {strFigureTitle.replace(' ', '_')}.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {out_path}")


    


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
    
    
    
    

