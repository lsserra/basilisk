import pickle, os, sys
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
    
    def plotTransError3sigma():
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
    
    
    
    

