import sys, os

import numpy as np
from numpy import linalg
import copy
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag

# Add the basilisk root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import numpy as np
from helpers.attitude.Quaternion import Quaternion






############################################
#       Dynamics functions to be
#           called by solve_ivp
############################################

def EkfPosVelProp(t, x_aug, MU_MOON, w_MN_M, Fw, Qww, nx):
    """
    Coupled propagation of mean and covariance for EKF in MCMF frame.
    x_aug = [x, P_flat] 
    where x = [r, rdot]

    Fw is the determinstic process noise mapping matrix and is not a function of the mean state
    Qww is the process noise PSD
    """

    # --- Unpack state ---
    x = x_aug[:nx]
    P_flat = x_aug[nx:]
    P = P_flat.reshape((nx, nx))

    # --- Unpack mean state ---
    r = x[:3]
    rdot = x[3:]
    rnorm = np.linalg.norm(r)

    # --- Mean dynamics ---
    fgrav = -MU_MOON * r / rnorm**3
    coriolis = 2 * np.cross(w_MN_M, rdot)
    centripetal = np.cross(w_MN_M, np.cross(w_MN_M, r))
    dr2dt2 = fgrav - coriolis - centripetal

    xdot = np.hstack([rdot, dr2dt2])

    # --- Compute dynamics Jacobian Fx ---
    I3 = np.eye(3)
    w_skew = np.array([
        [0, -w_MN_M[2], w_MN_M[1]],
        [w_MN_M[2], 0, -w_MN_M[0]],
        [-w_MN_M[1], w_MN_M[0], 0]
    ])
    F11 = np.zeros((3, 3))
    F12 = np.eye(3)
    F22 = -2 * w_skew
    F21 = -MU_MOON * (I3 / rnorm**3 - 3 * np.outer(r, r) / rnorm**5)
    Fx = np.block([[F11, F12],
                   [F21, F22]])

    # --- Covariance dynamics ---
    Pdot = Fx @ P + P @ Fx.T + Fw @ Qww @ Fw.T

    # --- Stack mean and covariance derivatives ---
    x_aug_dot = np.hstack([xdot, Pdot.flatten()])

    return x_aug_dot


def dqdt_wrapper(t, q, w_BM_B):
    return Quaternion.dqdt(t, w_BM_B, q)

def MekfCovProp(t,x_aug,nx,Fx,Fw,Qww):

    P = x_aug.reshape((nx, nx))

    # --- Covariance dynamics ---
    Pdot = Fx @ P + P @ Fx.T + Fw @ Qww @ Fw.T

    # --- Stack mean and covariance derivatives ---
    x_aug_dot = Pdot.flatten()
    
    return x_aug_dot

    


############################################
# Classes to hold Different Filter States
############################################
class FullFilterState():
    def __init__(self,nx):
        self.mx = np.zeros((nx, 1))
        self.Pxx = np.eye(nx)
        self.t = 0.0
    
class EkfPosVelState():
    def __init__(self,nx):
        self.r_BM_M_mean = np.zeros((3,1))
        self.Mdrdt_BM_M_mean = np.zeros((3,1))
        self.Pxx = np.eye(nx)

        self.t = 0.

class MekfState():
    def __init__(self,nx):
        # reference states
        self.q_BMref = Quaternion.identity()
        self.gyroBiasRef = np.zeros((3,1))
        
        # error states
        self.angleError_mean = np.zeros((3,1))
        self.gyroBiasError_mean = np.zeros((3,1))

        # error cov
        self.Pxx = np.eye(nx)

        self.t = 0.



############################################
#           Main Estimator class
############################################

class EkfPoseEstimator():
    def __init__(self):
        
        # define state sizes
        self.nx_full = 12
        self.nz = 3

        self.nx_posVel = 6
        self.nx_mekf = 6

        # full filter state
        self.mx_full = FullFilterState(self.nx_full) 
    
        # position and velocity state
        self.mx_posVel_prior_tk_ = EkfPosVelState(self.nx_posVel)
        self.mx_posVel_prior_tk = EkfPosVelState(self.nx_posVel)
        self.mx_posVel_post_tk = EkfPosVelState(self.nx_posVel)

        # MEKF state
        self.mx_mekf_prior_tk_ = MekfState(self.nx_mekf)
        self.mx_mekf_prior_tk = MekfState(self.nx_mekf)
        self.mx_mekf_post_tk = MekfState(self.nx_mekf)

        # Process Noise Shaping 
        self.Fw_posVel = np.eye(self.nx_posVel)
        self.Fw_mekf = block_diag(-np.eye(self.nx_mekf//2),np.eye(self.nx_mekf//2))

        # PSD
        self.QPosVel = np.zeros((self.nx_posVel,self.nx_posVel))
        self.QMekf = np.zeros((self.nx_mekf,self.nx_mekf))

        # --- Measurement Update Related --- #
        self.Hx = np.hstack( (-np.identity(3),np.zeros((3,3))) )
        self.Hv = np.eye(self.nz)
        self.Pvv = np.eye(self.nz)
        self.landmarkMap = None



        


        # Moon angular rate
        self.w_MN_M = np.array([0.0, 0.0, 2*np.pi/27.322/24/3600])
        wx,wy,wz = self.w_MN_M
        self.w_MN_M_skew = np.array([
            [0, -wz, wy],
            [wz, 0, -wx],
            [-wy, wx, 0]
        ])
        # Moon gravitational const
        self.MU_MOON = 4902.799 # km^3/s^3
        
       


        # --- Logging containers ---
        self.posVelState_log = []
        self.mekfState_log = []
        self.innovation_log = []  # stores (t, innovation)        
        self.outlier_log = []



    def initialize(self,initPosVelState: "EkfPosVelState" ,initMekfState: "MekfState",
                   QPosVel, Qmekf):
        '''
            Funtion to initialize mean position / velocity state, MEKF state, and PSD
        '''
        # initalize prior
        self.mx_posVel_prior_tk_ = initPosVelState
        self.mx_mekf_prior_tk_ = initMekfState

        # PSD
        self.QPosVel = QPosVel
        self.QMekf = Qmekf


    def loadLandmarkMap(self,landmarkMapMCMF):
        self.landmarkMap = landmarkMapMCMF


    def propagate(self,toTime,w_BM_B_meas):
        ''' 
            tk = toTime

            This function expects: 
                self.mx_prior_tk_ and self.xref_tk_
            to be set outside of this function           
            
            This function will set:
                self.xref_tk and self.mx_prior_tk 
        '''
        
        # set time 
        tk = toTime
        
        # log error state and reference state before propagation
        self.posVelState_log.append(copy.deepcopy(self.mx_posVel_prior_tk_))
        self.mekfState_log.append(copy.deepcopy(self.mx_mekf_prior_tk_))

        # grab prior error covariance, reference state, and time

        # --- Pos Vel Propagation --- #
        tkm = self.mx_posVel_prior_tk_.t
        Pxx_prior_tk_ = self.mx_posVel_prior_tk_.Pxx
        xRef_tk_ = np.hstack([
            self.mx_posVel_prior_tk_.r_BM_M_mean,
            self.mx_posVel_prior_tk_.Mdrdt_BM_M_mean
            ]).flatten()

        x_aug0 = np.hstack([
        xRef_tk_,                
        Pxx_prior_tk_.flatten()  
        ])

        # propagate the coupled mean and covariance dynamics
        sol = solve_ivp(
        fun=lambda t, x_aug: EkfPosVelProp(
            t,
            x_aug, 
            self.MU_MOON, 
            self.w_MN_M, 
            self.Fw_posVel, 
            self.QPosVel, 
            self.nx_posVel
            ),
        t_span=[tkm, tk],
        y0=x_aug0,
        method='RK45',
        rtol=1e-9,
        atol=1e-9
        )

        # reconstruct reference state and error covariance
        x_aug_sol = sol.y 
        x_sol_tk = x_aug_sol[:self.nx_posVel, -1]
        Pxx_sol_tk = x_aug_sol[self.nx_posVel:, -1].reshape(self.nx_posVel,self.nx_posVel)

        # update pos vel state obj post propagation
        self.mx_posVel_prior_tk.r_BM_M_mean = x_sol_tk[:3]
        self.mx_posVel_prior_tk.Mdrdt_BM_M_mean = x_sol_tk[3:]
        self.mx_posVel_prior_tk.t = tk
        self.mx_posVel_prior_tk.Pxx = Pxx_sol_tk


        # --- MEKF Propagation --- #
        # ensure time is aligned with pos/vel
        tkm_mekf = self.mx_mekf_prior_tk_.t
        if not np.isclose(tkm,tkm_mekf,1e-8):
            raise ValueError("Propagation tk minus for pos/vel and MEKF states do not align.")
        # error states are zero, error has been added to nominal

        # gyro mean dynamics = 0

        # propagate quaternion
        w_BM_B_corrected = (w_BM_B_meas - self.mx_mekf_prior_tk_.gyroBiasRef).flatten()
        q_BM_tk_ = self.mx_mekf_prior_tk_.q_BMref.as_array()

        sol = solve_ivp(
            fun=lambda t,x : dqdt_wrapper(t,x,w_BM_B=w_BM_B_corrected),
            t_span=[tkm, tk],
            y0=q_BM_tk_,
            method='RK45',
            rtol=1e-9,
            atol=1e-9
        )

        q_MN_tk = sol.y[:, -1]
        q_MN_tk_obj = Quaternion.from_array(q_MN_tk).normalize()
        
        # propagate error covarance
        wx,wy,wz = w_BM_B_corrected
        omega_skew = np.array([
            [0, -wz, wy],
            [wz, 0, -wx],
            [-wy, wx, 0]
        ])
        
        FxMekf = np.hstack((omega_skew,-np.eye(3)))
        FxMekf = np.vstack((FxMekf,np.zeros((3,6))))
        
        PxxFlat = self.mx_mekf_prior_tk_.Pxx.flatten()
        
        sol = solve_ivp(
        fun=lambda t,x : MekfCovProp(t,x,
                        nx=self.nx_mekf,
                        Fx=FxMekf,
                        Fw=self.Fw_mekf,
                        Qww=self.QMekf),
        t_span=[tkm, tk],
        y0=PxxFlat,
        method='RK45',
        rtol=1e-9,
        atol=1e-9
        )
        # reshape cov
        x_aug_sol = sol.y 
        PxxMekf_tk = x_aug_sol[:, -1].reshape(self.nx_mekf,self.nx_mekf)

        # update mekf prior state obj at end of prop
        self.mx_mekf_prior_tk.t = tk
        self.mx_mekf_prior_tk.q_BMref = q_MN_tk_obj
        self.mx_mekf_prior_tk.Pxx = PxxMekf_tk













        # log states after propagation
        self.posVelState_log.append(copy.deepcopy(self.mx_posVel_prior_tk))
        self.mekfState_log.append(copy.deepcopy(self.mx_mekf_prior_tk))



    def updateWithLandmarks(self, z_meas_array, measTime):
        foo =1
        
        # get propagated error and reference state
        dmx_prior_tk = self.mx_prior_tk.mx
        Pxx_prior_tk = self.mx_prior_tk.Pxx
        r_BM_M_tk = self.xRef_tk.r_BM_M_mean
        Mdrdt_BM_M = self.xRef_tk.Mdrdt_BM_M_mean

        
        # shape measurement array
        z = z_meas_array.reshape(-1, 3)
        
        for i in range(len(z)):

            # Innovation
            lmId = z[i,-1]
            r_LM_M = self.landmarkMap[lmId,:]
            mzk = r_LM_M - r_BM_M_tk
            zk = z[i,:]
            inn =  - mzk
        



    # ---------------------------------------------------------
    # Internal Logging Helpers
    def _log_state(self, x, P, t):
        self.state_log.append((t, x.copy()))
        self.cov_log.append((t, P.copy()))

    def _log_innovation(self, inn, t):
        self.innovation_log.append((t, inn.copy()))

    def _log_outlier(self, t, rejected):
        """Log whether the measurement was rejected at this time."""
        self.outlier_log.append((t, rejected))


   # ---------------------------------------------------------
    # Convenience accessors
    def get_state_history(self):
        times = np.array([t for t, _ in self.state_log])
        states = np.hstack([x for _, x in self.state_log]).T  # shape (N, nx)
        covs = [P for (_, P) in self.cov_log]
        return times, states, covs  # list of Pxx for each time

    def get_innovation_history(self):
        times = np.array([t for t, _ in self.innovation_log])
        innovations = np.hstack([inn for _, inn in self.innovation_log])
        return times, innovations  # shape (nz, N)
    def get_outlier_history(self):
        times = np.array([t for t, _ in self.outlier_log])
        rejected = np.array([flag for _, flag in self.outlier_log])
        return times, rejected
