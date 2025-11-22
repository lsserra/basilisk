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

    xdot = np.concatenate((rdot.flatten(), dr2dt2.flatten()), axis=0)

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
    F21 = -MU_MOON * ((I3 / rnorm**3) - ((3 * r @ r.T) / rnorm**5))
    Fx = np.block([[F11, F12],
                   [F21, F22]])


                     # Defensive shape checks (will raise helpful errors if wrong)
    if Fw.ndim != 2:
        raise ValueError("Fw must be 2D; got shape {}".format(Fw.shape))
    if Qww.shape[0] != Qww.shape[1]:
        raise ValueError("Qww must be square")
    Qterm = Fw @ Qww @ Fw.T
    if Qterm.shape != (nx, nx):
        raise ValueError("Process noise term shape mismatch: expected ({},{}) got {}".format(nx, nx, Qterm.shape))


    # --- Covariance dynamics ---
    Pdot = Fx @ P + P @ Fx.T + Fw @ Qww @ Fw.T

    # --- Stack mean and covariance derivatives ---
    x_aug_dot = np.concatenate((xdot.flatten(), Pdot.flatten()),axis=0)

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



def posAttFullStateProp(t, x_aug, MU_MOON, w_MN_M, w_BM_B_corrected, Fw, Qww, nx):
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
    # translational
    r = x[:3]
    rdot = x[3:6]
    rnorm = np.linalg.norm(r)
    # attitude
    alpha = x[6:9]
    erwbias = x[9:]

    # --- Mean dynamics ---
    fgrav = -MU_MOON * r / rnorm**3
    coriolis = 2 * np.cross(w_MN_M, rdot)
    centripetal = np.cross(w_MN_M, np.cross(w_MN_M, r))
    dr2dt2 = fgrav - coriolis - centripetal

    xdot_trans = np.concatenate((rdot.flatten(), dr2dt2.flatten()), axis=0)

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
    F21 = -MU_MOON * ((I3 / rnorm**3) - ((3 * r @ r.T) / rnorm**5))
    FxTrans = np.block([[F11, F12],
                   [F21, F22]])

    # propagate error covarance
    wx,wy,wz = w_BM_B_corrected
    omega_skew = np.array([
        [0, -wz, wy],
        [wz, 0, -wx],
        [-wy, wx, 0]
    ])
    
    # MEKF dynamics Jacobian
    FxMekf = np.hstack((-omega_skew,-np.eye(3)))
    FxMekf = np.vstack((FxMekf,np.zeros((3,6))))

    # Formulate full state Jacobian and derivatives
    Fx = np.zeros((nx,nx))
    Fx[:6,:6] = FxTrans
    Fx[6:,6:] = FxMekf

    xdot = np.zeros((nx,1)).flatten()
    xdot[:6] = xdot_trans



    # Defensive shape checks (will raise helpful errors if wrong)
    if Fw.ndim != 2:
        raise ValueError("Fw must be 2D; got shape {}".format(Fw.shape))
    if Qww.shape[0] != Qww.shape[1]:
        raise ValueError("Qww must be square")
    Qterm = Fw @ Qww @ Fw.T
    if Qterm.shape != (nx, nx):
        raise ValueError("Process noise term shape mismatch: expected ({},{}) got {}".format(nx, nx, Qterm.shape))

    # --- Covariance dynamics ---
    Pdot = Fx @ P + P @ Fx.T + Fw @ Qww @ Fw.T

    # --- Stack mean and covariance derivatives ---
    x_aug_dot = np.concatenate((xdot.flatten(), Pdot.flatten()),axis=0)

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



class LandMarkInnovation():
    def __init__(self):
        self.t=0.0
        self.innovation = np.zeros((3,1))
        self.innovationCov = np.zeros((3,3))
        self.landmarkId = -1



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
        self.MU_MOON = 4902.799 # km^3/kg/s^2




        # --- Logging containers ---
        self.posVelState_log = []
        self.mekfState_log = []
        self.innovation_log = []       
        self.outlier_log = []



    def initialize(self,initPosVelState: "EkfPosVelState" ,initMekfState: "MekfState",
                   QPosVel, Qmekf):
        '''
            Funtion to initialize mean position / velocity state, MEKF state, and PSD
        '''
        # initalize prior
        self.mx_posVel_prior_tk_ = initPosVelState
        self.mx_mekf_prior_tk_ = initMekfState
        
        # initialize full state
        self.mx_full.Pxx = np.zeros((self.nx_full,self.nx_full))
        self.mx_full.Pxx[:6,:6] = initPosVelState.Pxx   
        self.mx_full.Pxx[6:,6:] = initMekfState.Pxx
        self.mx_full.t = initPosVelState.t

        # PSD
        self.QPosVel = QPosVel
        self.QMekf = Qmekf


    def loadLandmarkMap(self,landmarkMapMCMF):
        self.landmarkMap = landmarkMapMCMF


    def propagate(self,toTime,w_BN_B_meas):
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

        # --- Full State Propagation --- #
        tkm = self.mx_full.t
        Pxx_prior_tk_ = self.mx_full.Pxx
        xRef_tk_ = np.concatenate((
            self.mx_posVel_prior_tk_.r_BM_M_mean.flatten(),
            self.mx_posVel_prior_tk_.Mdrdt_BM_M_mean.flatten(),
            np.zeros((self.nx_mekf,1)).flatten()
            ),axis=0)

        x_aug0 = np.concatenate((
        xRef_tk_.flatten(),                
        Pxx_prior_tk_.flatten()  
        ),axis=0)

        # compute corrected gyro output 
        # collect gyro measurement and correct with current bias est
        w_BN_B_corrected = (w_BN_B_meas - self.mx_mekf_prior_tk_.gyroBiasRef.flatten()).flatten()
        # rotate MCMF angular velocity into body frame
        w_MN_B = self.mx_mekf_prior_tk_.q_BMref.rotate(self.w_MN_M)
        w_BM_B_corrected = w_BN_B_corrected - w_MN_B

        # propagate the coupled mean and covariance dynamics
        # construct full state noise mapping and psd matrices
        Qfull = np.zeros((self.nx_full,self.nx_full))
        Fwwfull = np.zeros((self.nx_full,self.nx_full))

        Qfull[:6,:6] = self.QPosVel
        Qfull[6:,6:] = self.QMekf
        Fwwfull[:6,:6] = self.Fw_posVel
        Fwwfull[6:,6:] = self.Fw_mekf

        

        sol = solve_ivp(
        fun=lambda t, x_aug: posAttFullStateProp(
            t,
            x_aug, 
            self.MU_MOON, 
            self.w_MN_M, 
            w_BM_B_corrected,
            Fwwfull, 
            Qfull, 
            self.nx_full
            ),
        t_span=[tkm, tk],
        y0=x_aug0,
        method='RK45',
        rtol=1e-9,
        atol=1e-9
        )

        # reconstruct reference state and error covariance
        x_aug_sol = sol.y 
        x_sol_tk = x_aug_sol[:self.nx_full, -1]
        Pxx_sol_tk = x_aug_sol[self.nx_full:, -1].reshape(self.nx_full,self.nx_full)

        # update pos vel state obj post propagation
        self.unpackFullState(
            xin=x_sol_tk,
            Pin=Pxx_sol_tk,
            t=tk,
            transStateObj=self.mx_posVel_prior_tk,
            mekfStateObj=self.mx_mekf_prior_tk
        )

        # if any non-finite values, thrown an exception
        if np.any(Pxx_sol_tk[np.diag_indices_from(Pxx_sol_tk)] < 0.0):
            raise ValueError(
                f"Invalid covariance: Pxx has negative diagonal entries after translational propagation at time {tk}"
            )
            

        # --- MEKF Quaternion Propagation --- #
        # ensure time is aligned with pos/vel
        tkm_mekf = self.mx_mekf_prior_tk_.t
        if not np.isclose(tkm,tkm_mekf,1e-8):
            raise ValueError("Propagation tk minus for pos/vel and MEKF states do not align.")

        ## --- propagate quaternion --- ##
        
        q_BM_tk_ = self.mx_mekf_prior_tk_.q_BMref.as_array()

        sol = solve_ivp(
            fun=lambda t,x : dqdt_wrapper(t,x,w_BM_B=w_BM_B_corrected),
            t_span=[tkm, tk],
            y0=q_BM_tk_,
            method='RK45',
            rtol=1e-9,
            atol=1e-9
        )

        q_BM_tk = sol.y[:, -1]
        q_BM_tk_obj = Quaternion.from_array(q_BM_tk)
        if np.abs(1. - q_BM_tk_obj.norm()) > 1e-7:
            q_BM_tk_obj.normalize()

        # update mekf prior state obj at end of prop
        self.mx_mekf_prior_tk.t = tk
        self.mx_mekf_prior_tk.q_BMref = copy.deepcopy(q_BM_tk_obj)
        self.mx_mekf_prior_tk.q_BMref.ensureScalarPos()


        # log states after propagation
        self.posVelState_log.append(copy.deepcopy(self.mx_posVel_prior_tk))
        self.mekfState_log.append(copy.deepcopy(self.mx_mekf_prior_tk))



    def updateWithLandmarks(self, z_meas_matrix, PvvBodyFrame, measTime):       

        self.Pvv = PvvBodyFrame
        # get column of id's
        landmarkIds = z_meas_matrix[:,-1].astype(int)
        landmarkMeas = z_meas_matrix[:,:3]

        # get position at measurement time
        mr_BM_M_tk = self.mx_posVel_prior_tk.r_BM_M_mean.flatten()


        # construct measurement matrix
        HxStack = None
        PvvStack = None
        mzkStack = None
        innovationsVec = None

        # create innovation objects list
        innObjList = []
        for i in range(z_meas_matrix.shape[0]):
            innObjList.append(copy.deepcopy(LandMarkInnovation()))



        for i in range(z_meas_matrix.shape[0]):

            # get map landmark position
            self.mx_mekf_prior_tk.q_BMref.ensureScalarPos()
            map_r_LM_M = self.landmarkMap[landmarkIds[i],:].flatten()
            map_r_LM_B= self.mx_mekf_prior_tk.q_BMref.rotate(
                map_r_LM_M
            ) 

            # compute mean of measurement model
            mzk = self.mx_mekf_prior_tk.q_BMref.rotate(map_r_LM_M - mr_BM_M_tk)
            innovation = (landmarkMeas[i,:].flatten() - mzk).reshape((3,1))
            # log innovation
            innObj = innObjList[i]
            innObj.t = measTime
            innObj.innovation = innovation.reshape((3,1))
            innObj.landmarkId = landmarkIds[i]

            # translation Hx with nx_fullstate columns
            TBMhat = self.mx_mekf_prior_tk.q_BMref.to_dcm()
            HxTrans = np.hstack((-TBMhat, np.zeros((self.nz, 3))))

            # MEKF Hx with nx_fullstate columns
            # skew of mean of mcmf to body frame position
            HxMekf = np.zeros((self.nz,self.nx_mekf))
            rx,ry,rz = TBMhat@map_r_LM_M
            map_r_LM_B_skew = np.array([
                [0, -rz, ry],
                [rz, 0, -rx],
                [-ry, rx, 0]
            ])
            rx,ry,rz = TBMhat@mr_BM_M_tk
            mr_BM_B_tk_skew = np.array([
                [0, -rz, ry],
                [rz, 0, -rx],
                [-ry, rx, 0]
            ])
            HxMekf[:,:3] = map_r_LM_B_skew - mr_BM_B_tk_skew

            

            if HxStack is not None:
                Hxi = np.concatenate((HxTrans,HxMekf),axis=1)
                HxStack = np.concatenate((HxStack,Hxi),axis=0)
                PvvStack = block_diag(
                    PvvStack,
                    self.Pvv)
                innovationsVec = np.vstack((innovationsVec, innovation.reshape(-1,1)))
            else:
                HxStack = np.concatenate((HxTrans,HxMekf),axis=1)
                # HxStack = HxTrans 
                # HxStack = HxMekf
                PvvStack = self.Pvv
                innovationsVec = innovation.reshape(-1,1)
                
        # prepare for kalman update
        Pxxk_prior = self.mx_full.Pxx
        # Pxxk_prior = self.mx_mekf_prior_tk.Pxx
        Pxzk = Pxxk_prior @ HxStack.T
        Pzzk = (HxStack @ Pxxk_prior @ (HxStack.T)) + PvvStack
        Kk = Pxzk @ linalg.inv(Pzzk)

        # store innovation covariance
        block_size = self.nz
        for i, innObj in enumerate(innObjList):
            start = i * block_size
            stop = start + block_size
            innObj.innovationCov = Pzzk[start:stop, start:stop]
            self.innovation_log.append(copy.deepcopy(innObj))

        # create full state 
        mxk_prior = np.concatenate((
            self.mx_posVel_prior_tk.r_BM_M_mean.flatten(),
            self.mx_posVel_prior_tk.Mdrdt_BM_M_mean.flatten(),
            self.mx_mekf_prior_tk.angleError_mean.flatten(),
            self.mx_mekf_prior_tk.gyroBiasError_mean.flatten()
        ),axis=0).reshape((-1,1))
        
        # kalman update
        mxk_post = mxk_prior + Kk @ innovationsVec

        # try Joseph's Formulation of the covariance update eq
        I12 = np.eye(self.nx_full)

        # Kalman Gain with non-linear measurements
        Pxxk_post = (I12 - Kk@HxStack)@Pxxk_prior

        # Joseph's
        # Pxxk_post = (I12 - Kk@HxStack) @ Pxxk_prior @ (I12 - Kk@HxStack).T + Kk@PvvStack@Kk.T
        # Basic Covariance update
        # Pxxk_post = Pxxk_prior - Pxzk@Kk.T - Kk@Pxzk.T + Kk @ Pzzk @ Kk.T

        # unpack posterior
        self.unpackFullState(
            xin=mxk_post,
            Pin=Pxxk_post,
            t=measTime,
            transStateObj=self.mx_posVel_post_tk,
            mekfStateObj=self.mx_mekf_post_tk
        )

         # if any negative diagonal entries, stop the update
        if np.any(self.mx_full.Pxx[np.diag_indices_from(self.mx_full.Pxx)] < 0.0):
            raise ValueError(
                f"Invalid covariance: Pxx has negative diagonal entries after measurement update at time {measTime}"
            )


        # add attitude error correction to nominal quaternion
        q_err = Quaternion(qv=self.mx_mekf_post_tk.angleError_mean.flatten()/2,
                           q0=1.0)
        q_BM_post = (q_err*self.mx_mekf_prior_tk.q_BMref)
        q_BM_post.ensureScalarPos()
        if np.abs(q_BM_post.scalar()-1.)>1e-7:
            q_BM_post.normalize()
        
        self.mx_mekf_post_tk.q_BMref = copy.deepcopy(q_BM_post)
        # add gyro bias error correction to nominal bias
        gyroBias_post = (self.mx_mekf_prior_tk.gyroBiasRef.flatten() +
                         self.mx_mekf_post_tk.gyroBiasError_mean.flatten())
        self.mx_mekf_post_tk.gyroBiasRef = gyroBias_post.flatten()

        # set error state to 0
        self.mx_mekf_post_tk.angleError_mean = np.zeros((3,1))
        self.mx_mekf_post_tk.gyroBiasError_mean = np.zeros((3,1))

        
        # log updated states
        self.posVelState_log.append(copy.deepcopy(self.mx_posVel_post_tk))
        self.mekfState_log.append(copy.deepcopy(self.mx_mekf_post_tk))


    def unpackFullState(self, xin, Pin, t, transStateObj, mekfStateObj):
        # unpack states
        transStateObj.r_BM_M_mean = xin[:3].flatten()
        transStateObj.Mdrdt_BM_M_mean = xin[3:6].flatten()
        mekfStateObj.angleError_mean = xin[6:9].flatten()
        mekfStateObj.gyroBiasError_mean = xin[9:].flatten()

        # unpack covariance
        transStateObj.Pxx = Pin[:6,:6]
        mekfStateObj.Pxx = Pin[6:,6:]

        # update times
        transStateObj.t = t
        mekfStateObj.t = t
        
        # construct full state error covariance post propagation
        self.mx_full.Pxx = Pin
        self.mx_full.mx = xin
        self.mx_full.t = t







        
            



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
