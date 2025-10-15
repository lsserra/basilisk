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
def McmfPoseDynamics(t,x):

    MU_MOON = 4902.799 * 1e9 # m^3/s^3
    w_MN_M = np.array([0.0, 0.0, 2*np.pi/27.322/24/3600])

    r_BM_M = x[:3]
    Mdrdt_BM_M = x[3:]
    r_BM_M_norm = np.linalg.norm(r_BM_M)

    fgrav = MU_MOON*r_BM_M/r_BM_M_norm**3
    coriolis = 2* np.cross(w_MN_M,Mdrdt_BM_M)
    centripital = np.cross(w_MN_M, np.cross(w_MN_M, r_BM_M))
    dr2dt2_BM_M_M = fgrav - coriolis - centripital
    
    xdot = np.hstack([Mdrdt_BM_M,dr2dt2_BM_M_M])

    return xdot

def CovDynamics(t, P_flat, Fx, Fw, Qww, nx):
    """Compute covariance derivative for EKF."""
    P = P_flat.reshape((nx, nx))
    Pdot = Fx @ P + P @ Fx.T + Fw @ Qww @ Fw.T
    return Pdot.flatten()




############################################
# Classes to hold Error and Reference States
############################################
class EkfErrorState():
    def __init__(self,nx):
        self.mx = np.zeros((nx, 1))
        self.Pxx = np.eye(nx)
        self.t = 0.0
    def update(self,x,P,t):
        self.mx = x
        self.Pxx = P
        self.t = t

class EkfReferenceState():
    def __init__(self):
        self.r_BM_M = np.zeros((3,1))
        self.Mdrdt_BM_M = np.zeros((3,1))
        self.q_BM = Quaternion.identity()
        self.t = 0






############################################
#           Main Estimator class
############################################

class EkfPoseEstimator():

    def __init__(self, nx, nz):

        self.nx = nx
        self.nz = nz

        # error state
        self.mx_prior_tk_ = EkfErrorState(self.nx) # posteriori tk-1
        self.mx_prior_tk = EkfErrorState(self.nx) # priori tk
        self.mx_post_tk = EkfErrorState(self.nx) # posteriori tk

        # reference state
        self.xRef_tk = EkfReferenceState()
        self.xRef_tk_ = EkfReferenceState()

        # Moon angular rate
        self.w_MN_M = np.array([0.0, 0.0, 2*np.pi/27.322/24/3600])
        wx,wy,wz = self.w_MN_M
        self.w_MN_M_skew = np.array([
            [0, -wz, wy],
            [wz, 0, -wx],
            [-wy, wx, 0]
        ])
        # Moon gravitational const
        self.MU_MOON = 4902.799 * 1e9 # m^3/s^3


        self.H = np.hstack( (np.identity(3),np.zeros((3,3))) )



        # process noise and measurement noise matrices
        self.Qww = np.eye(self.nx)
        self.Fw = np.eye(self.nx)
        self.Pvv = np.eye(self.nz)


        # chi squared innovation position meas. gate, alpha=3, DOF=3
        self.chi2gate_posmeas = 7.815
        self.implChi2 = False

        # --- Logging containers ---
        self.errorState_log = []
        self.refState_log = []
        self.innovation_log = []  # stores (t, innovation)
        
        self.outlier_log = []



    def initialize(self,xRef_tk_: "EkfReferenceState" ,mx_prior_tk_: "EkfErrorState"):
        self.xRef_tk_ = xRef_tk_
        self.mx_prior_tk_ = mx_prior_tk_

    def propagateReferenceState(self,toTime):

        tk = toTime
        # grab reference state
        tk_ = self.xRef_tk_.t
        xk_ = np.hstack([
            self.xRef_tk_.r_BM_M,
            self.xRef_tk_.Mdrdt_BM_M
            ])
        xk_.flatten()
        

        # numerically integrate
        sol = solve_ivp(
            t_span=[tk_,tk],
            y0=xk_,
            fun=lambda t, x: McmfPoseDynamics(t,x),
            method='RK45',
            rtol=1e-9,
            atol=1e-9
        )
        xsol = sol.y
        xk = xsol[:,-1]
        
        # write to tk ref solution 
        self.xRef_tk.t = tk
        self.xRef_tk.r_BM_M = xk[:3]
        self.xRef_tk.Mdrdt_BM_M = xk[:3]

    def propagateErrorCov(self,Fxk,toTime):

        tk = toTime

        # grab tk_ covariance
        Pxx_tk_ = self.mx_prior_tk_.Pxx
        tk_ = self.mx_prior_tk_.t
        P0 = Pxx_tk_.flatten()
        # numerically integrate
        sol = solve_ivp(
            t_span=[tk_,tk],
            y0=P0,
            fun=lambda t, P: CovDynamics(t, P, Fxk, self.Fw, self.Qww, self.nx),
            method='RK45',
            rtol=1e-9,
            atol=1e-9
        )
        Pxx_tk = sol.y[:, -1].reshape((self.nx, self.nx))
        
        # set internal error state obj
        self.mx_prior_tk.Pxx = Pxx_tk
        self.mx_prior_tk.mx = np.zeros((self.nx,1))
        self.mx_prior_tk.t = tk


    def computeDynamicsJacobian(self):

        # grab reference state
        r = self.xRef_tk.r_BM_M
        rdot = self.xRef_tk.Mdrdt_BM_M
        rnormSquared = np.linalg.norm(r)**2

        F11 = np.zeros((3,3))
        F12 = np.eye(3)
        F22 = -2*self.w_MN_M_skew

        # gravitational partial wrt position
        rnorm = np.linalg.norm(r)
        I3 = np.eye(3)
        F21_ = self.MU_MOON * (I3 / rnorm**3 - 3 * np.outer(r, r) / rnorm**5)

        # rdot partial wrt r
        F21 = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                F21[i,j] = (
                    self.MU_MOON*( rdot[i]* rnormSquared**(-3/2) +
                        r[i] * (-3*rnormSquared**(-5/2))* rdot[j] * r[j]
                    ))
        F = np.block([[F11,F12],
                     [F21_,F22]]) 
        return F        

    def propagate(self,toTime):
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
        self.errorState_log.append(copy.deepcopy(self.mx_prior_tk_))
        self.refState_log.append(copy.deepcopy(self.xRef_tk_))

        # Propagation reference state
            # inside this function we set xref_tk_ -> xref_tk
        self.propagateReferenceState(toTime=tk)

        # get jacobian by linearizing about reference
            # inside this function we linearize about xref_tk
        Fxk = self.computeDynamicsJacobian()

        # numerically integrate error covariance
            # inside this function we set mx_prior_tk_ -> mx_prior_tk
        self.propagateErrorCov(Fxk=Fxk,toTime=tk)

        # log error state and reference state after propagation
        self.errorState_log.append(copy.deepcopy(self.mx_prior_tk))
        self.refState_log.append(copy.deepcopy(self.xRef_tk))



    def update(self, z_meas, measTime):
        foo =1
        '''
        mxm = self.mxSol_mtk.mx
        Pxxm = self.mxSol_mtk.Pxx

        z = z_meas.reshape(-1, 1)
        
        # Innovation
        inn = z - self.H @ mxm
        
        # cross covariance
        Pxzm = Pxxm @ self.H.T

        # innovation covariance
        Pzzm = (self.H @ Pxxm @ self.H.T) + self.Pvv
        
        # Chi Squared Outlier Detection
        if self.implChi2:
            # perform outlier detection
            # NIS should follow a chi squared distribution,
            # inn ~ N(0,P), so nis ~ chi2
            try:
                Pzzm_inv = np.linalg.inv(Pzzm)
            except np.linalg.LinAlgError:
                # safeguard if Pzzm becomes singular
                Pzzm_inv = np.linalg.pinv(Pzzm)

            nis = float(inn.T @ Pzzm_inv @ inn)
            if not (nis < self.chi2gate_posmeas):
                rejected = True
                # solution is aprior solution
                self.mxSol_ptk_ = self.mxSol_mtk
                # Log data
                self._log_state(mxm, Pxxm, measTime)
                self._log_innovation(inn, measTime)
                self._log_outlier(measTime, rejected)
                return
            
            
        rejected = False   
        # kalman gain 
        Kk = Pxzm @ np.linalg.inv(Pzzm)

        # state update
        mx_upd = mxm + Kk @ inn
        # Update step
        mx_upd = mxm + Kk @ inn
        I = np.eye(self.nx)
        Pxx_upd = (I - Kk @ self.H) @ Pxxm

        # Store posterior state
        self.mxSol_ptk.update(mx_upd, Pxx_upd, measTime)

        # grab posteriori state and copy 
        self.mxSol_ptk_ = self.mxSol_ptk

        # Log data
        self._log_state(mx_upd, Pxx_upd, measTime)
        self._log_innovation(inn, measTime)
        self._log_outlier(measTime, rejected)
        '''


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
