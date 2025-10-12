import numpy as np
from numpy import linalg
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag


class FreeFallSolution():
    def __init__(self,nx):
        self.mx = np.zeros((nx, 1))
        self.Pxx = np.eye(nx)
        self.t = 0.0
    def update(self,x,P,t):
        self.mx = x
        self.Pxx = P
        self.t = t


class FreeFallKF():

    def __init__(self, sigma_pos=0.1, sigma_vel=0.01, sigma_meas=0.1):

        self.nx = 6
        self.nz = 3

        self.g = 9.81 # m/s^2

        self.mxSol_ptk_ = FreeFallSolution(self.nx) # posteriori tk-1
        self.mxSol_mtk = FreeFallSolution(self.nx) # priori tk
        self.mxSol_ptk = FreeFallSolution(self.nx) # posteriori tk


        self.H = np.hstack( (np.identity(3),np.zeros((3,3))) )



        # process noise and measurement noise matrices
        self.Qww = block_diag((sigma_pos**2) * np.eye(3), (sigma_vel**2) * np.eye(3))
        self.Pvv = (sigma_meas**2) * np.eye(3)


        # chi squared innovation position meas. gate, alpha=3, DOF=3
        self.chi2gate_posmeas = 7.815
        self.implChi2 = True


        # self.F = np.array([
        #     [np.zeros(3,3),np.eye(3)],
        #     [np.zeros(3,3),np.zeros(3,3)]
        # ])

        # --- Logging containers ---
        self.state_log = []       # stores (t, state)
        self.innovation_log = []  # stores (t, innovation)
        self.cov_log = []
        self.outlier_log = []



    def initialize(self,mx,Pxx,t):
        self.mxSol_ptk_.mx = mx
        self.mxSol_ptk_.Pxx = Pxx
        self.mxSol_ptk_.t = t
        self._log_state(x=mx,P=Pxx,t=t)



    def propagate(self,toTime):
        
        # process to propagate
        currentTime = self.mxSol_ptk_.t
        mx = self.mxSol_ptk_.mx
        Pxx = self.mxSol_ptk_.Pxx
        dt = toTime - currentTime

        STMk = np.block([
            [np.eye(3), dt * np.eye(3)],
            [np.zeros((3,3)), np.eye(3)]
            ])
        Uk = np.hstack((
            [0, 0, -self.g*0.5*(dt)**2],
            [0, 0, -self.g*dt]
            )).reshape(-1,1)
        
        # integrate process noise
        Pww = self.Qww*dt
        
        # Propagation step
        mx_pred = STMk @ mx + Uk
        Pxx_pred = STMk @ Pxx @ STMk.T + Pww

        # Store predicted state
        self.mxSol_mtk.update(mx_pred, Pxx_pred, toTime)
        self._log_state(mx_pred, Pxx_pred, toTime)


    def update(self, z_meas, measTime):
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

    def get_state(self):
        return self.mxSol_ptk
    

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
