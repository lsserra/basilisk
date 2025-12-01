import os, sys, copy
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from prjs.aero626.ekfMoon.EkfPoseEstimator import(
    EkfPoseEstimator,
    FullFilterState
) 
from helpers.attitude.Quaternion import Quaternion



class MoonEGMF():
    def __init__(self):
        
        ## list of gaussian pdfs
        self.gaussianPdfList_ = []

        ## pull in ekf for pdf propgation and update utility
        self._ekf = None
        self.Pwwkm1 = None

        

        
        #### STORAGE ####
        ## list to store gmLists at each epoch
        self.storeGmList_ = []

        ## list to store GM mean, cov at each up
        self.storeGmBestGuess_ = []

        ## list to store residual and residual


    def PropagateMixtureEkf(self,toTime,w_BN_B_meas):
        for k,g in enumerate(self.gaussianPdfList_):
            self._ekf.mx_full = g
            
            self._ekf.propagate(toTime=toTime,w_BN_B_meas=w_BN_B_meas)
            ## unique add discrete process noise
            g.Pxx = g.Pxx + self.Pwwkm1
            
            

        # compute some GM stats
        mean,cov = self.computeBestEstMeanAndCovAtEpoch()
        gyroBiasEst,q_BM_est= self.computeBestEstQuaternionAndGryoBias()
        GMstate = FullFilterState(nx=len(self._ekf.mx_full.mx))
        GMstate.mx = mean
        GMstate.Pxx = cov
        GMstate.t = toTime
        GMstate.gyroBiasRef = gyroBiasEst
        GMstate.q_BMref = q_BM_est
        self.storeGmBestGuess_.append(copy.deepcopy(GMstate))
   

    def LandmarkMeasUpdateEkf(self, z_meas_matrix, PvvBodyFrame, measTime):
        for k,g in enumerate(self.gaussianPdfList_):
            self._ekf.mx_full = g
            self._ekf.updateWithLandmarks(z_meas_matrix, PvvBodyFrame, measTime)

        # update weights 
        self.UpdateWeights()

        # compute some GM stats
        mean,cov = self.computeBestEstMeanAndCovAtEpoch()
        gyroBiasEst,q_BM_est= self.computeBestEstQuaternionAndGryoBias()
        GMstate = FullFilterState(nx=len(self._ekf.mx_full.mx))
        GMstate.mx = mean
        GMstate.Pxx = cov
        GMstate.t = measTime
        GMstate.gyroBiasRef = gyroBiasEst
        GMstate.q_BMref = q_BM_est
        self.storeGmBestGuess_.append(copy.deepcopy(GMstate))
    
   
    def UpdateWeights(self):
        # # compute normalization factor
        # denom = 0.
        # for i, gm in enumerate(self.gaussianPdfList_):
        #     denom += gm.k*gm.w

        # # normalize posterior weights
        # for i, gm in enumerate(self.gaussianPdfList_):
        #     gm.w = gm.k*gm.w/denom
        # compute unnormalized weights safely
        unnorm = np.array([gm.k * gm.w for gm in self.gaussianPdfList_], dtype=np.float64)

        # avoid underflow: floor tiny weights
        eps = 1e-300
        unnorm = np.clip(unnorm, eps, None)

        denom = np.sum(unnorm)

        # protect denominator from underflow
        if denom < eps or np.isnan(denom):
            # fallback: assign equal weights
            n = len(self.gaussianPdfList_)
            for gm in self.gaussianPdfList_:
                gm.w = 1.0 / n
            return

        # normalize
        for gm, u in zip(self.gaussianPdfList_, unnorm):
            gm.w = u / denom
    

    def computeBestEstMeanAndCovAtEpoch(self): 
        mean = np.zeros_like(self.gaussianPdfList_[0].mx)
        for i, gm in enumerate(self.gaussianPdfList_):
            mean += gm.w*gm.mx

        cov = np.zeros_like(self.gaussianPdfList_[0].Pxx)
        for i, gm in enumerate(self.gaussianPdfList_):
            cov += gm.w*(gm.Pxx + (gm.mx - mean)*(gm.mx - mean).T)
            
        return mean, cov
    
    def computeBestEstQuaternionAndGryoBias(self):
        # weighted average for gryo bias estimate
        gyroBiasEst = np.zeros_like(self.gaussianPdfList_[0].gyroBiasRef)
        for i, gm in enumerate(self.gaussianPdfList_):
            gyroBiasEst += gm.w*gm.gyroBiasRef
        # take most highest weighted est for now 
        q_BM_est = Quaternion()
        maxWeight = 0.
        for i, gm in enumerate(self.gaussianPdfList_):
            if gm.w > maxWeight:
                q_BM_est = copy.deepcopy(gm.q_BMref)
                maxWeight = gm.w

        return gyroBiasEst, q_BM_est

    
    def sampleFromThisGaussianMixList(self,seed=None):
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed=seed)
         # Extract weights in the same order as gaussianPdfList_
        weights = np.array([g.w for g in self.gaussianPdfList_], dtype=float)


        # Draw a component index according to mixture weights
        intArray =np.arange(0,len(self.gaussianPdfList_),1,dtype=int)
        l = rng.choice(a=intArray, p=weights)

        # Pull out mean and covariance
        mean = self.gaussianPdfList_[l].mx
        cov  = self.gaussianPdfList_[l].Pxx

        # Sample from the selected Gaussian
        sample = rng.normal(loc=mean, scale=np.sqrt(cov))

        return sample
    



class MoonGaussianMixtureModel():
    def __init__(self, Lx_input):
        self._nx = 12
        self.Lx = Lx_input
        ## list of full state objs
        self._gaussianPdfList = []

    @ staticmethod
    def gaussian_to_gmm(mx, Pxx, Lx=5, spread_sigma=3.0):
        mx = np.atleast_1d(mx)
        n = mx.shape[0]

        # --- Compute eigen decomposition of the covariance ---
        vals, vecs = np.linalg.eigh(Pxx)  # vals = eigenvalues (variances), vecs = eigenvectors

        stds = np.sqrt(vals)               # standard deviations along principal axes

        # --- Select K points evenly spread across eigen-directions ---
        # Example: K=5 produces positions [-3σ, -1.5σ, 0, 1.5σ, 3σ]
        a = np.linspace(-spread_sigma, spread_sigma, Lx)

        mxs = []
        for i in range(Lx):
            # Offset in principal-axis coordinates
            offset = (a[i] * stds)
            # Transform back to original coordinates
            new_mx = mx.flatten() + vecs @ offset
            mxs.append(new_mx)

        # All components share the original covariance (adjustable)
        Pxx = [Pxx.copy() for _ in range(Lx)]

        # Equal weights
        weights = np.ones(Lx) / Lx

        return weights, mxs, Pxx

        




        