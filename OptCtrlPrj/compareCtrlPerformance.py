
import sys
import os
import math


import matplotlib.pyplot as plt
import numpy as np
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions









def plotAttErr(timeMin,bodyEulerError,CtrlString):
    plt.figure()
    axis = ['X','Y','Z']
    for idx in range(3):
        plt.plot(timeMin, bodyEulerError[:, idx],
                color=unitTestSupport.getLineColor(idx, 3),
                label=r'$\delta\theta_' + axis[idx].lower() + '$')
        plt.legend(loc='lower right')
        plt.xlabel('Time [min]')
        plt.ylabel(r'Attitude Error Body Frame [deg]')
        plt.grid(True,'both','both')
        title = CtrlString + 'Body Frame Attitude Error'
        plt.title(title)
        pltName = title + "1"


def plotRateErr(timeMin,rateError,CtrlString):
    plt.figure()
    axis = ['X','Y','Z']
    for idx in range(3):
        plt.plot(timeMin, rateError[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\delta\omega_' + axis[idx].lower() + '$')
        plt.legend(loc='lower right')
        plt.xlabel('Time [min]')
        plt.ylabel(r'Ang Rate Tracking Error [deg/s]')
        plt.grid(True,'both','both')
        title = CtrlString + 'Ang Rate Tracking Error'
        plt.title(title)
        pltName = title + "1"

def plotCostQuatVsMrp(TimeMin,quatCost,MrpCost):
    plt.figure()
    plt.plot(TimeMin, quatCost,
                 label='Error Quaternion Ctrl Cost')
    plt.plot(TimeMin, MrpCost,
                 label='MRP Ctrl Cost')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'Cost')
    plt.grid(True,'both','both')
    title = 'Cost Comparision'
    plt.title(title)





if __name__ == "__main__":

    # retrieve data
    quatData = np.load('OptCtrlPRj/quat.npz')
    mrpData = np.load('OptCtrlPRj/mrp.npz')

    # process mrpData
    mrp_sigma_BR = mrpData['dataSigmaBR']
    mrp_dataLr = mrpData['dataLr']
    mrp_omega_BR = mrpData['dataOmegaBR']
    mrp_time_min = mrpData['time_min']
    mrp_eulerErr = np.zeros((mrp_omega_BR.shape[0],3))

    # compute euler err vec, R2D
    for i in range(mrp_omega_BR.shape[0]):
        mrp_eulerErr[i,:] = rbk.R2D* rbk.MRP2PRV(mrp_omega_BR[i,:])
        mrp_omega_BR[i,:] = rbk.R2D*mrp_omega_BR[i,:]

    # process quatData
    q_eulerErrDeg = quatData['eulerErrDeg']
    q_w_bi_b_Deg = quatData['w_bi_b_Deg']
    q_time_min = quatData['time_min']
    q_dataLr = quatData['dataLr']
    q_timeLr_min = quatData['timeLr_min']


    # compute cost | .5* [(attErr^T * attErr) + (rateErr^T * rateErr) +(Lr^T * Lr)]
    q_cost = np.zeros(q_eulerErrDeg.shape[0])
    mrp_cost = np.zeros(mrp_eulerErr.shape[0])
    costTime = q_time_min
    for i in range(mrp_omega_BR.shape[0]):
        q_euler = q_eulerErrDeg[i,:]
        q_rate = q_w_bi_b_Deg[i,:]
        q_Lr = q_dataLr[i,:]
        q_cost[i] = (
            .5*( (q_euler.T@q_euler) + (q_rate.T@q_rate) + (q_Lr.T@q_Lr))
            )
        
        mrp_euler = mrp_eulerErr[i,:]
        mrp_rate = mrp_omega_BR[i,:]
        mrp_Lr = mrp_dataLr[i,:]
        mrp_cost[i] = (
            .5*( (mrp_euler.T@mrp_euler) + (mrp_rate.T@mrp_rate) + (mrp_Lr.T@mrp_Lr))
            )

    plt.close("all")
    plotCostQuatVsMrp(costTime,q_cost,mrp_cost)

    plotAttErr(q_time_min,q_eulerErrDeg,'Error Quaternion Controller')
    plotRateErr(q_time_min,q_w_bi_b_Deg,'Error Quaternion Controller')

    plotAttErr(mrp_time_min,mrp_eulerErr,'MRP Feedback Controller')
    plotRateErr(mrp_time_min,mrp_omega_BR,'MRP Feedback Controller') 

    plt.show()
    plt.close("all")       










