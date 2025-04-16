#
# Basilisk Scenario Script and Integrated Test
#
# Purpose:  Integrated test showing how to setup and run a Python BSK module with C/C++ modules
# Author:   Hanspeter Schaub
# Creation Date:  Jan. 16, 2021
#

import sys
import os
import math
import quaternion


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# The path to the location of Basilisk
# Used to get the location of supporting data.
from Basilisk import __path__
# import message declarations
from Basilisk.architecture import messaging
from Basilisk.architecture import bskLogging
from Basilisk.fswAlgorithms import attTrackingError
# import FSW Algorithm related support
# from Basilisk.fswAlgorithms import mrpFeedback
from Basilisk.fswAlgorithms import inertial3DSpin
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import simpleNav
# import simulation related support
from Basilisk.simulation import spacecraft
from Basilisk.utilities import simIncludeGravBody
from Basilisk.utilities import orbitalMotion
# import general simulation support files
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import SpherePlot

# Add the full path to the 'examples' folder
sys.path.append(os.path.abspath("/Users/lukeserrano/repos/personal/basilisk/examples"))

from BskSim.plotting import BSK_Plotting as bskPlot
# attempt to import vizard
from Basilisk.utilities import vizSupport
from Basilisk.architecture import sysModel



bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])


def run(show_plots):
    """
    The scenarios can be run with the followings setups parameters:

    Args:
        show_plots (bool): Determines if the script should display plots

    """

    #
    #  From here on scenario python code is found.  Above this line the code is to setup a
    #  unitTest environment.  The above code is not critical if learning how to code BSK.
    #

    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    # set the simulation time variable used later on
  #  simulationTime = macros.min2nano(5.)

    #
    #  create the simulation process
    #
    dynProcess = scSim.CreateNewProcess(simProcessName, 10)

    # create the dynamics task and specify the integration update time
    simulationTimeStep = macros.sec2nano(.1)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    #
    #   setup the simulation tasks/objects
    #

    # initialize spacecraft object and set properties
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bsk-Sat"
    # define the simulation inertia
    '''
    I = [900., 0., 0.,
         0., 800., 0.,
         0., 0., 600.]
    '''
    I = [30., 10., 5.,
         10., 20., 3.,
         5., 3., 15.]
    scObject.hub.mHub = 10.0  # kg - spacecraft mass
    
    # scObject.hub.mHub = 750.0  # kg - spacecraft mass
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # m - position vector of body-fixed point B relative to CM
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
    # initial att states
    scObject.hub.sigma_BNInit =  rbk.C2MRP(np.identity(3)) # rbk.PRV2MRP([macros.D2R*45.0, 0.0, 0.0])  # sigma_BN_B
    scObject.hub.omega_BN_BInit = [0.0, macros.D2R*0.0, 0.0]  # rad/s - omega_BN_B

    # clear prior gravitational body and SPICE setup definitions
    gravFactory = simIncludeGravBody.gravBodyFactory()
    # setup Earth Gravity Body
    earth = gravFactory.createEarth()
    earth.isCentralBody = True  # ensure this is the central gravitational body
    mu = earth.mu
    # attach gravity model to spacecraft
    gravFactory.addBodiesTo(scObject)

    #
    #   setup orbit and simulation time
    #
    # setup the orbit using classical orbit elements
    oe = orbitalMotion.ClassicElements()
    rLEO = 7000. * 1000  # meters
    rGEO = math.pow(earth.mu / math.pow((2. * np.pi) / (24. * 3600.), 2), 1. / 3.)
    oe.a = rLEO
    oe.e = 0.0001
    oe.i = 0.0 * macros.D2R
    oe.Omega = -90 * macros.D2R
    oe.omega = 0.0 * macros.D2R
    oe.f = 0.0 * macros.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    scObject.hub.r_CN_NInit = rN  # m - r_CN_N
    scObject.hub.v_CN_NInit = vN  # m - v_CN_N

    # set the simulation time
    n = np.sqrt(earth.mu / oe.a / oe.a / oe.a)
    P = 2. * np.pi / n
    # simulationTime = macros.sec2nano(0.25 * P)
    simulationTime = macros.min2nano(3.)

    # add spacecraft object to the simulation process
    scSim.AddModelToTask(simTaskName, scObject)

    


    # setup extForceTorque module
    # the control torque is read in through the messaging system
    extFTObject = extForceTorque.ExtForceTorque()
    extFTObject.ModelTag = "externalDisturbance"
    scObject.addDynamicEffector(extFTObject)
    scSim.AddModelToTask(simTaskName, extFTObject)

    # add the simple Navigation sensor module.  This sets the SC attitude, rate, position
    # velocity navigation message
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(simTaskName, sNavObject)

    #
    #   setup the FSW algorithm tasks
    #

    ## DOC: quat = [q,Q_vec]'
    # setup desired attitude quatBodyRateAccelPropagation guidance module 
    attDesPropObj = quatBodyRateAccelPropagation()
    attDesPropObj.ModelTag = "quatDesProp"
        # assume q_ItoB(t=0) is identity 
    #attDesPropObj.current_q_ItoB_des = np.array([2*np.sqrt(2), 2*np.sqrt(2), 0.0, 0.0]) #initial des att is 90 deg rot ab inertial x
    attDesPropObj.current_q_ItoB_des = rbk.PRV2EP([macros.D2R*0.0, 0.0, 0.0])
    attDesPropObj.last_q_ItoB_des = attDesPropObj.current_q_ItoB_des
    attDesPropObj.omega_ItoB_B_des = np.array([macros.D2R*0.0, macros.D2R*0.063,macros.D2R*0.0]) # desired ang rate | LEO orbit, 90min/2pi -> .0011 rad/s
    attDesPropObj.ddtOmega_ItoB_B_des = np.zeros((3,1))
    scSim.AddModelToTask(simTaskName, attDesPropObj)

    # setup the attitude tracking error evaluation module
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    scSim.AddModelToTask(simTaskName, attError)
    
    # setup Error Quaternion closed loop control module
    pyErrQuatCtrlr = errQuatFeedback()
    pyErrQuatCtrlr.ModelTag = "pyErrQuat_FB"
    scSim.AddModelToTask(simTaskName, pyErrQuatCtrlr)

    #
    #   Setup data logging before the simulation is initialized
    #
    numDataPoints = 100
    samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStep, numDataPoints)
    desAttlog = attDesPropObj.currentDesAttMsgOut.recorder(samplingTime)
    attErrorLog = attError.attGuidOutMsg.recorder(samplingTime)
    errQuatLog = pyErrQuatCtrlr.cmdTorqueOutMsg.recorder(samplingTime)
    navSolLog =  sNavObject.transOutMsg.recorder(samplingTime)
    navAttSolLog =  sNavObject.attOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, desAttlog)
    scSim.AddModelToTask(simTaskName, attErrorLog)
    scSim.AddModelToTask(simTaskName, errQuatLog)
    scSim.AddModelToTask(simTaskName, navSolLog)
    scSim.AddModelToTask(simTaskName, navAttSolLog)
    #
    # connect the messages to the modules
    #
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    attDesPropObj.navAttMsgIn.subscribeTo(sNavObject.attOutMsg)
    attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
    attError.attRefInMsg.subscribeTo(attDesPropObj.currentDesAttMsgOut)
    pyErrQuatCtrlr.navAttMsgIn.subscribeTo(sNavObject.attOutMsg)
    pyErrQuatCtrlr.scMassIn.subscribeTo(scObject.scMassOutMsg)
    pyErrQuatCtrlr.desRefAttIn.subscribeTo(attDesPropObj.currentDesAttMsgOut)
    extFTObject.cmdTorqueInMsg.subscribeTo(pyErrQuatCtrlr.cmdTorqueOutMsg)

    # if this scenario is to interface with the BSK Viz, uncomment the following lines
    vizSupport.enableUnityVisualization(scSim, simTaskName, scObject
                                        # , saveFile=fileName
                                        )

    #
    #   initialize Simulation
    #
    scSim.InitializeSimulation()

    #
    #   configure a simulation stop time and execute the simulation run
    #
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    #
    #   retrieve the logged data
    #
    dataLr = errQuatLog.torqueRequestBody
    dataSigmaBR = attErrorLog.sigma_BR

    for errMRP in range(dataSigmaBR.shape[0]):
        dataSigmaBR[errMRP,:] = rbk.MRP2PRV(attErrorLog.sigma_BR[errMRP,:]) * macros.R2D 

    dataOmegaBR = attErrorLog.omega_BR_B * macros.R2D 
    timeAxis = attErrorLog.times()
    np.set_printoptions(precision=16)

    #
    #   plot the results
    #

    plt.close("all")  # clears out plots from earlier test runs
    plt.figure(1)
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2MIN, dataSigmaBR[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\sigma_' + str(idx) + '$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'Attitude Error $\sigma_{B/R}$ [deg]')
    plt.grid(True,'both','both')
    figureList = {}
    pltName = fileName + "1"
    figureList[pltName] = plt.figure(1)

    plt.figure(2)
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2MIN, dataLr[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label='$L_{r,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Control Torque $L_r$ [Nm]')
    plt.grid(True,'both','both')
    pltName = fileName + "2"
    figureList[pltName] = plt.figure(2)

    plt.figure(3)
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2MIN, dataOmegaBR[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\omega_{BR,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Rate Tracking Error [deg/s] ')
    plt.grid(True,'both','both')

    '''
## position
    navSolTime = navSolLog.times()
    navSolPos = navSolLog.r_BN_N
    plt.figure(4)
    time = timeAxis * macros.NANO2MIN
    
    for idx in range(3):
        plt.plot(navSolTime * macros.NANO2MIN, navSolPos[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$r_{BN_N,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('ECI pos [m] ')
    plt.grid(True,'both','both')
    # bskPlot.plot3components(navSolTime,navSolPos)


    ## desired quat
    q_ItoB_des = desAttlog.sigma_RN
    q_ItoB_nav = navAttSolLog.sigma_BN
    attTime = desAttlog.times()

    #for i in range(q_ItoB_des.shape[0])
    #    q_est = quaternion(q_ItoB_nav[i,:])
    plt.figure(5)
    for idx in range(3):
        plt.plot(attTime * macros.NANO2MIN, q_ItoB_des[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$q_{BN,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('des quat ')
    plt.grid(True,'both','both')

    '''   




    if show_plots:
        plt.show()

    # close the plots being saved off to avoid over-writing old and new figures
    plt.close("all")

    return figureList



class quatBodyRateAccelPropagation(sysModel.SysModel):
    def __init__(self):
        super(quatBodyRateAccelPropagation, self).__init__()
        # parameters
        self.navAttMsgIn = messaging.NavAttMsgReader()
        self.currentDesAttMsgOut = messaging.AttRefMsg()
        self.priorTime = 0.0
        
        # last Desired Attitude
        self.last_q_ItoB_des = np.array([1, 0, 0, 0])
        # Current Desired Attitude
        self.current_q_ItoB_des = np.array([1, 0, 0, 0])
        self.omega_ItoB_B_des = np.zeros(3)
        self.ddtOmega_ItoB_B_des = np.zeros(3)
    
    def Reset(self, CurrentSimNanos):
        """insert reset"""
        return

    def UpdateState(self, CurrentSimNanos):
        
        # get nav soluiton
        navSol = self.navAttMsgIn()

        # compute dt
        if self.priorTime < 1E-5:
            dt = 0.0
        else:
            dt = (CurrentSimNanos * macros.NANO2SEC) - self.priorTime

        # get last desired attiude
        last_q_ItoB = self.last_q_ItoB_des

        # integrate
        omega = self.omega_ItoB_B_des
        sol = solve_ivp(
        fun=lambda t, q: quatBodyRateAccelPropagation.quat_derivative(t, q, omega),
        t_span=[0.0, dt],
        y0=last_q_ItoB,
        method='RK45',
        rtol=1e-9,
        atol=1e-9
        )
        # brute force normalize quaternion
        new_q_ItoB_des = sol.y[:, -1]
        new_q_ItoB_des = new_q_ItoB_des / np.linalg.norm(new_q_ItoB_des)
        self.current_q_ItoB_des = new_q_ItoB_des

        # publish msg
        C_ItoB = rbk.MRP2C(navSol.sigma_BN)
        attRefMsg = messaging.AttRefMsgPayload()
        attRefMsg.sigma_RN = rbk.EP2MRP(new_q_ItoB_des)
        attRefMsg.omega_RN_N = C_ItoB @ self.omega_ItoB_B_des
        attRefMsg.domega_RN_N = C_ItoB @ self.ddtOmega_ItoB_B_des
        self.currentDesAttMsgOut.write(attRefMsg, CurrentSimNanos, self.moduleID)

        # set for next iteration
        self.priorTime = CurrentSimNanos * macros.NANO2SEC
        self.last_q_ItoB_des = self.current_q_ItoB_des


        # loggging 
        self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"Guide_internal: Time: {CurrentSimNanos * 1.0E-9} s")
        self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"Guide_internal: Des_sigma_BR: {attRefMsg.sigma_RN}")
        self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"Guide_internal: Des_omega_BR_B: {attRefMsg.omega_RN_N}")
        self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"Guide_internal: dt: {dt}")

        return


    @staticmethod
    def quat_derivative(t, q, omega):
        """ODE function: dq/dt = 0.5 * B(q) * omega"""
        return 0.5 * rbk.BmatEP(q) @ omega



class errQuatFeedback(sysModel.SysModel):
    def __init__(self):
        super(errQuatFeedback, self).__init__()
        
        # LQR determined gains
        self.K1 = np.array([[0.9999999999999996, 0.0, 0.0], [0.0, 0.9999999999999997, 0.0], [0.0, 0.0, 0.9999999999999997]])
        self.K2 = np.array([[1.732050807568878, 0.0, 0.0], [0.0, 1.732050807568878, 0.0], [0.0, 0.0, 1.732050807568878]])

        # LQR state cost weight (Q) and control cost weight (R) for cost calc
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.R = np.diag([1.0, 1.0, 1.0])
        
        # Input nav att message
        self.navAttMsgIn = messaging.NavAttMsgReader()
        # Input des att message 
        self.desRefAttIn = messaging.AttRefMsgReader()
        # Input sc mass state
        self.scMassIn = messaging.SCMassPropsMsgReader()
        # Output body torque message 
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()

    def Reset(self, CurrentSimNanos):
        

        # Ensure that self.dataInMsg's are linked
        if not self.navAttMsgIn.isLinked():
            self.bskLogger.bskLog(
                bskLogging.BSK_ERROR, "errQuatFeedback.navAttMsgIn is not linked."
            )
        if not self.desRefAttIn.isLinked():
            self.bskLogger.bskLog(
                bskLogging.BSK_ERROR, "errQuatFeedback.desRefAttIn is not linked."
            )
        if not self.cmdTorqueOutMsg.isLinked():
            self.bskLogger.bskLog(
                bskLogging.BSK_ERROR, "errQuatFeedback.cmdTorqueOutMsg is not linked."
            )

        '''
        # Initialiazing self.cmdTorqueMsg
        cmdTorqueMsg = messaging.CmdTorqueBodyMsg()
        cmdTorqueMsg.dataVector = np.array([0, 0, 0])
        self.cmdTorqueOutMsg.write(cmdTorqueMsg, CurrentSimNanos, self.moduleID)

        self.bskLogger.bskLog(bskLogging.BSK_INFORMATION, "Reset in errQuatFeedback")
        '''
        
        return


    def UpdateState(self, CurrentSimNanos):

        # Ensure that self.dataInMsg's are linked
        if not self.navAttMsgIn.isLinked():
            self.bskLogger.bskLog(
                bskLogging.BSK_ERROR, "errQuatFeedback.navAttMsgIn is not linked."
            )
        if not self.desRefAttIn.isLinked():
            self.bskLogger.bskLog(
                bskLogging.BSK_ERROR, "errQuatFeedback.desRefAttIn is not linked."
            )
        if not self.cmdTorqueOutMsg.isLinked():
            self.bskLogger.bskLog(
                bskLogging.BSK_ERROR, "errQuatFeedback.cmdTorqueOutMsg is not linked."
            )
        
        # copy nav and des att msg;s
        navMsgBuffer = self.navAttMsgIn()
        scMassBuffer = self.scMassIn()
        desAttMsgBuffer = self.desRefAttIn()

        # Set output message
        cmdTorqueMsg = messaging.CmdTorqueBodyMsgPayload()

        ##  optimal output torque to achieve error dynamics with LQR gains
        # estimates 
            # w_ItoB_B
        C_ItoB = rbk.MRP2C(navMsgBuffer.sigma_BN)
        w = np.array(navMsgBuffer.omega_BN_B)
        w = np.array(w)
        w_skew = errQuatFeedback.skew(w)
        Omega_ = errQuatFeedback.calcOmegaMatFromVec(w)
        q = rbk.MRP2EP(navMsgBuffer.sigma_BN) # q_ItoB
        # mass
        inertia = scMassBuffer.ISC_PntB_B
        inertia = np.array(inertia)
        # desired att
            # w_ItoB_B
        w_des = C_ItoB @ np.array(desAttMsgBuffer.omega_RN_N)
        w_des_skew = errQuatFeedback.skew(w_des)
        dotw_des = C_ItoB @ np.array(desAttMsgBuffer.domega_RN_N)
        
        qd = rbk.MRP2EP(desAttMsgBuffer.sigma_RN) # q_ItoB_des
        dotqd = 0.5 * rbk.BmatEP(qd) @ w_des
        dbleDot_qd = (
            (0.5 * rbk.BmatEP(qd) @ dotw_des) - 
            (0.25 * (w_des.T @ w_des) * qd)
        )
        # applied torque calculation
        term1 = w_skew @ inertia @ w 
        term2 = 2.0*inertia @ np.linalg.inv((rbk.BmatEP(qd).T @ rbk.BmatEP(q))) 
        k1term = self.K1 @ rbk.BmatEP(qd).T
        k2term = self.K2 @ (0.5*rbk.BmatEP(qd).T @ Omega_ + rbk.BmatEP(dotqd).T)
        term3 = ( 
            (0.25*(w.T @ w) * rbk.BmatEP(qd).T) - (rbk.BmatEP(dotqd).T @ Omega_) -
             rbk.BmatEP(dbleDot_qd).T)
        
        appliedTorque_2 = term1 + (term2 @ (term3 - k1term - k2term) @ q)

        appliedTorque = (
            ( w_skew @ inertia @ w ) +
            ( 2.0*inertia @ np.linalg.inv((rbk.BmatEP(qd).T @ rbk.BmatEP(q))) ) @
            ( (0.25*(w.T @ w) * rbk.BmatEP(qd).T) - (rbk.BmatEP(dotqd).T @ Omega_) -
             rbk.BmatEP(dbleDot_qd).T - (self.K1 @ rbk.BmatEP(qd).T) - 
             self.K2 @ (0.5*rbk.BmatEP(qd).T @ Omega_ + rbk.BmatEP(dotqd).T) ) @ q
        )

        # write output message
        cmdTorqueMsg.torqueRequestBody = appliedTorque.tolist()
        self.cmdTorqueOutMsg.write(cmdTorqueMsg, CurrentSimNanos, self.moduleID)


        ## logging
        self.bskLogger.bskLog(
            bskLogging.BSK_INFORMATION,
            f"Python Module ID {self.moduleID} ran Update at {CurrentSimNanos*1e-9}s",
        )


        # Cost Calculation

        # All Python SysModels have self.bskLogger available
        # The logger level flags (i.e. BSK_INFORMATION) may be
        # accessed from sysModel
        if True:
            """Sample Python module method"""
            self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"Time: {CurrentSimNanos * 1.0E-9} s")
            self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"Des_sigma_BR: {desAttMsgBuffer.sigma_RN}")
            self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"nav_sigma_BN: {navMsgBuffer.sigma_BN}")
            self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"Des_qd: {qd}")
            self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"nav_q: {q}")
            self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"errQuat_vec: {rbk.BmatEP(qd).T @ q}")
            self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"term1: {term1}")
            self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"term2: {term2}")
            self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"term3: {term3}")
            self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"k1term: {k1term}")
            self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"k2term: {k2term}")


            self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"Des_omega_BR_B: {w_des}")
            self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"nav_omega_BN_B: {w}")
            self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"TorqueRequestBody: {cmdTorqueMsg.torqueRequestBody}")
            self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"appliedTorque_2: {appliedTorque_2}")

        return
        
    @staticmethod
    def calcOmegaMatFromVec(v):
        # Ensure v is a 3x1 numpy array
        v = np.array(v).reshape((3, 1))  # Convert to 3x1 if not already
        if v.shape != (3, 1):
            raise ValueError("Input vector must be a 3x1 numpy array.")
        
        OmegaMat = np.zeros((4, 4))
        # Fill in top row: 
        OmegaMat[0, 1:] = -v.T
        # Fill in left column:
        OmegaMat[1:, 0] = v[:, 0]
        # Fill in bottom-right 3x3: 
        OmegaMat[1:, 1:] = -errQuatFeedback.skew(v)

        return OmegaMat
    
    @staticmethod
    def skew(v):
        v = np.array(v).flatten()  # Ensure it's 1D
        if v.shape[0] != 3:
            raise ValueError("Input must be a 3-element vector.")
        return np.array([
            [0,     -v[2],  v[1]],
            [v[2],   0,    -v[0]],
            [-v[1],  v[0],  0]
        ])
        




if __name__ == "__main__":
    run(show_plots=True)
