#
# Basilisk Scenario Script and Integrated Test
#
# Purpose:  Integrated test showing how to setup and run a Python BSK module with C/C++ modules
# Author:   Hanspeter Schaub
# Creation Date:  Jan. 16, 2021
#

import os

import matplotlib.pyplot as plt
import numpy as np
# The path to the location of Basilisk
# Used to get the location of supporting data.
from Basilisk import __path__
# import message declarations
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import attTrackingError
# import FSW Algorithm related support
# from Basilisk.fswAlgorithms import mrpFeedback
from Basilisk.fswAlgorithms import inertial3DSpin
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import simpleNav
# import simulation related support
from Basilisk.simulation import spacecraft
# import general simulation support files
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions
from Basilisk.utilities import RigidBodyKinematics as rbk
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
    simulationTime = macros.min2nano(10.)

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
    I = [900., 0., 0.,
         0., 800., 0.,
         0., 0., 600.]
    scObject.hub.mHub = 750.0  # kg - spacecraft mass
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # m - position vector of body-fixed point B relative to CM
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
    scObject.hub.sigma_BNInit = [[0.1], [0.2], [-0.3]]  # sigma_BN_B
    scObject.hub.omega_BN_BInit = [[0.001], [-0.01], [0.03]]  # rad/s - omega_BN_B

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

    # write a constant desired atittude
    #
    # Reference Frame Message = ECI frame
    #
    RefStateOutData = messaging.AttRefMsgPayload()  # Create a structure for the input message
    sigma_R0N = rbk.EP2MRP(np.array([1, 0 , 0, 0])) # reference frame = ECI frame
    RefStateOutData.sigma_RN = sigma_R0N
    omega_R0N_N = np.array([0.0, 0.0, 0.0]) # desired ang rate | LEO orbit, 90min/2pi -> .0011 rad/s
    RefStateOutData.omega_RN_N = omega_R0N_N
    domega_R0N_N = np.array([0.0, 0.0, 0.0]) # desired ang accel
    RefStateOutData.domega_RN_N = domega_R0N_N
    refStateMsg = messaging.AttRefMsg().write(RefStateOutData)

    # setup inertial3Dspin guidance module 
    # R = body , R0 = ECI as defined above
    inertial3DSpinObj = inertial3DSpin.inertial3DSpin()
    inertial3DSpinObj.ModelTag = "inertial3D"
    scSim.AddModelToTask(simTaskName, inertial3DSpinObj)
    inertial3DSpinObj.omega_RR0_R0 = np.array([0.0, 0.0011, 0.0])  # set the desired inertial orientation

    # setup the attitude tracking error evaluation module
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    scSim.AddModelToTask(simTaskName, attError)

    # setup Error Quaternion closed loop control module
    pyErrQuatCtrlr = errQuatFeedback()
    pyErrQuatCtrlr.ModelTag = "pyErrQuat_FB"
    pyErrQuatCtrlr.K = 3.5
    pyErrQuatCtrlr.P = 30.0
    scSim.AddModelToTask(simTaskName, pyErrQuatCtrlr)

    #
    #   Setup data logging before the simulation is initialized
    #
    numDataPoints = 50
    samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStep, numDataPoints)
    attErrorLog = attError.attGuidOutMsg.recorder(samplingTime)
    mrpLog = pyErrQuatCtrlr.cmdTorqueOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, attErrorLog)
    scSim.AddModelToTask(simTaskName, mrpLog)

    #
    # connect the messages to the modules
    #
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
    attError.attRefInMsg.subscribeTo(inertial3DSpinObj.attRefOutMsg)
    pyErrQuatCtrlr.guidInMsg.subscribeTo(attError.attGuidOutMsg)
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
    dataLr = mrpLog.torqueRequestBody
    dataSigmaBR = attErrorLog.sigma_BR
    dataOmegaBR = attErrorLog.omega_BR_B
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
    plt.ylabel(r'Attitude Error $\sigma_{B/R}$')
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
    pltName = fileName + "2"
    figureList[pltName] = plt.figure(2)

    plt.figure(3)
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2MIN, dataOmegaBR[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\omega_{BR,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Rate Tracking Error [rad/s] ')

    if show_plots:
        plt.show()

    # close the plots being saved off to avoid over-writing old and new figures
    plt.close("all")

    return figureList






class errQuatFeedback(sysModel.SysModel):
    def __init__(self):
        super(errQuatFeedback, self).__init__()

        # Proportional gain term used in control
        self.K = 0
        # Derivative gain term used in control
        self.P = 0
        # Input guidance structure message
        self.guidInMsg = messaging.AttGuidMsgReader()
        # Output body torque message name
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()

        

        # 
    def Reset(self, CurrentSimNanos):
        # Ensure that self.dataInMsg is linked
        if not self.dataInMsg.isLinked():
            self.bskLogger.bskLog(
                bskLogging.BSK_ERROR, "TestPythonModule.dataInMsg is not linked."
            )

        # Initialiazing self.dataOutMsg
        payload = self.dataOutMsg.zeroMsgPayload
        payload.dataVector = np.array([0, 0, 0])
        self.dataOutMsg.write(payload, CurrentSimNanos, self.moduleID)

        self.bskLogger.bskLog(bskLogging.BSK_INFORMATION, "Reset in TestPythonModule")



    def UpdateState(self, CurrentSimNanos):
        # Read input message
        inPayload = self.dataInMsg()
        inputVector = inPayload.dataVector

        # 

        # Set output message
        payload = self.dataOutMsg.zeroMsgPayload
        payload.dataVector = (
            self.dataOutMsg.read().dataVector + np.array([0, 1, 0]) + inputVector
        )
        self.dataOutMsg.write(payload, CurrentSimNanos, self.moduleID)

        self.bskLogger.bskLog(
            bskLogging.BSK_INFORMATION,
            f"Python Module ID {self.moduleID} ran Update at {CurrentSimNanos*1e-9}s",
        )

    
    @staticmethod
    def calcXiMatFromQuat(quat)
        
        return XiMat
        
    @staticmethod
    def calcOmegaMatFromVec(vector)

        return OmegaMat
        




if __name__ == "__main__":
    run()