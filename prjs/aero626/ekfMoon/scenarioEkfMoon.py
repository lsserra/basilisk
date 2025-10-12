
import os
from copy import copy

from Basilisk.topLevelModules import pyswice
from datetime import datetime, timedelta

from Basilisk.utilities.pyswice_spk_utilities import spkRead



import matplotlib.pyplot as plt
import numpy as np
# To play with any scenario scripts as tutorials, you should make a copy of them into a custom folder
# outside of the Basilisk directory.
#
# To copy them, first find the location of the Basilisk installation.
# After installing, you can find the installed location of Basilisk by opening a python interpreter and
# running the commands:
from Basilisk import __path__

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

# Copy the folder `{basiliskPath}/examples` into a new folder in a different directory.
# Now, when you want to use a tutorial, navigate inside that folder, and edit and execute the *copied* integrated tests.


# import simulation related support
from Basilisk.simulation import spacecraft
# general support file with common unit test functions
# import general simulation support files
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion,
                                simIncludeGravBody, unitTestSupport, vizSupport)

# always import the Basilisk messaging support

def run(showPlots, orbitCase, useSphericalHarmonics, planetCase):
    """
    At the end of the python script you can specify the following example parameters.

    Args:
        show_plots (bool): Determines if the script should display plots
        orbitCase (str):

            ======  ============================
            String  Definition
            ======  ============================
            'LEO'   Low Earth Orbit
            'GEO'   Geosynchronous Orbit
            'GTO'   Geostationary Transfer Orbit
            ======  ============================

        useSphericalHarmonics (Bool): False to use first order gravity approximation: :math:`\\frac{GMm}{r^2}`

        planetCase (str): {'Earth', 'Mars'}
    """

    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    # (Optional) If you want to see a simulation progress bar in the terminal window, the
    # use the following SetProgressBar(True) statement
    scSim.SetProgressBar(True)

    #  create the simulation process
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # create the dynamics task and specify the integration update time
    simulationTimeStep = macros.sec2nano(10.)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # setup the simulation tasks/objects
    # initialize spacecraft object and set properties
    # The dynamics simulation is setup using a Spacecraft() module.
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bsk-Sat"

    # add spacecraft object to the simulation process
    scSim.AddModelToTask(simTaskName, scObject)

    # setup Gravity Body
    # The first step to adding gravity objects is to create the gravity body factor class.  Note that
    # this call will create an empty gravitational body list each time this script is called.  Thus, there
    # is not need to clear any prior list of gravitational bodies.
    gravFactory = simIncludeGravBody.gravBodyFactory()

     # Setup gravity factory and gravity bodies
    # Include bodies as a list of SPICE names
    gravFactory = simIncludeGravBody.gravBodyFactory()
    gravBodies = gravFactory.createBodies('moon', 'earth')
    gravBodies['earth'].isCentralBody = True

    # Add gravity bodies to the spacecraft dynamics
    gravFactory.addBodiesTo(scObject)

    # Create default SPICE module, specify start date/time.
    timeInitString = "2022 August 31 15:00:00.0"
    spiceTimeStringFormat = '%Y %B %d %H:%M:%S.%f'
    timeInit = datetime.strptime(timeInitString, spiceTimeStringFormat)
    spiceObject = gravFactory.createSpiceInterface(time=timeInitString, epochInMsg=True)
    spiceObject.zeroBase = 'Earth'

    print(spiceObject.planetFrames)

    # Add SPICE object to the simulation task list
    scSim.AddModelToTask(simTaskName, spiceObject, 1)

    # Import SPICE ephemeris data into the python environment
    pyswice.furnsh_c(spiceObject.SPICEDataPath + 'de430.bsp')  # solar system bodies
    pyswice.furnsh_c(spiceObject.SPICEDataPath + 'naif0012.tls')  # leap second file
    pyswice.furnsh_c(spiceObject.SPICEDataPath + 'de-403-masses.tpc')  # solar system masses
    pyswice.furnsh_c(spiceObject.SPICEDataPath + 'pck00010.tpc')  # generic Planetary Constants Kernel

    # Set spacecraft ICs
    # Get initial moon data
    moonSpiceName = 'moon'
    moonInitialState = 1000 * spkRead(moonSpiceName, timeInitString, 'J2000', 'earth')
    moon_rN_init = moonInitialState[0:3]
    moon_vN_init = moonInitialState[3:6]
    moon = gravBodies['moon']
    earth = gravBodies['earth']
    oe = orbitalMotion.rv2elem(earth.mu, moon_rN_init, moon_vN_init)
    moon_a = oe.a

    # Direction Cosine Matrix (DCM) from earth centered inertial frame to earth-moon rotation frame
    DCMInit = np.array([moon_rN_init/np.linalg.norm(moon_rN_init),moon_vN_init/np.linalg.norm(moon_vN_init),
                        np.cross(moon_rN_init, moon_vN_init) / np.linalg.norm(np.cross(moon_rN_init, moon_vN_init))])

    # Set up non-dimensional parameters
    T_ND = np.sqrt(moon_a ** 3 / (earth.mu + moon.mu))      # non-dimensional time for one second
    mu_ND = moon.mu/(earth.mu + moon.mu)                    # non-dimensional mass

    # Set up initial conditions for the spacecraft
    x0 = 1.182212 * moon_a + moon_a * mu_ND
    z0 = 0.049 * moon_a
    dy0 = -0.167 * moon_a / T_ND
    X0 = np.array([[x0], [0], [z0]])
    dX0 = np.array([[0], [np.linalg.norm(moon_vN_init) + dy0], [0]])

    rN = np.dot(np.transpose(DCMInit), X0)
    vN = np.dot(np.transpose(DCMInit), dX0)

    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN

    # Set simulation time
    # simulationTime = macros.day2nano(17.5)
    simulationTime = macros.day2nano(5.5)
    # Setup data logging
    numDataPoints = 1000
    samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStep, numDataPoints)

    # Setup spacecraft data recorder
    scDataRec = scObject.scStateOutMsg.recorder(samplingTime)
    MoonDataRec = spiceObject.planetStateOutMsgs[0].recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, scDataRec)
    scSim.AddModelToTask(simTaskName, MoonDataRec)

    if vizSupport.vizFound:
        viz = vizSupport.enableUnityVisualization(scSim, simTaskName, scObject,
                                                  # saveFile=__file__
                                                  )
        viz.settings.showCelestialBodyLabels = 1
        viz.settings.mainCameraTarget = "earth"
        viz.settings.trueTrajectoryLinesOn = 4
        viz.settings.truePathRotatingFrame = "earth moon"

    # Initialize simulation
    scSim.InitializeSimulation()

    # Execute simulation
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    # Retrieve logged data
    posData = scDataRec.r_BN_N
    velData = scDataRec.v_BN_N
    timeData = scDataRec.times()
    moonPos = MoonDataRec.PositionVector
    moonVel = MoonDataRec.VelocityVector

    # Plot results
    np.set_printoptions(precision=16)
    plt.close("all")
    figureList = {}
    b = oe.a * np.sqrt(1 - oe.e * oe.e)

    # First plot: Draw orbit in inertial frame
    fig = plt.figure(1, figsize=tuple(np.array((1.0, b / oe.a)) * 4.75), dpi=100)
    plt.axis(np.array([-oe.rApoap, oe.rPeriap, -b, b]) / 1000 * 1.25)
    ax = fig.gca()
    ax.ticklabel_format(style='scientific', scilimits=[-5, 3])

    # Draw 'cartoon' Earth
    ax.add_artist(plt.Circle((0, 0), 0.2e5, color='b'))

    # Plot spacecraft orbit data
    rDataSpacecraft = []
    fDataSpacecraft = []
    for ii in range(len(posData)):
        oeDataSpacecraft = orbitalMotion.rv2elem(earth.mu, posData[ii], velData[ii])
        rDataSpacecraft.append(oeDataSpacecraft.rmag)
        fDataSpacecraft.append(oeDataSpacecraft.f + oeDataSpacecraft.omega - oe.omega)
    plt.plot(rDataSpacecraft * np.cos(fDataSpacecraft) / 1000, rDataSpacecraft * np.sin(fDataSpacecraft) / 1000,
             color='g', linewidth=3.0, label='Spacecraft')

    # Plot moon orbit data
    rDataMoon = []
    fDataMoon = []
    for ii in range(len(timeData)):
        oeDataMoon = orbitalMotion.rv2elem(earth.mu, moonPos[ii], moonVel[ii])
        rDataMoon.append(oeDataMoon.rmag)
        fDataMoon.append(oeDataMoon.f + oeDataMoon.omega - oe.omega)
    plt.plot(rDataMoon * np.cos(fDataMoon) / 1000, rDataMoon * np.sin(fDataMoon) / 1000, color='0.5',
             linewidth=3.0, label='Moon')

    plt.xlabel(r'$i_e$ Coord. [km]')
    plt.ylabel(r'$i_p$ Coord. [km]')
    plt.grid()
    plt.legend()
    pltName = fileName + "Fig1"
    figureList[pltName] = plt.figure(1)

    # Second plot: Draw orbit in frame rotating with the Moon (the center is L2 point)
    # x axis is moon position vector direction and y axis is moon velocity vector direction
    fig = plt.figure(2, figsize=tuple(np.array((1.0, b / oe.a)) * 4.75), dpi=100)
    plt.axis(np.array([-1e5, 5e5, -3e5, 3e5])  * 1.25)
    ax = fig.gca()
    ax.ticklabel_format(style='scientific', scilimits=[-5, 3])

    # Draw 'cartoon' Earth
    ax.add_artist(plt.Circle((0, 0), 0.2e5, color='b'))

    # Plot spacecraft orbit data
    rSpacecraft = np.zeros((len(posData), 3))

    for ii in range(len(posData)):
        # Get Moon position and velocity
        moon_rN = moonPos[ii]
        moon_vN = moonVel[ii]

        # Direction Cosine Matrix (DCM) from earth centered inertial frame to earth-moon rotation frame
        rSpacecraftMag = np.linalg.norm(posData[ii])
        rMoonMag = np.linalg.norm(moon_rN)
        DCM = [moon_rN / rMoonMag, moon_vN / np.linalg.norm(moon_vN),
               np.cross(moon_rN, moon_vN) / np.linalg.norm(np.cross(moon_rN, moon_vN))]

        # Spacecraft position in rotating frame
        rSpacecraft[ii,:] = np.dot(DCM, posData[ii])

    plt.plot(rSpacecraft[:,0] / 1000, rSpacecraft[:,1] / 1000,
             color='g', linewidth=3.0, label='Spacecraft')

    plt.xlabel('Earth-Moon axis [km]')
    plt.ylabel('Moon Velocity axis [km]')
    plt.grid()
    plt.legend()
    pltName = fileName + "Fig2"
    figureList[pltName] = plt.figure(2)

    # Third plot: Draw orbit in frame rotating with the Moon (the center is L2 point)
    # x axis is moon position vector direction and y axis is the cross product direction of the moon position vector and
    # velocity vector
    fig = plt.figure(3, figsize=tuple(np.array((1.0, b / oe.a)) * 4.75), dpi=100)
    plt.axis(np.array([-1e5, 5e5, -3e5, 3e5]) * 1.25)
    ax = fig.gca()
    ax.ticklabel_format(style='scientific', scilimits=[-5, 3])

    # Draw 'cartoon' Earth
    ax.add_artist(plt.Circle((0, 0), 0.2e5, color='b'))

    plt.plot(rSpacecraft[:, 0] / 1000, rSpacecraft[:, 2] / 1000,
             color='g', linewidth=3.0, label='Spacecraft')

    plt.xlabel('Earth-Moon axis [km]')
    plt.ylabel('Earth-Moon perpendicular axis [km]')
    plt.grid()
    plt.legend()
    pltName = fileName + "Fig3"
    figureList[pltName] = plt.figure(3)

    if showPlots:
        plt.show()

    plt.close("all")

    # Unload spice libraries
    gravFactory.unloadSpiceKernels()
    pyswice.unload_c(spiceObject.SPICEDataPath + 'de430.bsp')  # solar system bodies
    pyswice.unload_c(spiceObject.SPICEDataPath + 'naif0012.tls')  # leap second file
    pyswice.unload_c(spiceObject.SPICEDataPath + 'de-403-masses.tpc')  # solar system masses
    pyswice.unload_c(spiceObject.SPICEDataPath + 'pck00010.tpc')  # generic Planetary Constants Kernel

    return figureList

if __name__ == "__main__":
    run(
        True,        # show_plots
        'LEO',       # orbit Case (LEO, GTO, GEO)
        False,       # useSphericalHarmonics
        'moon'      # planetCase (Earth, Mars)
    )
