import sys, os

# Add the basilisk root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from helpers.attitude.Quaternion import Quaternion
import pickle
from scipy.integrate import solve_ivp 
import matplotlib.pyplot as plt


def T1(theta):
    """Passive DCM for rotation about axis 1 (x-axis)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])

def T2(theta):
    """Passive DCM for rotation about axis 2 (y-axis)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])

def T3(theta):
    """Passive DCM for rotation about axis 3 (z-axis)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])


with open("data/MoonCentralBody.pkl", "rb") as f:
    sim_data = pickle.load(f)

timeData = sim_data["time"] * 1e-9 # to seconds
sc_pos   = sim_data["sc_pos"]
sc_vel   = sim_data["sc_vel"]
moon_pos = sim_data["moon_pos"]
moon_vel = sim_data["moon_vel"]
earth_pos = sim_data["earth_pos"]
earth_vel = sim_data["earth_vel"]

print("Simulation data successfully unboxed.")

r_MoonEarth_N = moon_pos - earth_pos
v_MoonEarth_N = moon_vel - earth_vel


# create MCMF routine
inclination_MoonNPole = 6.68 # deg


# grab init moon r and r dot
r_MN_N_init = r_MoonEarth_N[0,:]
rd_MN_N_init = v_MoonEarth_N[0,:]

# check orthogonality
dotPrd = np.linalg.vecdot(
    r_MN_N_init/np.linalg.norm(r_MN_N_init),
    rd_MN_N_init/np.linalg.norm(rd_MN_N_init)
)

# Direction Cosine Matrix from earth centered inertial frame to earth-moon rotation frame
rhat1 = r_MN_N_init/np.linalg.norm(r_MN_N_init)
v_perp = rd_MN_N_init - np.dot(rd_MN_N_init, rhat1) * rhat1
rhat2 = v_perp / np.linalg.norm(v_perp)
rhat3 = np.cross(rhat1, rhat2)
T_RN = np.column_stack((rhat1, rhat2, rhat3))
                    
# DCM definition of MCMF frame
zhat = np.array([0.,0.,1.])
xhat = np.array([1.,0.,0.])
mhat3 = T2(np.deg2rad(inclination_MoonNPole)) @ zhat
rhat1_orthog = xhat - np.dot(xhat,mhat3) * xhat
rhat1_orthog = rhat1_orthog / np.linalg.norm(rhat1_orthog)
mhat2 = np.cross(mhat3,rhat1_orthog)
mhat1 = np.cross(mhat2,mhat3)
T_MN = np.column_stack([mhat1,mhat2,mhat3])


cosang = np.clip(np.dot(mhat3, zhat), -1.0, 1.0)
angle_deg = np.rad2deg(np.arccos(cosang))

q_MN_0 = Quaternion.from_DCM(T_MN)
q_MN_0.ensureScalarPos()
q_MN_0.normalize()



# random vector
randVec = np.array([52.,8.,96.])
randVec = randVec/np.linalg.norm(randVec)


# let's propagate truth MCMF frame attitude
# w_RN_R = np.array([0.0, 0.0, 2*np.pi/29.530/24/3600])
w_MN_M = np.array([0.0, 0.0, 2*np.pi/27.322/24/3600])


def dqdt_wrapper(t, q):
    return Quaternion.dqdt(t, w_MN_M, q)

# lets set up a for loop over sim time
# create a fake landmark on the 'lunar surface' 
# propagate MCMF attitude and ensure that the landmark does not move in it's frame 

q_MN_store = []

q_MN_tkm =q_MN_0.as_array()
inclChk_deg= np.rad2deg(2*np.arcsin(q_MN_tkm[1]))

for i, tk in enumerate(timeData):
    if i == 0:
        q_MN_store.append(q_MN_0)
        continue

    tkm = timeData[i-1]

    sol = solve_ivp(
        fun=dqdt_wrapper,
        t_span=[tkm, tk],
        y0=q_MN_tkm,
        method='RK45',
        rtol=1e-9,
        atol=1e-9
    )

    q_MN_tk = sol.y[:, -1]

    # Normalize quaternion
    if q_MN_tk[-1] < 0:
        q_MN_tk = -q_MN_tk
    q_MN_tk /= np.linalg.norm(q_MN_tk)

    qObj = Quaternion.from_array(q_MN_tk)
    q_MN_store.append(qObj)
    q_MN_tkm = q_MN_tk




# difference in attitude solutions should be only ab zhat
# this magnitude should be equal to w_MN_M*dt

dtTot = timeData[-1] - timeData[0]
totalAngleRad = w_MN_M*dtTot

q_end= q_MN_store[-1]
q_beg = q_MN_store[0]

eulerVec = Quaternion.computeEulerVecAttErrorFromQuats(q_ref=q_end,q_est=q_beg)
prAngle = np.linalg.norm(eulerVec)
prAxis = eulerVec/prAngle


# error in the beginning quat wrt the end quat should be a positive rotation of omega*dt
princAxisDotAngRateHat = np.dot(prAxis,totalAngleRad/np.linalg.norm(totalAngleRad))



    
    










