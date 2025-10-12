import sys, os

# Add the basilisk root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from helpers.attitude import Quaternion as q
import pickle
from scipy.integrate import solve_ivp 


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


with open("data/halosim.pkl", "rb") as f:
    sim_data = pickle.load(f)

timeData = sim_data["time"]
sc_pos   = sim_data["sc_pos"]
sc_vel   = sim_data["sc_vel"]
moon_pos = sim_data["moon_pos"]
moon_vel = sim_data["moon_vel"]

print("Simulation data successfully unboxed.")



# create MCMF routine
inclination_MoonNPole = 6.68 # deg


# grab init moon r and r dot
r_MN_N_init = moon_pos[0,:]
rd_MN_N_init = moon_vel[0,:]

# Direction Cosine Matrix from earth centered inertial frame to earth-moon rotation frame
rhat1 = r_MN_N_init/np.linalg.norm(r_MN_N_init)
rhat2 = rd_MN_N_init/np.linalg.norm(rd_MN_N_init)
rhat3 = np.cross(rhat1,rhat2)
T_RN = np.array([rhat1,rhat2,rhat3])
                    
# DCM definition of MCMF frame
mhat3 = T2(np.deg2rad(inclination_MoonNPole)) @ (r_MN_N_init/np.linalg.norm(r_MN_N_init))
mhat2 = np.cross(mhat3,rhat1)
mhat1 = np.cross(mhat2,mhat3)
T_MR = np.array([mhat1,mhat2,mhat3])


q_MR_0 = q.Quaternion.from_DCM(T_MR)




# let's propagate truth MCMF frame attitude
w_RN_R = np.array([0.0, 0.0, 2*np.pi/29.530/24/3600])
w_MR_M = np.array([0.0, 0.0, 2*np.pi/27.322/24/3600])

# lets set up a for loop over sim time
# create a fake landmark on the 'lunar surface' 
# propagate MCMF attitude and ensure that the landmark does not move in it's frame 

# integrate
sol = solve_ivp(
fun=q.Quaternion.dqdt(t,w_MR_M,q_MR.as_array())
t_span=[0.0, dt],
y0=last_q_ItoB,
method='RK45',
rtol=1e-9,
atol=1e-9
)


# brute force normalize quaternion
new_q_ItoB_des = sol.y[:, -1]
q_outIntegrator = new_q_ItoB_des
    # ensure pos real part quat
if new_q_ItoB_des[0] < 1e-6:
    print('AHH!')
    new_q_ItoB_des = -new_q_ItoB_des
new_q_ItoB_des = new_q_ItoB_des / np.linalg.norm(new_q_ItoB_des)
    











