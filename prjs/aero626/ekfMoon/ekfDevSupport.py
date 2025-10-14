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
lunarObliquityToEcliptic = 1.54 # deg


# grab init moon r and r dot
r_MN_N_init = r_MoonEarth_N[0,:]
rd_MN_N_init = v_MoonEarth_N[0,:]

# check orthogonality
dotPrd = np.linalg.vecdot(
    r_MN_N_init/np.linalg.norm(r_MN_N_init),
    rd_MN_N_init/np.linalg.norm(rd_MN_N_init)
)
                    
# DCM definition of MCMF frame
zhat = np.array([0.,0.,1.])
xhat = np.array([1.,0.,0.])
mhat3 = T2(np.deg2rad(lunarObliquityToEcliptic)) @ zhat
xhat1_orthog = xhat - np.dot(xhat,mhat3) * xhat
rhat1_orthog = xhat1_orthog / np.linalg.norm(xhat1_orthog)
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

r_BM_M_store = []
drMdt_BM_M_store = [] # time derivative of position B wrt M, as seen from M, coordinatized in M
q_MN_store = []

q_MN_tkm =q_MN_0.as_array()
inclChk_deg= np.rad2deg(2*np.arcsin(q_MN_tkm[1]))


# take vector in M and map to N, then plot traj
r_PM_M = np.array([1737.4e3 + 1.e3 , 0.0, 0.0])
eclipticPlane_r_PM_M = None
r_PM_N_store = []
eclipticPlane_r_PM_M_store = []

for i, tk in enumerate(timeData):
    if i == 0:
        q_MN_store.append(q_MN_0)
        continue

    tkm = timeData[i-1]
    ### MCMF TRUTH GENERATION ###

    # --- attitude --- #
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

    q_MN_tk_obj = qObj


    # --- position --- #
    r_BM_N = sc_pos[i,:]
    r_BM_M = q_MN_tk_obj.rotate(r_BM_N)
    r_BM_M_store.append(r_BM_M)

    # --- velocity --- #
    rdot_BM_N = sc_vel[i,:]
    drMdt_BM_M = rdot_BM_N - (np.cross(w_MN_M, r_BM_M))
    drMdt_BM_M_store.append(drMdt_BM_M)

    ## foo position vector of point p
    Nhat3 = np.array([0., 0., 1.])
    Nhat3_M = q_MN_tk_obj.rotate(Nhat3)
    # remove portion of vector normal to eclliptic plane
    eclipticPlane_r_PM_M = r_PM_M - np.dot(r_PM_M,Nhat3_M)*Nhat3_M
    eclipticPlane_r_PM_M_store.append(eclipticPlane_r_PM_M)
    q_NM_tk_obj = q_MN_tk_obj.inverse()
    q_NM_tk_obj.normalize()
    r_PM_N_store.append(q_NM_tk_obj.rotate(eclipticPlane_r_PM_M))





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




# lets plot MCMF vs MCI

r_BM_M_store_array = np.array(r_BM_M_store)        
drMdt_BM_M_store_array = np.array(drMdt_BM_M_store)

# Unpack coordinates
xM = r_BM_M_store_array[:, 0]
yM = r_BM_M_store_array[:, 1]
zM = r_BM_M_store_array[:, 2]

# --- Plot Z vs Y ---
plt.figure(figsize=(8, 6))
plt.plot(xM, yM, linewidth=1.8, label='Spacecraft Trajectory (M-frame)',color='r')

# Optional: plot Moon surface outline (for context)
moon_radius = 1737.4e3  # meters
theta = np.linspace(0, 2*np.pi, 200)
plt.plot(moon_radius * np.cos(theta), moon_radius * np.sin(theta),
         'k--', alpha=0.5, label='Moon Surface')

plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Spacecraft Trajectory in MCMF (Z vs Y)")
plt.axis("equal")
plt.grid(True)
plt.legend()



# grab point p data
r_PM_N_store_array = np.array(r_PM_N_store)
eclipticPlane_r_PM_M_store_array = np.array(eclipticPlane_r_PM_M_store)


plt.figure(figsize=(8,6))
plt.plot(r_PM_N_store_array[:,0], r_PM_N_store_array[:,1],color="r", label="Point P coordinated in N")
plt.plot(eclipticPlane_r_PM_M_store_array[:,0],eclipticPlane_r_PM_M_store_array[:,1],label="P in MCMF",marker='*',color="b")

# Optional: plot Moon surface outline (for context)
moon_radius = 1737.4e3  # meters
theta = np.linspace(0, 2*np.pi, 200)
plt.plot(moon_radius * np.cos(theta), moon_radius * np.sin(theta),
         'k--', alpha=0.5, label='Moon Surface')

plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Point P fixed in MCMF")
plt.axis("equal")
plt.grid(True)
plt.legend()


plt.figure(figsize=(8,6))
plt.plot(r_PM_N_store_array[:,1], r_PM_N_store_array[:,2],color="r", label="Point P coordinated in N")
plt.plot(eclipticPlane_r_PM_M_store_array[:,1],eclipticPlane_r_PM_M_store_array[:,2],label="P in MCMF",marker='*',color="b")

# Optional: plot Moon surface outline (for context)
moon_radius = 1737.4e3  # meters
theta = np.linspace(0, 2*np.pi, 200)
plt.plot(moon_radius * np.cos(theta), moon_radius * np.sin(theta),
         'k--', alpha=0.5, label='Moon Surface')

plt.xlabel("Y [m]")
plt.ylabel("Z [m]")
plt.title("Point P fixed in MCMF")
plt.axis("equal")
plt.grid(True)
plt.legend()

# vector norm should be preserved
plt.figure(figsize=(8,6))
plt.plot(timeData[1:],(np.linalg.norm(r_PM_N_store_array,axis=1)-np.linalg.norm(eclipticPlane_r_PM_M_store_array,axis=1)))
plt.ylabel("magnitude [m]")
plt.xlabel("Time [s]")
plt.title("mag vector")
plt.grid(True)
plt.legend()




plt.show()




    
    










