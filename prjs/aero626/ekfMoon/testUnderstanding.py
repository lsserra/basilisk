import pickle
import numpy as np
from scipy.integrate import solve_ivp


from matplotlib import pyplot as plt



with open("data/MoonCentralBody_MoonGrav.pkl", "rb") as f:
    sim_data = pickle.load(f)

timeData = sim_data["time"] * 1e-9 # to seconds
sc_pos   = sim_data["sc_pos"] / 1000 # to km
sc_vel   = sim_data["sc_vel"] / 1000
moon_pos = sim_data["moon_pos"] / 1000
moon_vel = sim_data["moon_vel"] / 1000



# ICs
r_BN_N0 = sc_pos[0,:]
rdot_BN_N0 = sc_vel[0,:]
t0 = timeData[0]
tf = timeData[-1]
x0 = np.array([r_BN_N0,rdot_BN_N0]).flatten()



def MCIPoseDynamics(t,x):

    MU_MOON = 4902.799 # km^3/s^3

    r_BM_M = x[:3]
    Mdrdt_BM_M = x[3:]
    r_BM_M_norm = np.linalg.norm(r_BM_M)

    fgrav = -MU_MOON*np.array([0.,0.,1.])/r_BM_M_norm**3
    dr2dt2_BM_M_M = fgrav 
    
    xdot = np.hstack([Mdrdt_BM_M,dr2dt2_BM_M_M])

    return xdot



sol = solve_ivp(
    fun=lambda t, x_aug: MCIPoseDynamics(
        t,
        x_aug),
    t_span=[t0, tf],
    t_eval=timeData,
    y0=x0,
    method='RK45',
    rtol=1e-9,
    atol=1e-9
    )


solution = sol.y
solTime = sol.t

r_BN_N = solution[:3,:]
rdot_BN_N = solution[3:,:]


plt.figure()
plt.plot(r_BN_N[0,:],r_BN_N[1,:])
plt.show()

data_bundle = {
    "time": solTime,
    "r_BN_N": r_BN_N,
    "rdot_BN_N": rdot_BN_N,
}
with open("data/separateInertialSolution.pkl", "wb") as f:
    pickle.dump(data_bundle, f)
