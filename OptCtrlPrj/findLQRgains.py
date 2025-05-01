import numpy as np
from scipy.linalg import solve_continuous_are

def is_observable(A, Q):
    n = A.shape[0]
    sqrt_Q = np.linalg.cholesky(Q, upper=False)
    O = sqrt_Q
    for i in range(1, n):
        O = np.vstack((O, sqrt_Q @ np.linalg.matrix_power(A, i)))
    rank = np.linalg.matrix_rank(O)
    return rank 


# Define the system matrices (for example)
A = np.zeros((6,6))
A[:3,3:] = np.identity(3)  
B = np.zeros((6,3))
B[3:,:] = np.identity(3)            
Q = 1e-4 * np.identity(6)          
'''
Q[:3,:3] = 1 * np.identity(3)   
Q[3:,3:] = 1 * np.identity(3)
'''
R = 1 * np.identity(3)                     


# check observability of (A,sqrt(Q))
rankObsMat = is_observable(A,Q)
print("Rank (A,sqrt(Q) obs mat: ")
print(rankObsMat)


# Solve the continuous-time ARE
P = solve_continuous_are(A, B, Q, R)
print("Sinf:")
print(P)

# opt gains
K = np.linalg.inv(R) @ B.T @ P
K1 = K[:,:3]
K2 = K[:,3:]

# Output the solution
print("K1:")
print(K1)
print("K2:")
print(K2)


## pole placement method
import numpy as np
from scipy.signal import place_poles

# Natural frequency and damping ratio
wn = 0.1  # rad/s
damp = 0.9

# Compute poles (real because damping ratio > 1)
real_part = -damp * wn
imag_part = wn * np.sqrt(1 - damp**2)

s1 = complex(real_part, imag_part)
s2 = complex(real_part, -imag_part)
# Create a list of desired poles (6 total, repeated)
p = [s1, s2, s1, s2, s1, s2]


# Compute state feedback gain K
result = place_poles(A, B, p)
K_pol = result.gain_matrix

print("Desired poles:", p)
K1_pol = K_pol[:,:3]
K2_pol = K_pol[:,3:]

# Output the solution
print("K1 pole place:")
print(K1_pol)
print("K2 pole place:")
print(K2_pol)
# write gains and cost matrices to controller 

# Prepare the updated gain and cost matrices as strings
K1_str = f"self.K1 = np.array({K1.tolist()})"
K2_str = f"self.K2 = np.array({K2.tolist()})"
#K1_str = f"self.K1 = np.array({K1.tolist()})"
#K2_str = f"self.K2 = np.array({K2.tolist()})"
Q_str = f"self.Q = np.diag({Q.diagonal().tolist()})"
R_str = f"self.R = np.diag({R.diagonal().tolist()})"

# Path to the file
file_path = 'OptCtrlPrj/scenarioAttitudeErrQuatFeedback.py'

# Read and replace the lines in the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Find and replace the corresponding lines
for i, line in enumerate(lines):
    if 'self.K1 =' in line:
        lines[i] = '        ' +K1_str + '\n'
    elif 'self.K2 =' in line:
        lines[i] = '        ' + K2_str + '\n'
    elif 'self.Q =' in line:
        lines[i] = '        ' + Q_str + '\n'
    elif 'self.R =' in line:
        lines[i] = '        ' + R_str + '\n'

# Write the updated lines back to the file
with open(file_path, 'w') as file:
    file.writelines(lines)


# analyze closed loop dynamics
A_0 = (A-B@K)
eig_vals = np.linalg.eigvals(A_0)
print("Closed-loop eigenvalues:")
print(eig_vals)


