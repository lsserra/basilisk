import numpy as np
from scipy.linalg import solve_continuous_are

# Define the system matrices (for example)
A = np.zeros((6,6))
A[:3,3:] = np.identity(3)  
B = np.zeros((6,3))
B[3:,:] = np.identity(3)            
Q = np.identity(6)          
'''
Q[:3,:3] = 1e-4 * np.identity(3)   
Q[3:,3:] = 1 * np.identity(3)
'''
R = np.identity(3)                     

# Solve the continuous-time ARE
P = solve_continuous_are(A, B, Q, R)

# opt gains
K = np.linalg.inv(R) @ B.T @ P
K1 = K[:,:3]
K2 = K[:,3:]

# Output the solution
print("K1:")
print(K1)
print("K2:")
print(K2)

# write gains and cost matrices to controller 

# Prepare the updated gain and cost matrices as strings
K1_str = f"self.K1 = np.array({K1.tolist()})"
K2_str = f"self.K2 = np.array({K2.tolist()})"
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
