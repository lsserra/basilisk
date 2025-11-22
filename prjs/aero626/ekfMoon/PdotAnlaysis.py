import sympy as sp

# === Define symbols ===
r1, r2, r3 = sp.symbols("r_1 r_2 r_3", real=True)
w1, w2, w3 = sp.symbols("w_1 w_2 w_3", real=True)
MU = sp.symbols("MU", real=True, positive=True)

r = sp.Matrix([r1, r2, r3])
rnorm = sp.sqrt(r1**2 + r2**2 + r3**2)

# Skew(omega)
omega_skew = sp.Matrix([
    [0,   -w3,  w2],
    [w3,   0,  -w1],
    [-w2,  w1,  0]
])

# === Translational Jacobian F_t ===

I3 = sp.eye(3)

F11 = sp.zeros(3)
F12 = sp.eye(3)
F22 = -2 * omega_skew

# Gravitational curvature term
F21 = -MU * (I3/rnorm**3 - 3*(r*r.T)/rnorm**5)

Ft = sp.Matrix.vstack(
        sp.Matrix.hstack(F11, F12),
        sp.Matrix.hstack(F21, F22)
)

# === Attitude Jacobian F_a (MEKF small-angle) ===
Fa = sp.Matrix.vstack(
        sp.Matrix.hstack(-omega_skew, -sp.eye(3)),
        sp.Matrix.hstack(sp.zeros(3), sp.zeros(3))
)

# === Full F (12x12) ===
F = sp.Matrix.zeros(12,12)
F[:6,:6] = Ft
F[6:,6:] = Fa


# === Covariance P (12x12) with subscripts P_ij ===
P = sp.Matrix(12,12, lambda i,j: sp.symbols(f"P_{i+1}_{j+1}", real=True))

# Process noise Q (general 12x12)
Q = sp.Matrix(12,12, lambda i,j: sp.symbols(f"Q_{i+1}_{j+1}", real=True))


# === Covariance dynamics Ṗ = F P + P Fᵀ + Q ===
Pdot = F*P + P*F.T + Q

# === Print some example components ===
print("Example symbolic elements of Pdot:")
print("Pdot_1_1 =", Pdot[0,0])
print("Pdot_1_7 =", Pdot[0,6])
print("Pdot_1_2 =", Pdot[0,1])
print("Pdot_4_7 =", Pdot[3,6])
print("Pdot_10_10 =", Pdot[9,9])

# If you want the full 12x12 printed:
# sp.pretty_print(Pdot)
