# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import dm4bem
import matplotlib.pyplot as plt

# temperature nodes
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6', 'θ7', 'θ8', 'θ9', 'θ10', 'θ11', 'θ12', 'θ13', 'θ14', 'θ15', 'θ16', 'θ17',
     'θ18', 'θ19', 'θ20']

# flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17',
     'q18', 'q19', 'q20', 'q21', 'q22', 'q23', 'q24', 'q25', 'q26', 'q27']

A = np.zeros([28, 21])  # n° of branches X n° of nodes
A[0, 0] = 1  # branch 0: -> node 0
A[1, 1], A[2, 1] = 1, -1
A[2, 2], A[3, 2] = 1, -1
A[3, 3], A[4, 3] = 1, -1
A[4, 4], A[5, 4] = 1, -1
A[5, 5], A[6, 5], A[17, 5], A[22, 5],  = 1, -1, 1, 1
A[6, 6], A[7, 6] = 1, -1
A[7, 7], A[8, 7] = 1, -1
A[8, 8], A[9, 8] = 1, -1
A[9, 9], A[10, 9], A[19, 9], A[21, 9], A[25, 9] = 1, 1, 1, 1, 1
A[10, 10], A[11, 10] = -1, 1
A[11, 11], A[12, 11] = -1, 1
A[12, 12], A[13, 12] = -1, 1
A[13, 13], A[14, 13] = -1, 1
A[14, 14], A[15, 14] = -1, 1
A[16, 15], A[17, 15] = 1, -1  # node 15
A[18, 16], A[19, 16] = 1, -1  # node 16
A[26, 17], A[25, 17] = 1, -1  # node 17
A[27, 18], A[26, 18] = 1, -1  # node 18
A[23, 19], A[22, 19] = 1, -1  # node 19
A[24, 20], A[23, 20] = 1, -1  # node 20
#print(A)

G = np.zeros([28, 28])
G[0, 0] = 2200
G[1, 1] = 528
G[2, 2] = 22.46
G[3, 3] = 22
G[4, 4] = 352
G[5, 5] = 2200
G[6, 6] = 1500
G[7, 7] = 240
G[8, 8] = 240
G[9, 9] = 1500
G[10, 10] = 3000
G[11, 11] = 480
G[12, 12] = 30
G[13, 13] = 30.63
G[14, 14] = 720
G[15, 15] = 3000
G[16, 16] = 76.27
G[17, 17] = 76.27
G[18, 18] = 76.27
G[19, 19] = 76.27
G[20, 20] = 50.6
G[21, 21] = 101.2
G[22, 22] = 750
G[23, 23] = 8
G[24, 24] = 8
G[25, 25] = 750
G[26, 26] = 8
G[27, 27] = 8

# rint(G_diag)

# capacitances

CC = 0.9
CI = 0.18
CW = 1.76
CG = 0.84

# densities

DC = 2300
DI = 12
DW = 600
DG = 2500

# Surface
S1 = 45
S2 = 30
S3 = 60
S4 = 15
S5 = 30
S_window = 1

# Thickness
TC = 0.2
TI = 0.15
TW = 0.02
TG = 0.02

C = np.zeros([21, 21])
C[1, 1] = CC * DC * S1 * TC
C[2, 2] = CI * DI * S1 * TI
C[3, 3] = CW * DW * S1 * TW
C[7, 7] = CW * DW * S2 * TW
C[11, 11] = CW * DW * S3 * TW
C[12, 12] = CI * DI * S3 * TI
C[13, 13] = CC * DC * S3 * TC
C[15, 15] = CG * DG * S_window * TG
C[16, 16] = CG * DG * S_window * TG
C[17, 17] = S5 * CI * TI * DI
C[19, 19] = S4 * CI * TI * DI
# print (C_diag)


To = 15
Tf = 5  # cooling system
Ti_sp = 19
b = np.zeros(28)
b[0] = b[15] = b[16] = b[18] = To
b[20] = b[21] = Ti_sp
b[24] = b[27] = Tf
# print (b)
f = np.zeros(21)
f[[0,4,6, 8, 10,14,15,16,17,19 ]] = 1

y = np.zeros(21)  # nodes
y[[5]] = y[[9]] = 1  # nodes (temperatures) of interest
pd.DataFrame(y, index=θ)

[As, Bs, Cs, Ds] = dm4bem.tc2ss(A, G, b, C, f, y)
print('As = \n', As, '\n')
print('Bs = \n', Bs, '\n')
print('Cs = \n', Cs, '\n')
print('Ds = \n', Ds, '\n')
print(np.shape(Bs))
#print(np.shape(u))

θ = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b+f)
print(f'theta = {np.around(θ, 2)} °C')

# steady state analysis
bT = np.array([15,15,15,15,5,5,19,19])     # [To, To, To, Tisp]
fQ = np.array([0, 0, 0, 0, 0, 0, 0, 0,0,0])                         # [Φo, Φi, Qa, Φa]
u = np.hstack([bT, fQ])
print(f'u = {u}')
print(np.shape(u))

yss = (-Cs @ np.linalg.inv(As) @ Bs + Ds) @ u
print(f'yss = {np.around(yss, 2)} °C')
print(f'Max error between DAE and state-space: \
{max(abs(θ[5] - yss)):.2e} °C')

#dynamic simulation

λ = np.linalg.eig(As)[0]    # eigenvalues of matrix As

print('Time constants: \n', -1 / λ, 's \n')
print('2 x Time constants: \n', -2 / λ, 's \n')
dtmax = 2 * min(-1. / λ)
print(f'Maximum time step: {dtmax:.2f} s = {dtmax / 60:.10f} min')
# time step
dt = (dtmax / 60) * 20   # s
print(f'dt = {dt} s = {dt / 60:.10f} min')

# settling time
time_const = np.array([int(x) for x in sorted(-1 / λ)])
print('4 * Time constants: \n', 4 * time_const, 's \n')

t_settle = 4 * max(-1 / λ)
print(f'Settling time: \
{t_settle:.0f} s = \
{t_settle / 60:.1f} min = \
{t_settle / (3600):.2f} h = \
{t_settle / (3600 * 24):.2f} days')

# Step response
# -------------
# Find the next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_settle / 3600) * 3600
n = int(np.floor(duration / dt))    # number of time steps
t = np.arange(0, n * dt, dt)        # time vector for n time steps

print(f'Duration = {duration} s')
print(f'Number of time steps = {n}')
# pd.DataFrame(t, columns=['time'])

u = np.zeros([18, n])                # u = [To To To Tisp Φo Φi Qa Φa]
u[0:5, :] = 15 * np.ones([5, n])    # To = 10 for n time steps
u[5:8, :] = 19 * np.ones([3, n])      # Tisp = 20 for n time steps


pd.DataFrame(u)

n_s = As.shape[0]  # number of state variables
θ_exp = np.zeros([n_s, t.shape[0]])  # explicit Euler in time t
θ_imp = np.zeros([n_s, t.shape[0]])  # implicit Euler in time t

I = np.eye(n_s)  # identity matrix

for k in range(n - 1):
    θ_exp[:, k + 1] = (I + dt * As) @ θ_exp[:, k] + dt * Bs @ u[:, k]
    θ_imp[:, k + 1] = np.linalg.inv(I - dt * As) @ (θ_imp[:, k] + dt * Bs @ u[:, k])

print(θ_exp)

y_exp = Cs @ θ_exp + Ds @  u
y_imp = Cs @ θ_imp + Ds @  u
#print('y_exp',y_exp)
print(np.shape(Cs))
print(np.shape(θ_exp))
print(np.shape(Ds))
print(np.shape(u))
print(np.shape(y_exp))


fig, ax = plt.subplots()
ax.plot(t/3600, y_exp.T[:,0], t / 3600, y_imp.T[:,0])
ax.set(xlabel='Time, $t$ / h',
       ylabel='Temperatue, $θ_i$ / °C',
       title='Step input: outdoor temperature $T_o$')
ax.legend(['Explicit', 'Implicit'])
ax.grid()
plt.show()
print('y_exp=', y_exp.T[:,0])

print('Steady-state indoor temperature obtained with:')
print(f'- DAE model: Room 1: {float(θ[11]):.4f}, Room 2: {float(θ[10]):.4f} °C')
print(f'- state-space model: Room 1: {float(yss[0]):.4f}, Room 2: {float(yss[1]):.4f} °C')
print(f'- steady-state response to step input: Room1: {float(y_exp[0, -2]):.4f}, Room2: {float(y_exp[1, -2]):.4f} °C')




