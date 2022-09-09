import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as scio
from scipy import optimize
import math
from sympy import *
from scipy import linalg

def van_der_pol_solve(N, N_Traj):
    # 龙格库塔
    m = 1
    n = 2
    h = 0.05
    '''
    def f(t, x, u):
        x1 = x[0, :];
        x2 = x[1, :];
        return 2 * x2, -0.8 * x1 + 2 * x2 - 10 * x1 ** 2 * x2 + u
    '''

    f = lambda t, x, u: np.array([2.0 * x[1, :], 2.0 * x[1, :] - 10.0 * x[0, :] ** 2.0 * x[1, :] - 0.8 * x[0, :] + u])
    k1 = lambda t, x, u: np.array(f(t, x, u))
    k2 = lambda t, x, u: np.array(f(t + h / 2.0, x + 0.5 * h * k1(t, x, u), u))
    k3 = lambda t, x, u: np.array(f(t + h / 2.0, x + 0.5 * h * k2(t, x, u), u))
    k4 = lambda t, x, u: np.array(f(t + h, x + h * k3(t, x, u), u))
    f_update = lambda t, x, u: np.array( x + (h / 6.0) * (k1(t, x, u) + 2.0 * k2(t, x, u) + 2.0 * k3(t, x, u) + k4(t, x, u)))
    u0 = 4.0 * np.random.rand(N, N_Traj) - 2.0
    #u0 = np.zeros((0, N_Traj))
    #u1 = 4.0 * np.random.rand(1, N_Traj) - 2.0
    #for i in range (0, N):
    #    u0 = np.concatenate((u0, u1), axis=0)
    compareU = scio.loadmat('compareU.mat')
    #UU = compareU['U']
    #u0[0 : 100][0] = UU
    x0 = 4.0 * np.random.rand(n, N_Traj) - 2.0
    #x0[0][0] = 1.0
    #x0[1][0] = 1.0
    #x0[:, 0] = np.array([0.0, 0.07])
    x = x0;
    X = np.zeros((n, 0))
    Y = np.zeros((n, 0))
    U = np.zeros((m, 0))

    for i in range (0, N):
        nowu = np.array(u0[i, :])
        x_next = np.array(f_update(0, x, nowu))
        x_next = np.reshape(x_next, (n, N_Traj))
        X = np.concatenate((X, x), axis=1)
        Y = np.concatenate((Y, x_next), axis=1)
        nowu = nowu.reshape((m, N_Traj))
        U = np.concatenate((U, nowu), axis=1)
        x = x_next
    return X,Y,U

def duffing_RK_solve(N, N_Traj):
    # 龙格库塔
    m = 1
    n = 2
    h = 0.05
    '''
    def f(t, x, u):
        x1 = x[0, :];
        x2 = x[1, :];
        return 2 * x2, -0.8 * x1 + 2 * x2 - 10 * x1 ** 2 * x2 + u
    '''
    #duffing is [x2, -0.5 * x2 + x1 - x1**3]
    f = lambda t, x, u: np.array([x[1, :], -0.5 * x[1, :] + x[0, :] - x[0, :] ** 3.0 + u])
    k1 = lambda t, x, u: np.array(f(t, x, u))
    k2 = lambda t, x, u: np.array(f(t + h / 2.0, x + 0.5 * h * k1(t, x, u), u))
    k3 = lambda t, x, u: np.array(f(t + h / 2.0, x + 0.5 * h * k2(t, x, u), u))
    k4 = lambda t, x, u: np.array(f(t + h, x + h * k3(t, x, u), u))
    f_update = lambda t, x, u: np.array( x + (h / 6.0) * (k1(t, x, u) + 2.0 * k2(t, x, u) + 2.0 * k3(t, x, u) + k4(t, x, u)))
    u0 = 4.0 * np.random.rand(N, N_Traj) - 2.0
    #u0 = np.zeros((0, N_Traj))
    #u1 = 4.0 * np.random.rand(1, N_Traj) - 2.0
    #for i in range (0, N):
    #    u0 = np.concatenate((u0, u1), axis=0)
    #compareU = scio.loadmat('compareU.mat')
    #UU = compareU['U']
    #u0[0: 100][0] = UU
    x0 = 4.0 * np.random.rand(n, N_Traj) - 2.0
    #x0[0][0] = -2.0
    #x0[1][0] = -1.5
    # x0[:, 0] = np.array([0.0, 0.07])
    x = x0;
    X = np.zeros((n, 0))
    Y = np.zeros((n, 0))
    U = np.zeros((m, 0))
    for i in range (0, N):
        nowu = np.array(u0[i, :])
        x_next = np.array(f_update(0, x, nowu))
        x_next = np.reshape(x_next, (n, N_Traj))
        X = np.concatenate((X, x), axis=1)
        Y = np.concatenate((Y, x_next), axis=1)
        nowu = nowu.reshape((m, N_Traj))
        U = np.concatenate((U, nowu), axis=1)
        x = x_next
    return X,Y,U


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.Encoder = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 8),  # compress to 3 features which can be visualized in plt
        )
        self.Decoder = nn.Sequential(
            nn.Linear(8, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            #nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.Encoder(x)
        decoded = self.Decoder(encoded)
        return encoded, decoded

data = scio.loadmat('DeepLearning_KoopmanControl_Approach3_Vanderpol_13.mat')

torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.cuda.is_available())
#net = AutoEncoder()
#net = torch.load('AutoEncoder_20220324_5.pkl')
# 用7能出之前的结果
# 用20220414_1能无静差 Ts=0.05
# 4 也还行
# net = torch.load('AutoEncoder_20220414_4.pkl')
# 试试看Duffing system
net = torch.load('AutoEncoder_20220418_duffing_2.pkl')
optimizer = optim.Adam(net.parameters(), lr=0.001)
eta = 0.5

Nlift = 8

n = 2
m = 1

criterion = nn.MSELoss(reduction='sum')

N = 100
N_Traj = 100

# X,Y,U = van_der_pol_solve(N, N_Traj)

X,Y,U = duffing_RK_solve(N, N_Traj)

print(X.shape)
print(Y.shape)
print(U.shape)

# prepare for the data
X_temp = np.zeros((n, 0))
Y_temp = np.zeros((n, 0))
U_temp = np.zeros((m, 0))

for i in range (0, N_Traj):
    for j in range (0, N):

        x = X[:, i + j * N_Traj]
        x = np.reshape(x, (n, 1))
        y = Y[:, i + j * N_Traj]
        y = np.reshape(y, (n, 1))
        u = U[:, i + j * N_Traj]
        u = np.reshape(u, (m, 1))

        X_temp = np.concatenate((X_temp, x), axis=1)
        Y_temp = np.concatenate((Y_temp, y), axis=1)
        U_temp = np.concatenate((U_temp, u), axis=1)

X = X_temp
Y = Y_temp
U = U_temp

#X = X.T
#X_prime = Y.T
#U = U.T

A_init = np.random.rand(Nlift, Nlift)
B_init = np.random.rand(Nlift, m)

A = A_init
B = B_init

# 数据预处理
inputs_x = torch.tensor(X)
inputs_y = torch.tensor(Y)
inputs_u = torch.tensor(U)
A_init = torch.tensor(A_init)
B_init = torch.tensor(B_init)


i = 0
j = 0

pred_horizon = 30
epoch = 20
batch = N
batch_size = N_Traj

PHIX = torch.tensor([])
PHIY = torch.tensor([])


alpha_1 = 1.0
alpha_2 = 10.0
alpha_3 = 50.0
alpha_4 = 1E-6
alpha_5 = 1E-6

# testing
Loss_lin = 0
Loss_pred = 0
Loss_rec = 0
Loss = 0

pred_horizon = 30
epoch = 20
batch = N
batch_size = N_Traj


PHIX = torch.tensor([])
PHIY = torch.tensor([])

for items in range(0, N * N_Traj):
    g_x = net.Encoder(inputs_x[:, items])
    g_x = torch.reshape(g_x, (len(g_x), 1))
    g_x_prime = net.Encoder(inputs_y[:, items])
    g_x_prime = torch.reshape(g_x_prime, (len(g_x_prime), 1))

    PHIX = torch.cat([PHIX, g_x], dim=1)
    PHIY = torch.cat([PHIY, g_x_prime], dim=1)

    # nowu = inputs_u[items, :]
    # nowu = torch.reshape(nowu, (m, m))
    persent = (items + 1) / (N * N_Traj) * 100
    print("\rCalculating dictionary of X and Y ： {:3}% ".format(persent), end="")


PHIY_U = torch.cat([PHIX, inputs_u], dim=0)
K_hat = PHIY @ torch.pinverse(PHIY_U)
A_hat = K_hat[0: Nlift, 0: Nlift]
B_hat = K_hat[0: Nlift, Nlift]
B_hat = torch.reshape(B_hat, (len(B), m))
#A = eta * A_hat + (1 - eta) * A_init
#B = eta * B_hat + (1 - eta) * B_init
A = A_hat
B = B_hat
#PHIX = PHIX.detach().numpy()
C = X @ np.linalg.pinv(PHIX.detach().numpy())

for j in range(0, batch_size):
    s_lin = 0
    s_pred = 0

    k = i * batch + j #k为当前时刻
    if N * N_Traj - k <= pred_horizon:
        break
    if batch - k <= pred_horizon:
        break
    phix = PHIX[:, k]
    x_rec = net.Decoder(phix)
    x_rec = torch.reshape(x_rec, (len(x_rec), 1))

    x_k = inputs_x[:, k]
    x_k = torch.reshape(x_k, (len(x_k), 1))
    Loss_rec = criterion(x_rec, x_k)

    for p in range(1, pred_horizon + 1):
        sum_ABu = torch.tensor(0)
        for s in range (1, p + 1):
            tempu = inputs_u[:, k + s - 1]
            u = torch.reshape(tempu, (m, m))
            sum_ABu = sum_ABu + torch.matrix_power(A, p - s) @ B @ u
        phix_k = torch.reshape(phix, (len(phix), 1))
        phix_pred = torch.matrix_power(A, p) @ phix_k + sum_ABu # A^p @ phix + \sum^s_s=1 A^(p - s) @ B @ u
        phix_p = PHIX[:, k + p]
        phix_pred = torch.squeeze(phix_pred) # 解除warning
        #phix_p = torch.reshape(phix_p, (1 , len(phix_p)))
        #phix_pred = torch.reshape(phix_pred, (1, len(phix_pred)))
        Loss_lin = Loss_lin + criterion(phix_pred, phix_p)
        #Loss_lin = Loss_lin + s_lin

        x_p = inputs_x[:, k + p]
        #phix_pred = torch.reshape(phix_pred, (1, len(phix_pred)))
        x_rec_p = net.Decoder(phix_pred)
        Loss_pred = Loss_pred + criterion(x_p, x_rec_p)
        #Loss_pred = Loss_pred + s_pred

    Loss_lin = Loss_lin / pred_horizon
    Loss_pred = Loss_pred / pred_horizon

    weight = 0
    #Loss_lin = s_lin
    #Loss_pred = s_pred
    for param in net.parameters():
        weight = weight + torch.sum(torch.abs(param))
    #if eps <= 5:
    #    Loss = Loss + alpha_1 * Loss_rec

    Loss = Loss + alpha_1 * Loss_rec + alpha_2 * Loss_lin + alpha_3 * Loss_pred + alpha_4 * weight
    print("\rProcessing batch number ： %d / %d" % (j + 1, N_Traj - pred_horizon), end="")

Loss = Loss / batch_size
print("\nLoss_rec", Loss_rec)
print("\nLoss_lin", Loss_lin)
print("\nLoss_pred", Loss_pred)
print("\nLoss", Loss)
#optimizer.zero_grad()
#Loss.backward()
#optimizer.step()
#print("\nWeight updated!")
'''
A = data['A']
B = data['B']
A = torch.tensor(A)
B = torch.tensor(B)
'''



# Dynamic system
h = 0.05
# f = lambda t, x, u: np.array([2.0 * x[1, :], 2.0 * x[1, :] - 10.0 * x[0, :] ** 2.0 * x[1, :] - 0.8 * x[0, :] + u])
# -0.5 * x2 + x1 - x1**3
f = lambda t, x, u: np.array([x[1, :], -0.5 * x[1, :] + x[0, :] - x[0, :] ** 3.0 + u])

k1 = lambda t, x, u: np.array(f(t, x, u))
k2 = lambda t, x, u: np.array(f(t + h / 2.0, x + 0.5 * h * k1(t, x, u), u))
k3 = lambda t, x, u: np.array(f(t + h / 2.0, x + 0.5 * h * k2(t, x, u), u))
k4 = lambda t, x, u: np.array(f(t + h, x + h * k3(t, x, u), u))
f_update = lambda t, x, u: np.array(x + (h / 6.0) * (k1(t, x, u) + 2.0 * k2(t, x, u) + 2.0 * k3(t, x, u) + k4(t, x, u)))



test_X = torch.tensor([])
test_Y = torch.tensor([])
plotTime = 500
Uplot = U[0, 0:plotTime];
for i in range (0, plotTime):
    x = inputs_x[:, i]
    x_Lift, x_Rec = net(x)
    x_Lift = torch.reshape(x_Lift, (len(x_Lift), 1))
    #x_Rec = torch.tensor(C) @ x_Lift
    x = torch.reshape(x, (len(x), 1))
    x_Rec = torch.reshape(x_Rec, (len(x_Rec), 1))

    test_X = torch.cat([test_X, x], dim = 1)
    test_Y = torch.cat([test_Y, x_Rec], dim = 1)

test_X = test_X.detach().numpy()
test_Y = test_Y.detach().numpy()
tspan = np.arange(plotTime) * h
tspan = np.array(tspan)


decoder_X = torch.tensor([])
marker_X = torch.tensor([])
marker_T = []
real_X = inputs_x[:, 0 : plotTime]
x = inputs_x[:, 0]
tspan_pred = np.arange(plotTime) * h
tspan_pred = np.array(tspan_pred)
for i in range (0, plotTime):
    if i % 15 == 0:
        phix = net.Encoder(inputs_x[:, i])
        phix = torch.reshape(phix, (1, len(phix)))
        phix = torch.squeeze(phix)
        marker_x = net.Decoder(phix)
        marker_x = torch.reshape(marker_x, (len(marker_x), 1))
        marker_X = torch.cat([marker_X, marker_x], dim = 1)
        marker_T.append(i * h)

    u = inputs_u[:, i]
    #u = torch.tensor(0.2)
    u = torch.reshape(u, (m, m))
    phix = torch.squeeze(phix)
    decoder_x = net.Decoder(phix)
    decoder_x = torch.reshape(decoder_x, (len(decoder_x), 1))

    #decoder_x = np.reshape(decoder_x, (len(decoder_x), 1))
    phix = torch.reshape(phix, (len(phix), 1))
    #decoder_x = torch.tensor(C) @ phix
    decoder_X = torch.cat([decoder_X, decoder_x], dim = 1)
    phix = A @ phix + B @ u

decoder_X = decoder_X.detach().numpy()
real_X = real_X.detach().numpy()
marker_X = marker_X.detach().numpy()

plt.figure()
#plt.subplot(311)
plt.title('$x_1$ and $x_1$ reconstruction')

plt.plot(tspan, test_X[0, :], label = "$x_1$ trajectory ")
plt.plot(tspan, test_Y[0, :],marker='.', label = "reconstructed $x_1$ trajectory ")
plt.xlabel('$t/s$')
plt.ylabel('$x_1$')
plt.legend()
#plt.subplot(312)
plt.figure()

plt.title('$x_2$ and $x_2$ reconstruction')
plt.plot(tspan, test_X[1, :], label = "$x_2$ trajectory ")
plt.plot(tspan, test_Y[1, :],marker='.', label = "reconstructed $x_2$ trajectory ")
plt.xlabel('$t/s$')
plt.ylabel('$x_2$')
plt.legend()
#plt.subplot(313)
plt.figure()

plt.plot(tspan, Uplot)
plt.title('control inputs')
plt.xlabel('$t/s$')
plt.ylabel('$u$')
plt.tight_layout()



plt.figure()
#plt.subplot(311)
plt.title('$x_1$ and $x_1$ linear reconstruction')
plt.plot(tspan_pred, real_X[0, :], label = "$x_1$ trajectory ")
plt.plot(tspan_pred, decoder_X[0, :],marker='.', label = "linear predict $x_1$ trajectory ")
# plot marker
plt.scatter(marker_T, marker_X[0, :], marker = 'o', c = 'r')
plt.xlabel('$t/s$')
plt.ylabel('$x_1$')
plt.legend()
plt.figure()

#plt.subplot(312)
plt.title('$x_2$ and $x_2$ linear reconstruction')
plt.plot(tspan_pred, real_X[1, :], label = "$x_2$ trajectory ")
plt.plot(tspan_pred, decoder_X[1, :],marker='.', label = "linear predict $x_2$ trajectory ")
plt.scatter(marker_T, marker_X[1, :], marker = 'o', c = 'r')

plt.xlabel('$t/s$')
plt.ylabel('$x_2$')
plt.legend()
plt.figure()

#plt.subplot(313)
plt.plot(tspan, Uplot)
plt.title('control inputs')
plt.xlabel('$t/s$')
plt.ylabel('$u$')

plt.tight_layout()

plt.show()
# 家里主机训练的编号有误 以笔记本为准




# MPC Close-Loop Control
def costFunction(uSequence, r , AB, C, x0, pastu, Np, Nc, d):
    logY = np.zeros((2, 0))
    uSequence = np.array(uSequence)
    i = 0
    n_x = x0.shape[0]
    for u in uSequence:
        u=np.reshape(u,(1,1))
        x_u = np.concatenate([x0, u])
        x_u = np.reshape(x_u, (n_x + 1, 1))
        x0 = AB @ x_u + d
        y = C @ x0
        y = np.reshape(y, (2, 1))
        y = y - np.array(r[:, i]).reshape(2, 1)
        logY = np.concatenate((logY, y), axis=1)
        i = i + 1

    for i in range (Nc, Np):
        u = np.array(uSequence[-1]).reshape((1,1))
        x_u = np.concatenate([x0, u])
        x_u = np.reshape(x_u, (n_x + 1, 1))
        x0 = AB @ x_u + d
        y = C @ x0
        y = np.reshape(y, (2, 1))
        y = y - np.array(r[:, i]).reshape(2, 1)

        logY = np.concatenate((logY, y), axis=1)
    #print('\nerror:', y)
    #    logY+=[y[0][1]]
   # logY = np.array(logY)
    #    print(logY-r)
    #P = np.identity(8)
    delta_u = uSequence[1:] - uSequence[:-1]
    x0 = np.array(x0)
    x0 = np.reshape(x0, (len(x0), 1))
    x = x0.transpose()
    cost = 0.1 * np.sum(np.square(delta_u)+uSequence[0] - pastu) + 1000 * np.sum(np.square(logY)) + 0.01 * np.sum(np.square(uSequence))# + x @ P @ x0
    return cost
#
def solve_DARE(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 500
    eps = 0.01

    for i in range(maxiter):
        Xn = A.T @ X @ A - A.T @ X @ B @ np.linalg.pinv(R + B.T @ X @ B) @ B.T @ X @ A + Q
        if (abs(Xn - X)).max()  < eps:
            X = Xn
            break
        X = Xn

    return Xn

def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_DARE(A, B, Q, R)

    # compute the LQR gain
    K = np.linalg.pinv(B.T @ X @ B + R) @ (B.T @ X @ A)

    return K

def Jacobian(v_str, f_list):
    vars = symbols(v_str)
    f = sympify(f_list)
    J = zeros(len(f),len(vars))
    for i, fi in enumerate(f):
        for j, s in enumerate(vars):
            J[i,j] = diff(fi, s)
    return J

#A = A_hat
#B = B_hat
#C = np.identity(8)
print("\nA-eigenvalue : ", torch.linalg.eig(A))

maxStep = 5000
tspan = h * np.arange(0, maxStep)

MPCHorizon=15
ControlHorizon = 8
pastRes=np.zeros((ControlHorizon));
pastRes_loc=np.zeros((ControlHorizon));
bounds=[(-3,3) for i in range(0,ControlHorizon)]
nx, nu = B.shape

# This is the testing problem for MPC controller. Now its more silent :)

logUloc = np.zeros((1, 0))
logU = np.zeros((1, 0))
logYR = np.zeros((maxStep, 0))
logX = np.zeros((2, 0))
logXloc = np.zeros((2, 0))
logXlift = np.zeros((8, 0))
logR = np.zeros((2, 0))
init = np.array([-2.0, -2.0])

x0 = torch.tensor(init)

#x0 = torch.zeros((1, 2))
x_origin = np.array(init).reshape((2,1))
x_loc = np.array(init).reshape((2,1))
A = A.detach().numpy()
B = B.detach().numpy()

ctrb = np.zeros((Nlift, 0))

for i in range (A.shape[0]):
    ctrb_part = np.linalg.matrix_power(A, i) @ B
    ctrb = np.concatenate([ctrb, ctrb_part], axis=1)

Rank = np.linalg.matrix_rank(ctrb)

Q = 10*np.identity(Nlift)
R = 0.01
K_gain = dlqr(A,B,Q,R)
K_gain = np.reshape(K_gain, (1, 8))
P = solve_DARE(A, B, Q, R)
print(P)

x0 = torch.reshape(x0, (1, 2))
check_goal = net.Encoder(x0)
print(x0)
#xlift = x0.detach().numpy()
zero_bias = np.array([-0.0440, -0.2136,  0.0012,  0.0206,  0.1100, -0.1156,  0.1510,  0.1441])
zero_bias = np.reshape(zero_bias, (8, 1))
x_fixed = np.array([ 0.9289,  0.4656,  1.5074, -0.6017,  0.3931,  0.9058,  0.4351,  1.8487])
x_fixed = np.reshape(x_fixed, (8, 1))
LQR = False
#x0 = net.Encoder(x0)
#xlift = x0.detach().numpy()
#default is MPC
u = np.array(0.0).reshape((1,1))
u_loc = np.array(0.0).reshape((1,1))
# -0.5 * x2 + x1 - x1**3
# van der pol

# K_loc = Jacobian('x1 x2 uu', ['2.0 * x2', '-0.8 * x1 + 2.0 * x2 - 10.0 * x1 ** 2.0 * x2 + uu'])
# x2, -0.5 * x2 + x1 - x1**3.0
K_loc = Jacobian('x1 x2 uu', ['x2', '-0.5 * x2 + x1 - x1**3.0 + uu'])


# 局部线性化操作
A_Jacobian = K_loc[0 : 2, 0 : 2]
B_Jacobian = K_loc[0 : 2 , -1]

x1, x2, uu = symbols("x1 x2 uu")
# f1 = 2.0 * x2
# f2 = -0.8 * x1 + 2.0 * x2 - 10.0 * x1 ** 2.0 * x2 + uu
f1 = x2
f2 = -0.5 * x2 + 1.0 * x1 - 1.0 * x1 ** 3.0 + uu

f_ud = Matrix([[f1], [f2]])


# LTV-Parameters
Y_EX = np.zeros((Nlift, 0))
X_EX = np.zeros((Nlift, 0))
U_EX = np.zeros((m, 0))
XU_EX = np.zeros((m + Nlift, 0))
T_EX = []
A_error = []
B_error = []

q = 0.7
M = N * N_Traj
npPHIY = np.array(PHIY.detach())
npPHIY_U = np.array(PHIY_U.detach())
K_A = (npPHIY @ npPHIY_U.transpose()) / M
K_G = (npPHIY_U @ npPHIY_U.transpose()) / M

# Closed loop simulation
for i in range(0,maxStep):

    if i < maxStep / 2.0:
        r1=[ 0.7*np.array([[np.sin(j/(20+0.01*j))]])+.7 for j in range(i,i+MPCHorizon)]
    else:
        r1 = [np.array(1 * np.power(-1, math.ceil(i / 300)) + 0.01 * np.random.rand(1,1) - 0.01 / 2.0) for j in range(i, i + MPCHorizon)]
    # r1 = [np.array(1 * np.power(-1, math.ceil(i / 1000))) for j in range(i, i + MPCHorizon)]

    r1 = [0.7 * np.array([[np.sin(j / (20 + 0.01 * j))]])+.7  for j in range(i, i + MPCHorizon)]
    #r1 = [1 * np.array([[np.sin(0.01 * j)]]) for j in range(i, i + MPCHorizon)]
    #r1 = [0.5 * np.array([[np.cos(0.007 * j )]]) + 1.2 * np.array([[np.sin(0.002 * j )]]) for j in range(i, i + MPCHorizon)]
    r1 = np.array(r1).reshape((1, MPCHorizon))
    #r2 = [0.1 * np.array([[np.cos(0.002 * j )]]) for j in range(i, i + MPCHorizon)]
    r2 = np.zeros((1, MPCHorizon))
    r2 = np.array(r2).reshape((1, MPCHorizon))
    r = np.concatenate([r1, r2], axis=0)
    print("\nr : ",r[:, 0])
    logR = np.concatenate([logR, np.array(r[:, 0]).reshape(2, 1)], axis=1)
    #r = np.ones((1, 8))
    x0 = net.Encoder(x0)
    xlift = x0.detach().numpy()
    xlift = np.reshape(xlift, (8, 1))
    logXlift = np.concatenate((logXlift, xlift), axis=1)

    #print("\nxlift : ", xlift)
    x0 = torch.reshape(x0, (nx, 1))
    x_init = x0.detach().numpy()
    AB = np.concatenate([A, B], axis=1)
    # Koopman MPC
    if i % 1 == 0:
        lamdaCostFunction=lambda x: costFunction(x,r,AB,C,x_init,u, MPCHorizon, ControlHorizon, np.zeros((8, 1)))
        result=optimize.minimize(lamdaCostFunction,pastRes,bounds=bounds)
        u=np.array(result.x[0]).reshape((1,1))
    print("\nu : ", u)




    # Locally linear MPC
    if i % 1 == 0:
        '''
        # E555 =A_Jacobian.subs([(x1, x_loc[0]), (x2, x_loc[1]), (uu, u_loc)])
        A_loc = np.array(A_Jacobian.subs([(x1, x_loc[0,0]), (x2, x_loc[1,0]), (uu, u_loc[0,0])]))
        B_loc = np.array(B_Jacobian.subs([(x1, x_loc[0,0]), (x2, x_loc[1,0]), (uu, u_loc[0,0])]))
        C_loc = np.array(f_ud.subs([(x1, x_loc[0,0]), (x2, x_loc[1,0]), (uu, u_loc[0,0])])) - A_loc @ x_loc - B_loc @ u_loc
        '''
        A_loc = np.array(A_Jacobian.subs([(x1, 0.0), (x2, 0.0), (uu, u_loc.squeeze())]))
        B_loc = np.array(B_Jacobian.subs([(x1, 0.0), (x2, 0.0), (uu, u_loc.squeeze())]))
        C_loc = np.array(f_ud.subs([(x1, 0.0), (x2, 0.0), (uu, u_loc.squeeze())])) - B_loc @ u_loc

        Bc = np.concatenate((B_loc, C_loc), axis=1)
        ABC = np.concatenate((A_loc, Bc), axis=1)
        ABc = np.concatenate((ABC, np.zeros((2, 4))))
        AT = ABc * h
        ABc_d = linalg.expm(AT)
        Aloc_d = ABc_d[0 : 2, 0 : 2]
        Aloc_d = np.array(Aloc_d).reshape((2, 2))
        Bloc_d = ABc_d[0 : 2, 2]
        Bloc_d = np.array(Bloc_d).reshape((2, 1))
        Cloc_d = ABc_d[0 : 2, 3]
        Cloc_d = np.array(Cloc_d).reshape((2, 1))
        ABloc_d = np.concatenate((Aloc_d, Bloc_d), axis=1)
        lamdaCostFunction=lambda x: costFunction(x,r,ABloc_d,np.identity(2),x_origin,u_loc, MPCHorizon, ControlHorizon, Cloc_d)
        result_loc=optimize.minimize(lamdaCostFunction,pastRes_loc,bounds=bounds)
        u_loc=np.array(result_loc.x[0]).reshape((1,1))
    if LQR == True:
        u = -K_gain @ xlift
        xlift = A @ xlift + B @ u
        print("\nxlift : ", xlift)
    #u = np.array(0).reshape((1,1))
    print("\nuloc : ", u_loc)
    pastRes=result.x
    pastRes[0:-1]=pastRes[1:]


    x_origin = np.array(f_update(0, x_origin, u))
    x_origin = np.array(x_origin)
    x_origin = x_origin.astype(float)
    x_origin = np.reshape(x_origin,(2, 1))
    x0 = torch.tensor(x_origin)
    x0 = torch.squeeze(x0)
    logU = np.concatenate((logU, u), axis=1)
    logX = np.concatenate((logX, x_origin), axis=1)
    print("\nx_origin : ", x_origin)

    ylift = torch.tensor(x_origin)
    ylift = torch.reshape(ylift, (1, len(x_origin)))
    ylift = net.Encoder(ylift)
    ylift = ylift.detach().numpy()
    ylift = np.reshape(ylift, (8, 1))


    # Update LTV-Model

    X_EX = np.concatenate((X_EX, xlift), axis = 1)
    Y_EX = np.concatenate((Y_EX, ylift), axis = 1)
    U_EX = np.concatenate((U_EX, u), axis = 1)
    XU_EX = np.concatenate((X_EX, U_EX), axis = 0)

    # K_hat = PHIY @ torch.pinverse(PHIY_U)
    # A_hat = K_hat[0: Nlift, 0: Nlift]
    # B_hat = K_hat[0: Nlift, Nlift]
    # B_hat = torch.reshape(B_hat, (len(B), m))
    xlift_u = np.concatenate((xlift, u), axis=0)
    K_A = (M * K_A + q * ylift @ xlift_u.transpose()) / (M + q)
    K_G = (M * K_G + q * xlift_u @ xlift_u.transpose()) / (M + q)
    M = M + q
    K_ext = K_A @ np.linalg.pinv(K_G)
    # K_ext = Y_EX @ np.linalg.pinv(XU_EX)
    A_prev = K_ext[0: Nlift, 0: Nlift]
    B_prev  = K_ext[0: Nlift, Nlift]
    B_prev  = np.reshape(B, (len(B), m))
    C = logX @ np.linalg.pinv(X_EX)

    T_EX.append(i * h)
    print("A error: ", np.linalg.norm(A - A_prev, ord=2))
    print("B error: ", np.linalg.norm(B - B_prev, ord=2))
    A_error.append(np.linalg.norm((A - A_prev), ord=2))
    B_error.append(np.linalg.norm(B - B_prev, ord=2))
    A = A_prev
    B = B_prev

    pastRes_loc = result_loc.x
    pastRes_loc[0:-1] = pastRes_loc[1:]
    x_loc = np.array(f_update(0, x_loc, u_loc))
    x_loc = np.array(x_loc)
    x_loc = x_loc.astype(float)
    x_loc = np.reshape(x_loc, (2, 1))
    logUloc = np.concatenate((logUloc, u_loc), axis=1)
    logXloc = np.concatenate((logXloc, x_loc), axis=1)
    print("\nx_loc : ", x_loc)

    '''
    plt.clf()
    plt.plot(np.linspace(0.0, (i + 1) * h, i + 1),  logX[0, :], label="y_real", linewidth=3.0)
    plt.legend(loc="lower right")
    # plt.plot(tspan, logR[0, :], label = "Ref = $0.5cos(0.007t) + 1.2sin(0.002t)$",linewidth=2.0, linestyle='--')#sin(j/(20+0.01*j))
    # plt.plot(tspan, logR[0, :], label = "Ref = $sin(t/(20+0.01*t))$",linewidth=2.0, linestyle='--')
    plt.plot(np.linspace(0.0, (i + 1) * h, i + 1), logR[0, :], label="y_ref", linewidth=2.0, linestyle='--')
    # plt.plot(tspan, logR[0, :], label = "Ref = squared wave, T = 50s",linewidth=2.0, linestyle='--')
    plt.legend(loc="lower right")
    # plt.plot(tspan, logR[1, :], label = "Ref_2 ",linewidth=2.0, linestyle='--')

    plt.grid()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('MPC Tracking trajectory')

    plt.pause(0.001)
    '''
    pass
print('\n')
scio.savemat('DeepLearning_KoopmanControl_Approach3_vanderpol_11.mat', {'A': A,'B': B,'PHIX' : PHIX.detach().numpy(), 'X':X, 'Y':Y , 'x0': x_init, 'logU': logU})

plt.figure()
for i in range (0, 8):
    plt.plot(tspan, logXlift[i, :], label = "Lifted trajectory %d"%(i + 1))
    plt.legend(loc="lower right")
plt.grid()
plt.title('Lifted trajectories')
plt.xlabel('$t/s$')
plt.ylabel('$Lifted x$')


plt.figure()

plt.plot(tspan, logX[0, :], label = "y = $x_1$ trajectory ",linewidth=3.0)
plt.legend(loc="lower right")
#plt.plot(tspan, logX[1, :], label = "x_2 trajectory ",linewidth=3.0)
plt.legend(loc="lower right")
#plt.plot(tspan, logR[0, :], label = "Ref = $0.5cos(0.007t) + 1.2sin(0.002t)$",linewidth=2.0, linestyle='--')#sin(j/(20+0.01*j))
#plt.plot(tspan, logR[0, :], label = "Ref = $sin(t/(20+0.01*t))$",linewidth=2.0, linestyle='--')
#plt.plot(tspan, logR[0, :], label = "Ref = $0.7sin(t/(20+0.01t))+0.7$",linewidth=2.0, linestyle='--')

plt.plot(tspan, logR[0, :], label = "Reference",linewidth=2.0, linestyle='--')
pltR = plt.legend(loc="lower right")
#plt.plot(tspan, logR[1, :], label = "Ref_2 ",linewidth=2.0, linestyle='--')
pltR = plt.legend(loc="lower right")
plt.grid()
plt.xlabel('$t/s$')
plt.ylabel('$yRef$')
plt.title('MPC Tracking trajectory')

#plt.legend([[pltX1], [pltX2],[pltR1], [pltR2]], ['x_1 trajectory', 'x_2 trajectory', 'Ref_1 = 1*sin(0.001x)', 'Ref_2 = 0'], loc='lower right')

plt.figure()
pltPhase, = plt.plot(logX[0, :], logX[1, :], label = "x_1 and x_2 Phase plots ")
plt.legend(loc="lower right")
#plt.legend([[pltPhase]], ['x_1 and x_2 Phase plots'], loc='lower right')
plt.grid()

plt.figure()
pltU, = plt.plot(tspan, logU.ravel(), label = "Control inputs ")
plt.legend(loc="lower right")
plt.grid()
#plt.legend([[pltU]], ['Control inputs'], loc='lower right')
plt.xlabel('$t/s$')
plt.ylabel('$u$')
plt.title('Control inputs')

plt.figure()
# plt.plot(T_EX, A_error / np.linalg.norm(A_error, ord=np.inf), label = "A_error  ")
plt.plot(T_EX, A_error, label = "A_error  ")

plt.legend(loc="lower right")
plt.grid()
#plt.legend([[pltU]], ['Control inputs'], loc='lower right')
plt.xlabel('$t/s$')
plt.ylabel('$A_(k+1) - A_k$')
# plt.xlim(0, 10)
plt.ylim(0, 1)
plt.title('Matrix A error')

plt.figure()
# pltMatrixErrorB, = plt.plot(T_EX, B_error / np.linalg.norm(B_error, ord=np.inf), label = "B_error ")
pltMatrixErrorB, = plt.plot(T_EX, B_error, label = "B_error ")

plt.legend(loc="lower right")
plt.grid()
#plt.legend([[pltU]], ['Control inputs'], loc='lower right')
plt.xlabel('$t/s$')
plt.ylabel('$B_(k+1) - B_k$')
# plt.xlim(0, 10)
plt.ylim(0, 1)
plt.title('Matrix B error')



#locally linear plots
plt.figure()

plt.plot(tspan, logXloc[0, :], label = "y = $x_1$ trajectory ",linewidth=3.0)
plt.legend(loc="lower right")
#plt.plot(tspan, logX[1, :], label = "x_2 trajectory ",linewidth=3.0)
plt.legend(loc="lower right")
#plt.plot(tspan, logR[0, :], label = "Ref = $0.5cos(0.007t) + 1.2sin(0.002t)$",linewidth=2.0, linestyle='--')#sin(j/(20+0.01*j))
#plt.plot(tspan, logR[0, :], label = "Ref = $sin(t/(20+0.01*t))$",linewidth=2.0, linestyle='--')
#plt.plot(tspan, logR[0, :], label = "Ref = $0.7sin(t/(20+0.01t))+0.7$",linewidth=2.0, linestyle='--')

plt.plot(tspan, logR[0, :], label = "Ref = squared wave, T = 50s",linewidth=2.0, linestyle='--')
pltR = plt.legend(loc="lower right")
#plt.plot(tspan, logR[1, :], label = "Ref_2 ",linewidth=2.0, linestyle='--')
pltR = plt.legend(loc="lower right")
plt.grid()
plt.xlabel('$t/s$')
plt.ylabel('$y$')
plt.title('MPC Tracking trajectory (locally linear)')

#plt.legend([[pltX1], [pltX2],[pltR1], [pltR2]], ['x_1 trajectory', 'x_2 trajectory', 'Ref_1 = 1*sin(0.001x)', 'Ref_2 = 0'], loc='lower right')

plt.figure()
pltPhase, = plt.plot(logXloc[0, :], logXloc[1, :], label = "x_1 and x_2 Phase plots ")
plt.legend(loc="lower right")
#plt.legend([[pltPhase]], ['x_1 and x_2 Phase plots'], loc='lower right')
plt.grid()

plt.figure()
pltU, = plt.plot(tspan, logUloc.ravel(), label = "Control inputs ")
plt.legend(loc="lower right")
plt.grid()
#plt.legend([[pltU]], ['Control inputs'], loc='lower right')
plt.xlabel('$t/s$')
plt.ylabel('$u$')
plt.title('Control inputs (locally linear)')

plt.show()













































































































