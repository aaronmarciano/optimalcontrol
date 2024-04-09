import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# definissons les paramètres de la simu

eta_inf = 1
n_sup = 0.006
lambd = 1 / 10

T = 350
timestep = 0.1
n_steps = T / timestep

N = 30
M = 40
r_0  = 15
gamma = 0.09
mu = 2 / 100
sigma = 1
C = 0.5
pop = 1e6
h = 1.05 / pop
transmission_rate = 0.2

# useful routines

def tensordot(u,v):
    dim_u = len(u)
    dim_v = len(v)

    res = np.zeros((dim_u,dim_v))

    for iu in range(dim_u):
        for iv in range(dim_v):
            res[iu,iv] = u[iu] * v[iv]
    return res

def psi(x):
    if abs(x) < h:
        return 0
    elif abs(x) > 2 * h:
        return x
    elif x <= 2 * h  and x >= h:
        return -3 * (x**3)/(h**2) + 14 * (x**2) / h - 19 * x + 8 * h
    elif x <= -h and x >= -2 * h:
        return -3 * (x**3)/(h**2) - 14 * (x**2) / h - 19 * x - 8 * h
    
def dpsi(x):
    if abs(x) < h:
        return 0
    elif abs(x) > 2 * h:
        return 1
    elif x <= 2 * h  and x >= h:
        return -9 * (x**2)/(h**2) + 28 * x / h - 19
    elif x <= -h and x >= -2 * h:
        return -9 * (x**2)/(h**2) - 28 * x / h - 19 

def grandPsi(mat):
    return np.vectorize(psi)(mat)

def grandDpsi(mat):
    return np.vectorize(dpsi)(mat)

def grandP():
    return np.array([[[[1/(2*np.pi*sigma**2)*np.exp(-((i-k)**2+(j-l)**2)/(2*sigma**2)) for i in range(N)] for j in range(M)] for k in range(N)] for l in range(M)])





# START

n = np.zeros(n_steps)
eta = np.zeros(n_steps)

I = np.zeros((n_steps,N,M))
R = np.zeros((n_steps,N,M))
S = np.zeros(n_steps)
V = np.zeros(n_steps)
D = np.zeros(n_steps)

beta = np.array([gamma * r_0 * (i - 1) / (N - 1) for i in range(1,N+1)])
omega = np.array([(j - 1) / (M - 1) for j in range(1,M+1)])
ksi = np.array([[[[C / M * max(omega[l] - omega[j],0) for l in range(M)] for k in range(N)] for j in range(M)] for i in range(N)])
ones = np.ones(M)

I_0 = 200e-5
R_0 = 0.25
V_0 = 0
D_0 = 0

I[0,0,0] = I_0
R[0,0,0] = R_0
V[0] = V_0
D[0] = D_0
S[0] = 1 - V_0 - D_0 - R_0 - I_0





n[0], eta[0] = n_sup, eta_inf

for t in range(1,n_steps):
    S_prime = - n[t-1] - eta[t-1] * np.tensordot(np.tensordot(beta,ones,axes=0),I[t-1],axes=2) * S[t-1]
    V_prime = n[t-1] - eta[t-1] * np.tensordot(np.tensordot(beta,omega,axes=0),I[t-1],axes=2) * V[t-1]
    I_prime_1 = eta[t-1] * np.multiply(np.tensordot(beta,ones,axes=0),I[t-1]) * S[t-1]
    I_prime_2 = eta[t-1] * np.multiply(np.tensordot(beta,omega,axes=0),I[t-1]) * V[t-1]
    I_prime_3 = -mu * I[t-1] - gamma * I[t-1] 
    I_prime_4 = eta[t-1] * np.multiply(np.tensordot(ksi,R[t-1],axes=2),I[t-1])
    I_prime_5 = grandPsi(gaussian_filter(I[t-1],sigma,mode='constant',cval=0) - I[t-1])

    I_prime = I_prime_1 + I_prime_2 + I_prime_3 + I_prime_4 + I_prime_5

    R_prime = gamma * I[t-1] - eta[t-1] * np.tensordot(np.transpose(ksi,(2,3,0,1)),R[t-1],axes=2)*I[t-1]
    D_prime = mu * np.tensordot(I[t-1],np.tensordot(ones,ones,axes=0),axes=2)

    S[t] = S[t-1] + timestep * S_prime
    V[t] = V[t-1] + timestep * V_prime
    I[t] = I[t-1] + timestep * I_prime
    R[t] = R[t-1] + timestep * R_prime
    D[t] = D[t-1] + timestep * D_prime

p_S = np.zeros(n_steps)
p_I = np.zeros((n_steps,N,M))
p_R = np.zeros((n_steps,N,M))
p_V = np.zeros(n_steps)

for t in range(n_steps-1,-1,-1):
    ps_prime_1 = -eta[t] * np.tensordot(p_I[t],np.multiply(np.tensordot(beta,ones,axes=0),I[t]),axes=2)
    ps_prime_2 = eta[t] * p_S[t] *np.tensordot(np.tensordot(beta,ones,axes=0),I[t],axes=2)
    ps_prime = ps_prime_1 + ps_prime_2

    pv_prime_1 = -eta[t] * np.tensordot(p_I[t],np.multiply(np.tensordot(beta,omega,axes=0),I[t]),axes=2)
    pv_prime_2 = eta[t] * p_V[t] * np.tensordot(np.tensordot(beta,omega,axes=0),I[t],axes=2)
    pv_prime = pv_prime_1 + pv_prime_2

    pi_prime_1 = mu * (p_I[t] - np.tensordot(ones,ones,axes=0)) + gamma *(p_I[t] - p_R[t])
    pi_prime_2 = eta[t] * (p_S[t] * np.tensordot(beta,ones,axes=0) - np.multiply(p_I[t],np.tensordot(beta,ones,axes=0))) * S[t]
    pi_prime_3 = eta[t] * (p_V[t] * np.tensordot(beta,omega,axes=0) - np.multiply(p_I[t],np.tensordot(beta,omega,axes=0))) * V[t]
    pi_prime_4 = -np.tensordot(np.transpose(grandP(),(2,3,0,1)),p_I[t] * grandDpsi(gaussian_filter(I[t],sigma,mode='constant',cval=0)-I[t]),axes=2)
    pi_prime_5 = np.multiply(p_I[t],grandDpsi(gaussian_filter(I[t],sigma,mode='constant',cval=0)-I[t]))
    pi_prime_6 = -eta[t] * np.multiply(p_I[t],np.tensordot(ksi,R[t],axes=2))
    pi_prime_7 = eta[t] * np.tensordot(ksi,np.multiply(R[t],p_R[t]),axes=2)

    pi_prime = pi_prime_1+pi_prime_2+pi_prime_3+pi_prime_4+pi_prime-5+pi_prime_6+pi_prime_7

    pr_prime_1 = -eta[t] * np.multiply(I[t],np.transpose(ksi,(2,3,0,1)) * p_I[t])
    pr_prime_2 = eta[t] * np.multiply(I[t],np.tensordot(ksi,p_R[t],axes=2))

    pr_prime = pr_prime_1 + pr_prime_2


    p_S[t-1] = p_S[t] - timestep * ps_prime
    p_V[t-1] = p_V[t] - timestep * pv_prime
    p_I[t-1] = p_I[t] - timestep * pi_prime
    p_R[t-1] = p_R[t] - timestep * pr_prime


glam_n = p_V - p_S
glam_eta_1 = - p_S * np.tensordot(np.tensordot(beta,ones,axes=0),I,axes=2) * S
glam_eta_2 = - p_V * np.tensordot(np.tensordot(beta,omega,axes=0),I,axes=2) * V
glam_eta_3 = - np.tensordot(p_R,np.multiply(np.tensordot(np.transpose(ksi,(2,3,0,1)),I,axes=2),R),axes=2)
glam_eta_4 =  np.tensordot(p_I,np.multiply(np.tensordot(beta,ones,axes=0),I),axes=2) * S
glam_eta_5 =  np.multiply(np.tensordot(beta,omega,axes=0),I) * V
glam_eta_6 = np.multiply(np.tensordot(ksi,I,axes=2),I)

glam_eta = glam_eta_1+glam_eta_2+glam_eta_3+glam_eta_4+glam_eta_5+glam_eta_6


n_tilde = np.zeros(n_steps)
eta_tilde = np.zeros(n_steps)

threshold = 1e-2

for t in range(n_steps):
    if glam_eta[t] <= 0:
        eta_tilde[t] = eta_inf
    else:
        eta_tilde[t] = 1
    if glam_n[t] >=0:
        n_tilde[t] = 0
    else:
        n_tilde[t] = n_sup

new_n = (1 - lambd) * n + n_tilde
new_eta = (1 - lambd) * eta + eta_tilde

itermax = 10000
iter=0
while not (np.linalg.norm(new_n - n) <= threshold and np.linalg.norm(new_eta - eta) <= threshold) and iter < itermax :
    n = new_n
    eta = new_eta

    new_n = (1 - lambd) * n + n_tilde
    new_eta = (1 - lambd) * eta + eta_tilde
    iter += 1



