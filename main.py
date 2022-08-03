from random_vector import Random_Vector
from filter import EKF, SPKF
import numpy as np
import matplotlib.pyplot as plt


#                                           Test
# ------------------------------------------------------------------------------
# Initialization
def state_func(x_k, u_k, w_k):
    return np.sqrt(5 + x_k) + w_k
def output_func(x_k, u_k, v_k):
    return x_k**3 + v_k
A_hat = np.array([[lambda x,u,w: (1/(2*np.sqrt(5+x)))]])
B_hat = np.array([[lambda x,u,w: 1]])
C_hat = np.array([[lambda x,u,v: 3*(x**2)]])
D_hat = np.array([[lambda x,u,w: 1]])
X = Random_Vector(np.array([[2]]), np.array([[1]]), 1)
W = Random_Vector(np.array([[0]]), np.array([[1]]), 1)
V = Random_Vector(np.array([[0]]), np.array([[1]]), 1)
max_iter = 40

# -------------- Create true values ----------------------------------------
v_vector = np.random.normal(loc=0, scale=2,size=(max_iter,))
w_vector = np.random.normal(loc=0, scale=1,size=(max_iter,))
x_true_0 = np.random.normal(loc=2, scale=1)
x_true = np.zeros(max_iter)
x_true[0] = x_true_0
for i in range(1,max_iter):
    x_true[i] = state_func(x_true[i-1], 0, w_vector[i])
y_true = np.zeros(max_iter-1)
for i in range(max_iter-1):
    y_true[i] = output_func(x_true[i+1],0,v_vector[i+1])
# ------------------------------------------------------------------------
y_actual = y_true
u = np.zeros(max_iter)

kf = SPKF(state_func, output_func, X, W, V, u, y_actual, kind = "CDKF")
X_estimate_kf, X_covariances_kf = kf.calc()

ekf = EKF(state_func, output_func, A_hat, B_hat, C_hat, D_hat, X, W, V, u, y_actual)
X_estimate, X_covariances = ekf.calc()

t = np.arange(max_iter)
plt.plot(t, x_true, label = "true")
plt.plot(t[1:], X_estimate_kf[0,:], "black", label = "SPKF estimates")
# plt.plot(t[1:], (X_estimate_kf + X_covariances_kf)[0,0,:], 'r--')
# plt.plot(t[1:], (X_estimate_kf - X_covariances_kf)[0,0,:], 'r--')
plt.plot(t[1:], X_estimate[0,:], "red", label = "EKF estimates")
plt.legend()
plt.show()