from random_vector import Random_Vector
import numpy as np
import scipy
import types
import matplotlib.pyplot as plt


class BaseKF:
    def __init__(self, f_k, h_k, X, W, V, u, y_actual):
        if isinstance(f_k, types.FunctionType) is False:
            raise TypeError('Expected state_func to be of type "Function".')
        if isinstance(h_k, types.FunctionType) is False:
            raise TypeError('Expected output_func to be of type "Function".')
        if type(X) is not Random_Vector:
            raise TypeError("Expected state vector, X, to be of type Random_Vector.")
        if type(W) is not Random_Vector:
            raise TypeError("Expected process noise vector, W, to be of type Random_Vector.")
        if type(V) is not Random_Vector:
            raise TypeError("Expected sensor noise vector, V, to be of type Random_Vector.")
        if type(u) is not np.ndarray:
            raise TypeError("Expected input to be a Numpy array.")
        if type(y_actual) is not np.ndarray:
            raise TypeError("Expected actual output to be a Numpy array.")
        self.f_k = f_k
        self.h_k = h_k
        self.X = X
        self.W = W
        self.V = V
        self.u = u
        self.y_actual = y_actual


class EKF(BaseKF):
    def __init__(self, f_k, h_k, A_hat, B_hat, C_hat, D_hat, X, W, V, u, y_actual):
        super().__init__(f_k, h_k, X, W, V, u, y_actual)
        if type(A_hat) is not np.ndarray:
            raise TypeError('Expected A_bar to be Numpy object')
        if type(B_hat) is not np.ndarray:
            raise TypeError('Expected B_bar to be Numpy object')
        if type(C_hat) is not np.ndarray:
            raise TypeError('Expected C_bar to be Numpy object')
        if type(D_hat) is not np.ndarray:
            raise TypeError('Expected D_bar to be Numpy object')
        self.A_hat = A_hat
        self.B_hat = B_hat
        self.C_hat = C_hat
        self.D_hat = D_hat

    @staticmethod
    def cov_pred(A_hat, B_hat, X_cov, W_cov):
        return A_hat @ X_cov @ A_hat.transpose() + B_hat @ W_cov @ B_hat.transpose()

    @staticmethod
    def kalman_gain(C_hat, D_hat, X_cov, V_cov):
        y_cov = C_hat @ X_cov @ C_hat.transpose() + D_hat @ V_cov @ D_hat.transpose()
        y_conv_inv = np.linalg.inv(y_cov.astype(float))
        L_k = X_cov @ C_hat @ y_conv_inv
        return L_k, y_cov

    @staticmethod
    def cov_est(L_k, X_cov, y_cov):
        return X_cov - L_k @ y_cov @ L_k.transpose()

    def calc(self):
        X_estimates = np.zeros((self.X.length, len(self.u) - 1))
        X_covariances = np.zeros((self.X.length, self.X.length, len(self.u) - 1))
        for i in range(1, len(self.u)):
            # Step 1a: State prediction
            # print(self.X.mean, self.u[i-1], self.W.mean.float())
            self.X.mean = self.f_k(self.X.mean.astype(float), self.u[i-1], self.W.mean)
            # Step 1b: State covariance prediction
            A_hat = np.vectorize(lambda f, x, u, w: f(x, u, w), otypes=[object])
            B_hat = np.vectorize(lambda f, x, u, w: f(x, u, w), otypes=[object])
            A_hat = A_hat(self.A_hat, self.X.mean, self.u[i - 1], self.W.mean)
            B_hat = B_hat(self.B_hat, self.X.mean, self.u[i - 1], self.W.mean)
            self.X.covariance = self.cov_pred(A_hat, B_hat, self.X.covariance, self.W.covariance)
            ## Step 1c: output prediction
            y_pred = self.f_k(self.X.mean, self.u[i], self.V.mean)

            # Step 2a: Kalman Gain
            C_hat = np.vectorize(lambda f, x, u, w: f(x, u, w), otypes=[object])
            C_hat = C_hat(self.C_hat, self.X.mean, self.u[i], self.V.mean)
            D_hat = np.vectorize(lambda f, x, u, w: f(x, u, w), otypes=[object])
            D_hat = D_hat(self.D_hat, self.X.mean, self.u[i], self.V.mean)
            L_k, Y_cov = self.kalman_gain(C_hat, D_hat, self.X.covariance, self.V.covariance)
            # Step 2b: State-estimate
            y_tilde = (self.y_actual[i-1] - y_pred).reshape(-1, 1)
            self.X.mean = self.X.mean + np.matmul(L_k, y_tilde)
            # Step 2c: State covariance estimate
            self.X.covariance = self.cov_est(L_k, self.X.covariance, Y_cov)

            X_estimates[:, i - 1] = self.X.mean.flatten()
            X_covariances[:, :, i - 1] = self.X.covariance

        return X_estimates, X_covariances

    def __repr__(self):
        return "EFK"

    def __str__(self):
        return "EFK"

class SPKF(BaseKF):
    def __init__(self, f_k, h_k, X, W, V, u, y_actual, kind = "CDKF"):
        super().__init__(f_k, h_k, X, W, V, u, y_actual)
        if type(f_k) != types.FunctionType:
            raise TypeError("Expected state eqn (f_k) to be a function type.")
        if type(h_k) != types.FunctionType:
            raise TypeError("Expected output eqn (h_K) to be of function type.")
        if (kind != "CDKF") and (kind != "UKF"):
            raise ValueError("Expected SPKF kind to be CDKF (Central Difference KF) or UKF (Unscented KF).")
        self.kind = kind
        self.Nxa = self.X.length + self.W.length + self.V.length  # The length of the augmented random state vector
        self.N_sigma_pts = 2 * self.Nxa + 1
        self.h = np.sqrt(3) # scaling parameter is sqrt(3) for Guassian RVs.

    @property
    def gamma(self):
        """
        :return: Float of tuning parameter h
        """
        if self.kind == "CDKF":
            return self.h

    @property
    def alpha_m(self):
        """
        :return: Numpy array of alpha_m parameters.
        """
        if self.kind == "CDKF":
            alpha_m_ = np.zeros(self.N_sigma_pts)
            alpha_m_[0] = ((self.h ** 2) - self.Nxa) / (self.Nxa ** 2)
            for i in range(1,len(alpha_m_)):
                alpha_m_[i] = 1/(2 * (self.h ** 2))
        return alpha_m_

    @property
    def alpha_c(self):
        """
        :return: Numpy array of alpha_c parameters.
        """
        if self.kind == "CDKF":
            return self.alpha_m

    def create_aug_vec(self):
        xa = np.concatenate((self.X.mean, self.W.mean, self.V.mean), axis=0)
        Sigma_xa = scipy.linalg.block_diag(self.X.covariance, self.W.covariance, self.V.covariance)
        sSigma_xa = np.linalg.cholesky(np.absolute(Sigma_xa))
        return xa, Sigma_xa, sSigma_xa

    def create_sigma_pts(self, xa, sSigma_xa):
        Xa = np.zeros((self.Nxa, 2 * self.Nxa + 1))
        Xa[:, 0] = xa.flatten() # first column of Xa is equal to xa
        for n, i in enumerate(range(1, self.Nxa + 1)):
            Xa[:,i] = xa.flatten() + self.gamma * sSigma_xa[:,n]
        for n, i in enumerate(range(self.Nxa + 1, 2 * self.Nxa + 1)):
            Xa[:,i] = xa.flatten() - self.gamma * sSigma_xa[:,n]
        return Xa

    def calc(self):
        X_estimates = np.zeros((self.X.length, len(self.u) - 1))
        X_covariances = np.zeros((self.X.length, self.X.length, len(self.u) - 1))
        for i in range(1, len(self.u)):
            # Step 1a: State prediction
            xa, Sigma_xa, sSigma_xa = self.create_aug_vec()
            Xa = self.create_sigma_pts(xa, sSigma_xa)
            X_x, X_w, X_v = Xa[0:self.X.length,:], Xa[self.X.length: self.X.length + self.W.length,:], Xa[self.X.length + self.W.length:,:]
            X_x = self.f_k(X_x, self.u[i-1], X_w)
            # print(np.matmul(X_x, np.array([self.alpha_m]).transpose()))
            self.X.mean = np.matmul(X_x, np.array([self.alpha_m]).transpose())
            # Step 1b: State covariance prediction
            X_x_tilde = X_x - self.X.mean
            self.X.covariance = np.matmul(np.matmul(X_x_tilde, np.diag(self.alpha_c)), X_x_tilde.transpose())
            # Step 1c: predict output
            Y_k = self.h_k(X_x, self.u[i], X_v)
            y_k = np.matmul(Y_k, self.alpha_c)

            # Step 2a: Kalman Gain
            Y_tilde = Y_k - y_k.reshape(-1,1)
            # print(Y_tilde)
            y_cov = np.matmul(np.matmul(Y_tilde, np.diag(self.alpha_c)), Y_tilde.transpose())
            xy_cov = np.matmul(np.matmul(X_x_tilde, np.diag(self.alpha_c)), Y_tilde.transpose())
            L_k = np.matmul(xy_cov, np.linalg.inv(y_cov))
            # Step 2b: State estimate
            self.X.mean = self.X.mean + np.matmul(L_k, (self.y_actual[i-1] - y_k).reshape(-1,1))
            # Step 2c: Covariance estimate
            self.X.covariance = self.X.covariance - np.matmul(np.matmul(L_k, y_cov), L_k.transpose())
            # _,S,V = np.linalg.svd(self.X.covariance)
            # HH = V @ S @ V
            # self.X.covariance = (self.X.covariance + self.X.covariance.transpose() + HH + HH.transpose())/4
            # print(i)

            X_estimates[:, i - 1] = self.X.mean.flatten()
            X_covariances[:,:,i - 1] = self.X.covariance

        return X_estimates, X_covariances

    def __repr__(self):
        return "SPKF"

    def __str__(self):
        return "SPFK"