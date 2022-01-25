"""
optim.py

Sets up linearization and state space mapping to produce column concentration estimates.

Author(s): Adyn Miles, Shiqi Xu, Rosie Liang
"""

import matplotlib.pyplot as plt
import numpy as np
from isrf import ISRF

import libs.hapi as hp


class Optim:
    def __init__(self, cfg, wave_meas):
        self.cfg = cfg
        self.wave_meas = wave_meas

    def jacobian(self, dev_ch4, dev_co2, dev_h2o, show_fig=True):
        """
        Composes Jacobian matrix K which is based on the derivatives of each element of the forward model matrix.

        K is a matrix of dimensions (i x j)

        i = number of points along the spectral grid
        j = number of molecules being investigated (in our case, 3: CH4, CO2, and H2O)

        Parameters:
            dev_ch4: vector containing the derivatives of the methane forward model response.
            dev_co2: vector containing the derivatives of the carbon dioxide forward model response.
            dev_h2o: vector containing the derivatives of the water vapour forward model response.

        Returns:
            K: Jacobian matrix of derivatives convolved with the instrument spectral response function.
        """

        # This uses the outputs from slit_conv() to produce the Jacobian values. 
        self.K = np.zeros((len(dev_ch4), 3))
        self.K[:, 0] = dev_ch4
        self.K[:, 1] = dev_co2
        self.K[:, 2] = dev_h2o

        return self.K

    def gain(self, S_y):
        """
        Composes gain matrix G using the following equation:

        G = (((K^T)*(S_y^(-1))*K)^(-1)) * ((K^T)*(S_y^(-1)))

        K = Jacobian matrix
        S_y = random error covariance matrix

        Parameters:
            self.K: Jacobian matrix of derivatives convolved with the instrument spectral response function.
            S_y: random error covariance matrix

        Returns:
            G: gain matrix for mapping the measurement space to the state space.
        """
        self.G_1 = np.linalg.inv(
            np.matmul(np.transpose(self.K), np.matmul(np.linalg.inv(S_y), self.K))
        )
        self.G_2 = np.matmul(np.transpose(self.K), np.linalg.inv(S_y))
        self.G = np.matmul(self.G_1, self.G_2)

        return self.G

    def modify_meas_vector(self, x_0, F, S_y):
        """
        Modifies the measurement vector to include the forward model information, and linearizes it for
        error analysis.

        Parameters:
            x_0: state vector
            F: forward model response, which is the instrument spectral response function convolved with the radiance.
            S_y: random error covariance matrix
            self.K: Jacobian matrix of derivatives convolved with the instrument spectral response function

        Returns:
            y_tilde: modified state vector
        """
        # Poisson noise vector, taken to be the square root of the diagonal elements of the covariance matrix, 
        # which consists of the covariance of random noise along the diagonal.
        e = []
        iter_range = np.shape(S_y)
        for i in range(0, iter_range[0]):
            e.append(np.sqrt(S_y[i, i]))

        y = F + e
        y_tilde = y - F + np.matmul(self.K, x_0)

        return y_tilde

    def state_estimate(self, S_y, y_tilde):
        """
        Calculates the state estimates x_est and S_x for both spectral resolution and the signal to noise ratio.

        Parameters:
            S_y: random error covariance matrix
            y_tilde: modified measurement vector

        Returns:
            x_est: spectral resolution estimate
            S_x: signal to noise ratio estimate
        """
        x_est = np.matmul(self.G, y_tilde)
        S_x = np.matmul(self.G, np.matmul(S_y, np.transpose(self.G)))

        return x_est, S_x

    def system_est_error(self, system_error):
        """
        Calculates the state estimates x_est and S_x for both spectral resolution and the signal to noise ratio.

        Parameters:
            system_error: specific error chosen from the system_errors vector 

        Returns:
            dx_est: Deviation caused by this systematic error.
        """
        dx_est = np.matmul(self.G, system_error)
        dx_est = np.reshape(dx_est, (np.size(dx_est),))

        return dx_est
