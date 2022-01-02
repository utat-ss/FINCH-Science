"""
optim.py  

Sets up matrices and runs optimization for the LEA program.

Author(s): Adyn Miles, Shiqi Xu, Rosie Liang
"""

import matplotlib.pyplot as plt
import numpy as np

import lib.hapi as hp


class Optim:
    def __init__(self, cfg, wave_meas):
        self.cfg = cfg
        self.wave_meas = wave_meas

    def jacobian(self, dev_ch4, dev_co2, dev_h2o, show_fig=True):
        '''
        Composes Jacobian matrix K which is based on the derivatives of each element of the forward model matrix.

        K is a matrix of dimensions (i x j) 

        i = number of points along the spectral grid
        j = number of molecules being investigated (in our case, 3: CH4, CO2, and H2O)

        Parameters:
            dev_ch4: vector containing the derivatives of the methane forward model response.
            dev_co2: vector containing the derivatives of the carbon dioxide forward model response.
            dev_h2o: vector containing the derivatives of the water vapour forward model response.

        Returns:
            K: instrument spectral response function convolved with the spectral response.
        '''
        # Conversion to cm^(-1), the units for wavenumber.
        self.wave_meas = np.flip(1e7/(self.wave_meas)) 
        self.fwhm = self.wave_meas[2] - self.wave_meas[0]
        self.wave_meas = np.append(self.wave_meas, self.wave_meas[-1] + (0.5*self.fwhm))
        self.wave_meas = np.insert(self.wave_meas, 0, self.wave_meas[0] - (0.5*self.fwhm))
        self.wave_meas_1, self.d_isrf_conv_1, i1, i2, slit = hp.convolveSpectrum(
                    self.wave_meas, dev_ch4, 
                    SlitFunction=hp.SLIT_DIFFRACTION, AF_wing=0.0, Resolution=self.fwhm)

        self.wave_meas_2, self.d_isrf_conv_2, i1, i2, slit = hp.convolveSpectrum(
                    self.wave_meas, dev_co2, 
                    SlitFunction=hp.SLIT_DIFFRACTION, AF_wing=0.0, Resolution=self.fwhm)

        self.wave_meas_3, self.d_isrf_conv_3, i1, i2, slit = hp.convolveSpectrum(
                    self.wave_meas, dev_h2o, 
                    SlitFunction=hp.SLIT_DIFFRACTION, AF_wing=0.0, Resolution=self.fwhm)            

        self.K = np.zeros((len(self.wave_meas_1), 3))
        self.K[:, 0] = self.d_isrf_conv_1
        self.K[:, 1] = self.d_isrf_conv_2
        self.K[:, 2] = self.d_isrf_conv_3
        
        return self.K

    def gain(self, S_y):
        '''
        Composes gain matrix G using the following equation:

        G = (((K^T)*(S_y^(-1))*K)^(-1)) * ((K^T)*(S_y^(-1)))

        K = Jacobian matrix
        S_y = random error covariance matrix

        Parameters:
            S_y: random error covariance matrix

        Returns:
            K: instrument spectral response function convolved with the spectral response.
        '''
        self.G = np.matmul(np.matmul(np.matmul(np.transpose(self.K),np.linalg.inv(S_y)),self.K),(np.matmul(np.transpose(self.K),np.linalg.inv(S_y))))

        return self.G

    def modify_state_vector(self, x_0, F, S_y):
        '''
        Modifies the state vector to include the forward model information

        Parameters:
            y: measurement vector
            x_0: state vector
            F: forward model response, which is the instrument spectral response function convolved with the radiance.

        Returns:
            y_tilde: modified state vector
        '''
        # Poisson noise vector
        e = []
        iter_range = np.shape(S_y)
        for i in range(0, iter_range[0]):
            e.append(np.sqrt(S_y[i, i]))

        y = F + e
        y_tilde = y - F + np.matmul(self.K, x_0)

        return y_tilde

    def state_estimate(self, S_y, y, system_errors):
        '''
        Calculates the state estimates x_est and S_x for both spectral resolution and the signal to noise ratio.

        Parameters:
            S_y: random error covariance matrix
            y: modified state vector

        Returns:
            x_est: spectral resolution estimate
            S_x: signal to noise ratio estimate
        '''
        x_est = np.matmul(self.G, y)
        S_x = np.matmul(self.G, np.matmul(S_y, np.transpose(self.G)))

        return x_est, S_x