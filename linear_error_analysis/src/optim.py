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
        self.wave_meas_1, self.d_isrf_conv_1, i1, i2, slit = hp.convolveSpectrum(
                    self.wave_meas, dev_ch4, 
                    SlitFunction=hp.SLIT_DIFFRACTION, Resolution=self.cfg.fwhm)

        self.wave_meas_2, self.d_isrf_conv_2, i1, i2, slit = hp.convolveSpectrum(
                    self.wave_meas, dev_co2, 
                    SlitFunction=hp.SLIT_DIFFRACTION, Resolution=self.cfg.fwhm)

        self.wave_meas_3, self.d_isrf_conv_3, i1, i2, slit = hp.convolveSpectrum(
                    self.wave_meas, dev_h2o, 
                    SlitFunction=hp.SLIT_DIFFRACTION, Resolution=self.cfg.fwhm)            

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
        self.G = (np.transpose(self.K)*np.linalg.inv(S_y)*self.K)*(np.transpose(self.K)*np.linalg.inv(S_y))

        return self.G




