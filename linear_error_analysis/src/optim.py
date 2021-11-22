"""
optim.py  

Sets up equations and runs optimization for the LEA program.

Author(s): Adyn Miles, Shiqi Xu, Rosie Liang
"""

import numpy as np
import scipy.interpolate as interp

import lib.photon_noise as pn

class Optim:

    def __init__(self, cfg, wave_meas):
        self.cfg = cfg
        self.wave_meas = wave_meas


    def sys_errors(self):
        '''Performs calculations on inputted systematic errors from config.py.

        Args:
            self: contains configuration details from the initialization.

        Returns:
            self.sys_errors: an array containing [total error, non-linearity, 
                stray light, crosstalk, flat-field, bad pixel, keystone/smile, 
                striping, memory] estimates.
        '''
        self.nonlinearity = self.cfg.nonlinearity
        self.stray_light = np.sqrt( (self.cfg.fo_reflectivity)**2 
                                  + (self.cfg.lens_reflectivity)**2
                                  + (self.cfg.mirror_reflectivity)**2
                                  + (self.cfg.ar_coatings)**2
                                  + (self.cfg.leakage)**2
                                  + (self.cfg.ghosting)**2 )
        self.ffr = self.cfg.uniformity
        self.bad_pixels = self.cfg.bad_pixels
        self.crosstalk = self.cfg.crosstalk
        self.key_smile = np.sqrt((self.cfg.keystone)**2 + (self.cfg.smile)**2)
        self.striping = self.cfg.striping
        self.memory = self.cfg.memory

        self.sys_error = np.sqrt( (self.nonlinearity)**2
                                + (self.stray_light)**2
                                + (self.crosstalk)**2
                                + (self.ffr)**2
                                + (self.bad_pixels)**2
                                + (self.key_smile)**2
                                + (self.striping)**2
                                + (self.memory)**2 )

        self.sys_errors = [self.sys_error, self.nonlinearity, self.stray_light, 
                           self.crosstalk, self.ffr, self.bad_pixels, 
                           self.key_smile, self.striping, self.memory]
        print(self.sys_errors)

        return self.sys_errors


    def rand_errors(self):
        '''Performs calculations on inputted random errors from config.py.

        See SNR Analysis Confluence page for breakdown of types of random error:
        http://spacesys.utat.ca/confluence/display/FIN/Signal-to-Noise+Ratio+Analysis

        Args:
            self: contains configuration details from the initialization.

        Returns:
            self.rand_errors: an array containing [total error, dark current, 
                readout, quantization, photon noise] estimates.
        '''
        self.area_detector = self.cfg.x_pixels * (self.cfg.pixel_pitch / 1e6) \
                     * self.cfg.y_pixels * (self.cfg.pixel_pitch / 1e6)
        self.photon_noise = np.array(pn.photon_noise(self.cfg.fwhm))
        self.quant_noise = self.cfg.well_depth / (2**(self.cfg.dynamic_range) \
                     * np.sqrt(12))
        self.dark_current = self.cfg.dark_current * (1e-9) * (6.242e18) \
                     * (self.area_detector * 1e2 * 1e2)
        self.dark_noise = self.dark_current * self.cfg.t_int
        self.readout_noise = self.cfg.readout_noise

        self.rand_error_matrix = np.zeros((len(self.wave_meas), 5), dtype=object)

        # interpolate self.photon_noise to match self.wave_meas grid
        spec_res_series = np.arange(self.cfg.spectral_lower, 
                                    self.cfg.spectral_upper, self.cfg.fwhm)
        pn_func = interp.interp1d(spec_res_series, self.photon_noise, 
                                  fill_value="extrapolate")     # linear
        self.photon_noise_interp = pn_func(self.wave_meas)
        self.signal = self.photon_noise_interp**2
        # note: self.dark_noise is standard deviation of the dark signal squared
        # which is the dark noise added in quadrature, hence no **2
        self.rand_error = np.sqrt(self.signal + self.readout_noise**2 \
                     + self.quant_noise**2 + self.dark_noise)

        for i in range(len(self.wave_meas)):
            self.rand_error_matrix[i, :] = [self.rand_error[i]/self.signal[i], 
                                np.sqrt(self.dark_noise)/self.signal[i], 
                                self.readout_noise/self.signal[i], 
                                self.quant_noise/self.signal[i], 
                                self.photon_noise_interp[i]/self.signal[i]]
        print(self.rand_error_matrix)

        return self.rand_error_matrix


    def sys_err_vector(self, error_type: int):
        '''Composes delta_y error vector for systematic errors.

        Parameters:
            self: contains configuration details from the initialization.
            self.sys_errors: array containing [total error, non-linearity, stray light,
                crosstalk, flat-field, bad pixel, keystone/smile, memory, striping].
            error_type: integer between 1 and 8 inclusive that indexes the type of 
                error in self.sys_errors

        Returns:
            delta_y: Systematic error vector.
        '''
        # systematic errors assumed constant across spectral range for now
        delta_y = np.full((len(self.wave_meas), 1), self.sys_errors[error_type])

        sys_error_types = ["total", "non-linearity", "stray light", "cross-talk", 
            "flat-field", "bad pixel", "smile/keystone", "striping", "memory effect"]

        print("\n" + sys_error_types[error_type] + "systematic error:")
        print(delta_y)
        print()

        return delta_y


    def error_covariance(self):
        '''Composes Sy error covariance matrix for random errors.

        Args:
            self: contains configuration details from the initialization.
            self.rand_error_matrix: array containing [total error, dark current, 
                readout, quantization, photon noise] by spectral band.

        Returns:
            S_y: Random error covariance matrix.
        '''
        meas_err_vector = np.array([band[0] for band in self.rand_error_matrix])
        meas_err_vector = np.transpose(meas_err_vector[np.newaxis])
        # print(meas_err_vector)

        S_y = np.cov(meas_err_vector, bias=True)
        # print(S_y.shape)
        print(S_y)

        return S_y
