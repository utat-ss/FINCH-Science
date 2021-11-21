"""
optim.py  

Sets up equations and runs optimization for the LEA program.

Author(s): Adyn Miles, Shiqi Xu, Rosie Liang
"""

import numpy as np

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
                                + (self.ffr)**2
                                + (self.bad_pixels)**2
                                + (self.crosstalk)**2
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

        Args:
            self: contains configuration details from the initialization.

        Returns:
            self.rand_errors: an array containing [total error, dark current, 
                readout, quantization, photon noise] estimates.
        '''
        self.area_detector = self.cfg.x_pixels * (self.cfg.pixel_pitch / 1e6) \
                     * self.cfg.y_pixels * (self.cfg.pixel_pitch / 1e6)
        self.photon_noise = pn.photon_noise(self.cfg.fwhm)
        self.quant_noise = self.cfg.well_depth / (2**(self.cfg.dynamic_range) \
                     * np.sqrt(12))
        self.dark_current = self.cfg.dark_current * (1e-9) * (6.242e18) \
                     * (self.area_detector * 1e2 * 1e2)
        self.dark_noise = self.dark_current * self.cfg.t_int
        self.readout_noise = self.cfg.readout_noise
        self.spec_res_series = np.arange(self.cfg.spectral_lower, self.cfg.spectral_upper, self.cfg.fwhm)

        self.rand_error_matrix = np.zeros((len(self.spec_res_series), 5))

        for i in range(0, len(self.spec_res_series)):
            self.signal = self.photon_noise[i]**2
            self.rand_error = np.sqrt((self.signal) + (self.readout_noise)**2 \
                     + (self.quant_noise)**2 + (self.dark_noise))
            self.rand_error_matrix[i, :] = [self.rand_error/self.signal, np.sqrt(self.dark_noise)/self.signal, 
                            self.readout_noise/self.signal, self.quant_noise/self.signal, 
                            self.photon_noise[i]/self.signal]

        print(self.rand_error_matrix)

        return self.rand_error_matrix


    def error_covariance(self, random_errors):
        '''Composes Sy error covariance matrix for random errors.

        Args:
            self: contains configuration details from the initialization.
            random_errors: array containing [total error, dark current, 
                readout, quantization, photon noise].

        Returns:
            S_y: Random error covariance matrix.
        '''
        meas_err_vector = np.full((len(self.wave_meas), 1), random_errors[0])
        S_y = np.cov(meas_err_vector)

        return S_y


    def sys_err_vector(self, sys_errors, error_type: int):
        '''Composes delta_y error vector for systematic errors.

        Parameters:
            self: contains configuration details from the initialization.
            sys_errors: array containing [total error, non-linearity, stray light, 
                crosstalk, flat-field, bad pixel, keystone/smile, memory, striping].
            error_type: integer between 1 and 8 inclusive that indexes the type of 
                error in self.sys_errors

        Returns:
            delta_y: Systematic error vector.
        '''
        # systematic errors assumed constant across spectral range for now
        delta_y = np.full((len(self.wave_meas), 1), sys_errors[error_type])

        return delta_y
