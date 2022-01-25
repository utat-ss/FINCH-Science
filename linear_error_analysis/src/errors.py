"""
errors.py  

Collects all sources of random and systematic errors
and sets them up appropriately for the LEA program.

Author(s): Adyn Miles, Shiqi Xu, Rosie Liang
"""

import numpy as np
import scipy.interpolate as interp

import libs.photon_noise as pn


class Errors:
    def __init__(self, cfg, wave_meas):
        self.cfg = cfg
        self.wave_meas = wave_meas

    def sys_errors(self):
        """Performs calculations on inputted systematic errors from config.py.

        Args:
            self: contains configuration details from the initialization, as well
                  as the spectral grid.

        Returns:
            self.sys_errors: an array containing [total error, non-linearity, 
                stray light, crosstalk, flat-field, bad pixel, keystone/smile, 
                striping, memory] estimates relative to input signal.
        """
        self.nonlinearity = self.cfg.nonlinearity
        self.stray_light = np.sqrt( 
            (self.cfg.fo_reflectivity) ** 2 
            + (self.cfg.lens_reflectivity) ** 2
            + (self.cfg.mirror_reflectivity) ** 2
            + (self.cfg.ar_coatings) ** 2
            + (self.cfg.leakage) ** 2
            + (self.cfg.ghosting) ** 2 
        )
        self.ffr = self.cfg.uniformity
        self.bad_pixels = self.cfg.bad_pixels
        self.crosstalk = self.cfg.crosstalk
        self.key_smile = np.sqrt((self.cfg.keystone) ** 2 + (self.cfg.smile) ** 2)
        self.striping = self.cfg.striping
        self.memory = self.cfg.memory

        self.sys_error = np.sqrt( 
            (self.nonlinearity) ** 2
            + (self.stray_light) ** 2
            + (self.crosstalk) ** 2
            + (self.ffr) ** 2
            + (self.bad_pixels) ** 2
            + (self.key_smile) ** 2
            + (self.striping) ** 2
            + (self.memory) ** 2 
        )

        self.sys_errors = [
            self.sys_error,
            self.nonlinearity, 
            self.stray_light, 
            self.crosstalk, 
            self.ffr, 
            self.bad_pixels, 
            self.key_smile, 
            self.striping, 
            self.memory,
        ]
        print(self.sys_errors)

        return self.sys_errors

    def rand_errors(self):
        """Performs calculations on inputted random errors from config.py.

        See SNR Analysis Confluence page for breakdown of types of random error:
        http://spacesys.utat.ca/confluence/display/FIN/Signal-to-Noise+Ratio+Analysis

        Args:
            self: contains configuration details from the initialization.

        Returns:
            self.rand_error_matrix: a matrix containing [total error, dark current, 
                readout, quantization, photon noise] estimates relative to input signal.
        """
        self.area_detector = (
            self.cfg.x_pixels
            * (self.cfg.pixel_pitch / 1e6)
            * self.cfg.y_pixels
            * (self.cfg.pixel_pitch / 1e6)
        )

        # Interpolate photon noise based on the spectral range and resolution.
        self.photon_noise = np.array(pn.photon_noise(self.cfg.spectral_lower, self.cfg.spectral_upper, self.wave_meas, self.cfg.fwhm))
        self.quant_noise = self.cfg.well_depth / (
            2 ** (self.cfg.dynamic_range) * np.sqrt(12)
        )
        self.dark_current = (
            self.cfg.dark_current
            * (1e-9)
            * (6.242e18)
            * (self.area_detector * 1e2 * 1e2)
        )
        self.dark_noise = self.dark_current * self.cfg.t_int
        self.readout_noise = self.cfg.readout_noise

        self.rand_error_matrix = np.zeros((len(self.wave_meas), 5), dtype=object)

        self.signal = (self.photon_noise ** 2)*self.cfg.t_int/0.1667
        # note: self.dark_noise is standard deviation of the dark signal squared. 
        # When added in quadrature, this is just self.dark_noise.
        
        self.rand_error = np.sqrt(
            self.signal
            + self.readout_noise ** 2
            + self.quant_noise ** 2
            + self.dark_noise
        )

        for i in range(len(self.wave_meas)):
            self.rand_error_matrix[i, :] = [
                self.rand_error[i] / self.signal[i],
                np.sqrt(self.dark_noise) / self.signal[i],
                self.readout_noise / self.signal[i],
                self.quant_noise / self.signal[i],
                self.photon_noise[i] / self.signal[i],
            ]

        return self.rand_error_matrix

    def sys_err_vector(self, error_type: int):
        """Composes delta_y error vector for systematic errors.

        Parameters:
            self: contains configuration details from the initialization.
            self.sys_errors: array containing [total error, non-linearity, stray light,
                crosstalk, flat-field, bad pixel, keystone/smile, memory, striping].
            error_type: integer between 1 and 8 inclusive that indexes the type of 
                error in self.sys_errors

        Returns:
            delta_y: Systematic error vector containing [total error, non-linearity,
                stray light, crosstalk, flat-field, bad pixel, keystone/smile, memory, 
                striping]
        """
        # Systematic errors assumed constant across spectral range.
        delta_y = np.full((len(self.wave_meas), 1), self.sys_errors[error_type])

        sys_error_types = [
            "total",
            "non-linearity",
            "stray light",
            "cross-talk",
            "flat-field",
            "bad pixel",
            "smile/keystone",
            "striping",
            "memory effect",
        ]

        return delta_y

    def error_covariance(self):
        """Composes Sy error covariance matrix for random errors.

        Sy is of dimensions (i x i)

        i = number of points in the spectral grid (wave_meas)

        Args:
            self: contains configuration details from the initialization.
            self.rand_error_matrix: matrix containing [total error, dark current, 
                readout, quantization, photon noise] at each spectral grid point.

        Returns:
            S_y: Random error covariance matrix.
        """

        # The error covariance matrix is simply the square of the random errors
        # on the diagonal. There are assumed to be no correlation between bands.
        meas_err_vector = np.array([band[0] for band in self.rand_error_matrix])
        S_y = np.diag(np.square(meas_err_vector))

        return S_y
