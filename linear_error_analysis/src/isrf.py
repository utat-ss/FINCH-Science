"""
isrf.py  

Convolves the transmittance spectrum with the instrument spectral response function.

Author(s): Adyn Miles, Shiqi Xu, Rosie Liang
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import lib.hapi as hp

class ISRF:

    def __init__(self, cfg):
        """Initializes instrument spectral response function class.

        Args:
            cfg: the configuration arguments generated from config.py.
        
        Returns:
            None
        """
        self.cfg = cfg

        self.wave_edges = [self.cfg.spectral_lower, self.cfg.spectral_upper]

        ### Optics Information
        self.fwhm = self.cfg.fwhm
        self.samp_dist = 0.5 * self.fwhm
        self.wave_meas = np.arange(self.wave_edges[0], self.wave_edges[1], 
                                   self.samp_dist)


    def define_isrf(self, show_fig=True):
        """Defines instrument spectral response function based on a simple Gaussian.
        Converts from nm to cm^(-1).

        Args:
            self.cfg.fwhm: full-width-half-maximum, in wavelength.
            self.wave_meas: spectral grid, in wavelength.
        
        Returns:
            self.isrf: the isrf curve, in wavenumber.
        """
        self.wave_meas = 1e7/(self.wave_meas)
        self.wave_meas = np.flip(self.wave_meas)
        self.fwhm = self.wave_meas[2] - self.wave_meas[0]
        self.isrf = 1/(self.fwhm) * ((np.sin((np.pi/self.fwhm)*self.wave_meas))**2) \
                                  / ((np.pi/self.fwhm)*self.wave_meas)**2

        plt.plot(self.wave_meas, self.isrf)
        plt.title("Instrument Spectral Response Function")
        if show_fig == True:
            plt.show()

        return self.isrf


    def convolve_isrf(self, transmittance, show_fig=True):
        """Convolve instrument spectral response function with forward model 
        transmittance.

        Args:
            self.isrf: the isrf curve.
            self.transmittance: total transmittance curve from the forward model.

        Returns:
            self.isrf_conv: TODO
        """
        self.wave_meas, self.isrf_conv, i1, i2, slit = hp.convolveSpectrum(
                    self.wave_meas, transmittance, 
                    SlitFunction=hp.SLIT_DIFFRACTION, Resolution=self.fwhm)

        plt.plot(self.wave_meas, self.isrf_conv)
        plt.title("ISRF Convolved with forward model response")
        if show_fig == True:
            plt.show()

        return self.isrf_conv
