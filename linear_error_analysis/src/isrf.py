"""
isrf.py

Convolves the transmittance spectrum with the instrument spectral response function.
This file is included in case it becomes useful in the future, but at the moment the
slit_conv() function inside of forward.py is being used to perform the convolution.

Author(s): Adyn Miles, Shiqi Xu, Rosie Liang
"""

import matplotlib.pyplot as plt
import numpy as np

import libs.hapi as hp


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
        self.wave_extend = 5
        self.wave_meas = np.arange(
            self.wave_edges[0], self.wave_edges[1], self.samp_dist
        )

    def define_isrf(self, show_fig=True):
        """Defines instrument spectral response function based on a simple Gaussian.
        Converts from nm to cm^(-1).

        Args:
            self.cfg.fwhm: full-width-half-maximum, in wavelength.
            self.wave_meas: spectral grid, in wavelength.

        Returns:
            self.isrf: the isrf curve, measured at each wavenumber.
        """
        # Convert wavelength to wavenumber
        self.wave_meas = np.flip(1e7 / (self.wave_meas))
        self.fwhm = self.wave_meas[2] - self.wave_meas[0]
        # Add two more values, one on each end of the spectral grid, so that after
        # being cut by convolveSpectrum, the spectral grid is the correct size.
        self.wave_meas = np.append(
            self.wave_meas, self.wave_meas[-1] + (0.5 * self.fwhm)
        )
        self.wave_meas = np.insert(
            self.wave_meas, 0, self.wave_meas[0] - (0.5 * self.fwhm)
        )
        # ISRF function sourced from HAPI manual found at ../doc/hapi_manual.pdf
        self.isrf = (
            1
            / (self.fwhm)
            * ((np.sin((np.pi / self.fwhm) * self.wave_meas)) ** 2)
            / ((np.pi / self.fwhm) * self.wave_meas) ** 2
        )

        plt.plot(self.wave_meas, self.isrf)
        plt.title("Instrument Spectral Response Function")
        if show_fig == True:
            plt.show()

        return self.isrf

    def convolve_isrf(self, radiance, wave_meas, show_fig=False):
        """Convolve instrument spectral response function with forward model
        radiance.

        Args:
            radiance: the radiance calculated over the spectral grid
            wave_meas: the spectral grid
            show_fig: set to True if output plots are desired.

        Returns:
            self.isrf_conv: ISRF convolved with forward model radiance at each
            point in the spectral grid.
        """

        # Convert wavelength to wavenumber
        wave_meas = np.flip(1e7 / (wave_meas))
        self.fwhm = wave_meas[2] - wave_meas[0]
        # Add two more values, one on each end of the spectral grid, so that after
        # being cut by convolveSpectrum, the spectral grid is the correct size.
        wave_meas = np.append(
            wave_meas, wave_meas[-1] + (0.5 * self.fwhm)
        )
        wave_meas = np.insert(
            wave_meas, 0, wave_meas[0] - (0.5 * self.fwhm)
        )

        # AF_wing refers to how much you would like to clip the spectral grid on
        # each side, with AF_wing = 0.0. 
        self.wave_meas_conv, self.isrf_conv, i1, i2, slit = hp.convolveSpectrum(
            wave_meas,
            radiance,
            SlitFunction=hp.SLIT_DIFFRACTION,
            AF_wing=0.0,
            Resolution=self.fwhm,
        )

        plt.plot(self.wave_meas_conv, self.isrf_conv)
        plt.title("ISRF Convolved with forward model response")
        plt.show()

        return self.wave_meas_conv, self.isrf_conv, i1, i2, slit
