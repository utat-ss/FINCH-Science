"""
forward.py  

Applies the forward model to get the optical depth of necessary greenhouse gases.

Author(s): Adyn Miles, Shiqi Xu, Rosie Liang
"""

import hapi as hp
import numpy as np
import matplotlib.pyplot as plt
import libRT as lrt

hp.db_begin('hitran_data')

hp.fetch('CH4', 6, 1, 5952, 6289.3)
hp.fetch('H2O', 1, 1, 5952, 6289.3)
hp.fetch('CO2', 2, 1, 5952, 6289.3)

class Forward:
    def __init__(self, cfg, molecule, spectral_range):
        """Fetches required isotope information for a specified molecule at a given spectral range and initializes the class.

        Args:
            cfg: the configuration arguments generated from config.py
            molecule: the name of the inputted molecule for analysis (options: CO2, H2O, CH4)
            spectral_range: a 2 element tuple [lower, upper] with the lower and upper wavelength bounds (in nm).
        
        Returns:
            structure containing the relevant molecule information.

        """

        # Initializing the hitran_data folder as the place to save fetched 
        hp.db_begin('hitran_data')
        # Converting the spectral range in nanometres to wavenumber
        sr_upper = (10**7)/(spectral_range[0])
        sr_lower = (10**7)/(spectral_range[1])

        # Vertical layering
        self.dzlay= 1.E3          #geometrical thickness of the model atmosphere  
        self.nlay = 20            #number of layers
        self.nlev = self.nlay + 1      #number of levels

        self.zlay = (np.arange(self.nlay-1,-1,-1)+0.5)*self.dzlay  #altitude of layer midpoint
        self.zlev = np.arange(self.nlev-1,-1,-1)*self.dzlay        #altitude of layer interfaces = levels 

        self.psurf = 985          
        # Average surface pressure 
        # Sourced from: https://en.wikipedia.org/wiki/Atmospheric_pressure#:~:text=The%20average%20value%20of%20surface%20pressure%20on%20Earth,to%20sea-level%20for%20locations%20above%20or%20below%20sea-level.

        # Observation geometry
        self.sza  = 50. #[deg]
        self.vza  = 30. #[deg]
        self.phiv = 0.
        self.phi0 = 0.
        self.mu0 = np.cos(np.deg2rad(self.sza))
        if self.mu0 < 0.: print("ERROR! main: solar cosine < 0; needs to be > 0 by definition.")
        self.muv      = np.cos(np.deg2rad(self.vza))
        self.deltaphi = np.deg2rad(self.phiv-self.phi0)

        if molecule.lower() == "ch4":
            self.molecule = hp.fetch('CH4', 6, 1, sr_lower, sr_upper)
            self.nu, self.sw1, self.gamma_air, self.gamma_self, self.n_air, self.delta_air = hp.getColumns('CH4', ['nu', 'sw', 'gamma_air', 'gamma_self', 'n_air', 'delta_air'])
            self.molecularMass = hp.molecularMass(6, 1)
        elif molecule.lower() == "h2o":
            self.molecule = hp.fetch('H2O', 1, 1, sr_lower, sr_upper)
            self.nu, self.sw1, self.gamma_air, self.gamma_self, self.n_air, self.delta_air = hp.getColumns('H2O', ['nu', 'sw', 'gamma_air', 'gamma_self', 'n_air', 'delta_air'])
            self.molecularMass = hp.molecularMass(1, 1)
        elif molecule.lower() == "co2":
            self.molecule = hp.fetch("CO2", 2, 1, sr_lower, sr_upper)
            self.nu, self.sw1, self.gamma_air, self.gamma_self, self.n_air, self.delta_air = hp.getColumns('CO2', ['nu', 'sw', 'gamma_air', 'gamma_self', 'n_air', 'delta_air'])
            self.molecularMass = hp.molecularMass(2, 1)

        self.cfg = cfg
        self.molecule_name = molecule.lower()

    def absorption_coeff(self):

        """Gets the absorption coefficient of the required molecule using hapi.py.

        Args:
        
        Returns:
            absorption coefficient for the considered molecule.

        """
        if self.cfg.altitude < 12:
            if self.molecule_name == "ch4":
                self.nu, self.coeff = hp.absorptionCoefficient_Lorentz(SourceTables='CH4', OmegaStep=0.01, Environment={'p':self.cfg.pressure, 'T':self.cfg.temperature})
            elif self.molecule_name == "h2o":
                self.nu, self.coeff = hp.absorptionCoefficient_Lorentz(SourceTables='H2O', OmegaStep=0.01, Environment={'p':self.cfg.pressure, 'T':self.cfg.temperature})
            elif self.molecule_name == "co2":
                self.nu, self.coeff = hp.absorptionCoefficient_Lorentz(SourceTables='CO2', OmegaStep=0.01, Environment={'p':self.cfg.pressure, 'T':self.cfg.temperature})
        else:
            if self.molecule_name == "ch4":
                self.nu, self.coeff = hp.absorptionCoefficient_Doppler(SourceTables='CH4', OmegaStep=0.01, Environment={'p':self.cfg.pressure, 'T':self.cfg.temperature})
            elif self.molecule_name == "h2o":
                self.nu, self.coeff = hp.absorptionCoefficient_Doppler(SourceTables='H2O', OmegaStep=0.01, Environment={'p':self.cfg.pressure, 'T':self.cfg.temperature})
            elif self.molecule_name == "co2":
                self.nu, self.coeff = hp.absorptionCoefficient_Doppler(SourceTables='CO2', OmegaStep=0.01, Environment={'p':self.cfg.pressure, 'T':self.cfg.temperature})
                
    def optical_depth(self):

        """Gets the optical depth for the given molecule and absorption features.

        Args:
        
        Returns:
            Optical depth using Rayleigh scattering (self.optics.cal_rayleigh())

        """
        self.wave = np.arange(self.cfg.spectral_lower, self.cfg.spectral_upper)
        self.optics = lrt.optic_ssc_prop(self.wave, self.zlay)
        
        # Read simulation atmosphere
        self.atm = lrt.atmosphere_data(self.zlay, self.zlev, self.psurf)
        self.atm.get_data_AFGL('atm_data/prof.AFGL.US.std')

        # Read Rayleigh scattering properties
        self.rayleigh = lrt.rayleigh_data(self.wave)
        self.rayleigh.get_data_Rayleigh('atm_data/depol.dat')

        self.surface = lrt.surface_prop(self.wave)
        self.surface.get_albedo_CUSTOM('atm_data/albedo.dat','vegetation')

        return self.optics.cal_rayleigh(self.rayleigh, self.atm, self.mu0, self.muv, self.deltaphi)

    def radiance(self):
        """Solve the radiative transfer equation and plot the result.

        Args:
        
        Returns:
            None.

        """
        # self.rad_ssc = lrt.single_scattering_LA(self.optics, self.surface, self.mu0, self.muv, self.deltaphi)
        self.nu, self.rad = hp.radianceSpectrum(self.nu, self.coeff, Environment={'l':1000., 'T':self.cfg.temperature})
        plt.plot(self.nu, self.rad)
        plt.title("Radiance of %s" % self.molecule_name.upper())
        plt.show()

    def transmittance(self):
        """Find transmittance and plot the result.

        Args:
        
        Returns:
            None.

        """
        # self.rad_ssc = lrt.single_scattering_LA(self.optics, self.surface, self.mu0, self.muv, self.deltaphi)
        self.nu, self.trans = hp.transmittanceSpectrum(self.nu, self.coeff, Environment={'l':1000., 'T':self.cfg.temperature})
        plt.plot(self.nu, self.trans)
        plt.title("Transmittance of %s" % self.molecule_name.upper())
        plt.show()
        
