"""
forward.py  

Generates forward model spectrum for methane, carbon dioxide, and water vapour.

Author(s): Jochen Landgraf, Adyn Miles, Shiqi Xu, Rosie Liang
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import lib.libRT as libRT
import lib.hapi as hp

class Forward:
    def __init__(self, cfg):
        """
        Initializes spectral and atmospheric grids, as well as useful constants for forward model.

        Args:
            cfg: the configuration arguments generated from config.py
        
        Returns:
            None

        """
        self.cfg = cfg
        self.recalc_xsec = self.cfg.recalc_xsec

        ### Spectral Grid Initialization
        self.wave_edges = [self.cfg.spectral_lower, self.cfg.spectral_upper]
        # Extend the spectral grid by 5nm on each end
        self.wave_extend = 5
        self.dwave = 0.001 
        self.wave_lbl =  np.arange(self.wave_edges[0]-self.wave_extend,self.wave_edges[1]+self.wave_extend,self.dwave)

        ### Atmospheric Grid Initialization
        self.dzlay= 1.E3          #geometrical thickness of the model atmosphere  
        self.nlay = 20            #number of layers
        self.nlev = self.nlay + 1      #number of levels

        self.zlay = (np.arange(self.nlay-1,-1,-1)+0.5)*self.dzlay  #altitude of layer midpoint
        self.zlev = np.arange(self.nlev-1,-1,-1)*self.dzlay        #altitude of layer interfaces = levels 

        self.sza_meas = self.cfg.sza       #solar zenith angle (degrees)
        self.vza_meas = self.cfg.vza       #viewing zenith angle (degrees)

        ### Atmospheric Constants
        self.mu0 = np.cos(np.deg2rad(self.sza_meas))
        if self.mu0 < 0.: print("ERROR! main: solar cosine < 0; needs to be > 0 by definition.")
        self.muv = np.cos(np.deg2rad(self.vza_meas))
        self.psurf = 1013                           #surface pressure (HPa)

        ### Optics Information
        self.fwhm = self.cfg.fwhm
        self.samp_dist = 0.5 * self.fwhm
        self.wave_meas = np.arange(self.wave_edges[0],self.wave_edges[1],self.samp_dist)



    def get_solar_model_spectrum(self, filen):
        """
        Find the solar spectrum from an inputted .dat file

        Args:
            self.wave_lbl: spectral grid
            self.cfg.filen: solar spectrum .dat file path.
        
        Returns:
            self.sun: solar model spectrum

        """

        h_pl      = 6.626E-34    # Planck constant [Js]
        c_li      = 2.9979E+8    # velocity of light [m/s]
        data = np.genfromtxt(filen,skip_header=3)   
        wave_data  = 1.E7/data[:,0]
        spec_data  = data[:,1]         #[W/(m2 mu)]
        conv1      = 1.E-3             #[W/(m2 nm)]
        conv2      = wave_data/(h_pl*c_li)*1.E-9 #[W sec / photon]
        spec_data  = conv1*conv2*spec_data
        
        self.sun = np.interp(self.wave_lbl,wave_data,spec_data)
    
        return(self.sun)


    def slit_conv(self, rad_lbl):

        """

        Args:
            self.fwhm: full width half maximum (spectral resolution)
            self.wave_lbl: line-by-line spectral grid
            self.wave_meas: sampling spectral grid
            rad_lbl: line-by-line radiance 
        
        Returns:
            self.rad_conv: 

        """
    
        dmeas = self.wave_meas.size
        dlbl  = self.wave_lbl.size
        slit  = np.zeros(shape=(dmeas,dlbl))
        const = self.fwhm**2/(4*np.log(2))
        for l,wmeas in enumerate(self.wave_meas):
            wdiff = self.wave_lbl - wmeas
            slit[l,:] = np.exp(-wdiff**2/const)
            slit[l,:] = slit[l,:]/np.sum(slit[l,:])

        self.rad_conv = slit.dot(rad_lbl)
        return(self.rad_conv)

    def get_atm_params(self):
        """
        gets solar model, molecular cross-section information, surface albedo properties.

        Args:
            self.wave_lbl: spectral grid
            self.zlay: altitudes of layer midpoints
            self.zlev: altitudes of layer interfaces (levels)
            self.psurf: surface pressure
        
        Returns:
            self.surface: albedo profile
            self.molec: molecular cross-section data for CH4, CO2, H2O
            self.atm: atmospheric profile
            self.sun_lbl: solar spectrum

        """
        file_sun = os.path.normpath(self.cfg.solar_spectrum)
        file_atm = os.path.normpath(self.cfg.atm_model)

        self.sun_lbl = self.get_solar_model_spectrum(file_sun)
        self.atm = libRT.atmosphere_data(self.zlay, self.zlev, self.psurf)
        self.atm.get_data_AFGL(file_atm)

        iso_ids=[('CH4',32),('CO2',7),('H2O',1)]   #see hapi manual  sec 6.6
        self.molec = libRT.molecular_data(self.wave_lbl)
        self.molec.get_data_HITRAN(os.path.normpath(self.cfg.output_folder),iso_ids)

        self.surface = libRT.surface_prop(self.wave_lbl)
        self.surface.get_albedo_flat(.2)

        return self.surface, self.molec, self.atm, self.sun_lbl

    def opt_properties(self):
        """
        Creates a struct containing optical system properties

        Args:
            self.wave_lbl: spectral grid
            self.zlay: altitudes of layer midpoints
            self.molec: molecular cross-section data for CH4, CO2, H2O
            self.atm: atmospheric model
        
        Returns:
            self.optics: optical properties class

        """

        # Calculate optical properties
        pklfile = os.path.normpath(self.cfg.pickle_file)
        #If pickle file exists read from file
        #if os.path.exists(pklfile):
        if self.recalc_xsec == True:
            # Init class with optics.prop dictionary
            self.optics=libRT.optic_abs_prop(self.wave_lbl, self.zlay)
            # Rayleigh optical depth, single scattering albedo, phase function
            # optics.cal_rayleigh(rayleigh, atm, mu0, muv, deltaphi)
            # Molecular absorption optical properties
            self.optics.cal_molec(self.molec, self.atm)
            # Dump optics.prop dictionary into temporary pkl file
            pkl.dump(self.optics.prop,open(pklfile,'wb'))
        else:
            # Init class with optics.prop dictionary
            self.optics=libRT.optic_ssc_prop(self.wave_lbl, self.zlay)
            # Read optics.prop dictionary from pickle file
            self.optics.prop = pkl.load(open(pklfile,'rb'))
        
        return self.optics

    def plot_transmittance(self):
        """
        Plots transmittance for CH4, CO2, H2O, and their combined transmittance

        Args:
            self.optics: optical properties class
            self.surface: albedo profile
            self.mu0: cosine of solar zenith angle
            self.muv: cosine of viewing zenith angle
            self.fwhm: full-width half maximum
            self.wave_lbl: line-by-line spectral grid
            self.wave_meas: sampling spectral grid

        
        Returns:
            self.rad_conv_tot: total radiance
            self.rad_conv_ch4: methane radiance
            self.rad_conv_co2: carbon dioxide radiance
            self.rad_conv_h2o: water vapour radiance

        """
        self.optics.combine('molec_32','molec_07','molec_01')

        self.rad_trans_tot, self.dev_rad = libRT.transmission(self.optics, self.surface, self.mu0, self.muv, 'molec_32')
        self.rad_conv_tot = self.slit_conv(self.rad_trans_tot)

        self.optics.combine('molec_32')
        self.rad_trans_ch4 = libRT.transmission(self.optics, self.surface, self.mu0, self.muv)
        self.rad_conv_ch4 = self.slit_conv(self.rad_trans_ch4)

        self.optics.combine('molec_07')
        self.rad_trans_co2 = libRT.transmission(self.optics, self.surface, self.mu0, self.muv)
        self.rad_conv_co2 = self.slit_conv(self.rad_trans_co2)

        self.optics.combine('molec_01')
        self.rad_trans_h2o = libRT.transmission(self.optics, self.surface, self.mu0, self.muv)
        self.rad_conv_h2o = self.slit_conv(self.rad_trans_h2o)

        fig = plt.figure(figsize=[15, 10])
        plt.subplot(2,2,1)
        plt.plot(self.wave_meas, self.rad_conv_tot)
        plt.ylim([0.035,0.041])
        plt.title('total transmission')

        plt.subplot(2,2,2)
        plt.plot(self.wave_meas, self.rad_conv_ch4,color = 'green')
        plt.ylim([0.035,0.041])
        plt.title('CH$_4$ transmission')

        plt.subplot(2,2,3)
        plt.plot(self.wave_meas, self.rad_conv_co2,color = 'orange')
        plt.ylim([0.035,0.041])
        plt.title('CO$_2$ transmission')
        plt.xlabel('$\lambda$ [nm]')

        plt.subplot(2,2,4)
        plt.title('H$_2$O transmission')
        plt.plot(self.wave_meas, self.rad_conv_h2o,color = 'red')
        plt.ylim([0.0409,0.04093])
        plt.xlabel('$\lambda$ [nm]')

        plt.show()

        return self.rad_conv_tot, self.rad_conv_ch4, self.rad_conv_co2, self.rad_conv_h2o