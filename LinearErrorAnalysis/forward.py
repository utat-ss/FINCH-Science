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

# Add user-defined ../lib to system path
lib_path = os.path.abspath(os.path.join('../lib'))
if lib_path not in sys.path:
    sys.path.append(lib_path)
import libRT as libRT
import hapi as hp

# Add user-defined ../data to system path
data_path = os.path.abspath(os.path.join('../data'))
if data_path not in sys.path:
    sys.path.append(data_path)

class Forward:
    def __init__(self, cfg):
        self.cfg = cfg


    def get_solar_model_spectrum(self, filen, wave_lbl):

        h_pl      = 6.626E-34    # Planck constant [Js]
        c_li      = 2.9979E+8    # velocity of light [m/s]
        data = np.genfromtxt(filen,skip_header=3)   
        wave_data  = 1.E7/data[:,0]
        spec_data  = data[:,1]         #[W/(m2 mu)]
        conv1      = 1.E-3             #[W/(m2 nm)]
        conv2      = wave_data/(h_pl*c_li)*1.E-9 #[W sec / photon]
        spec_data  = conv1*conv2*spec_data
        
        sun = np.interp(wave_lbl,wave_data,spec_data)
    
        return(sun)


    def slit_conv(self, fwhm, wave, wave_meas,rad_lbl):
    
        dmeas = wave_meas.size
        dlbl  = wave.size
        slit  = np.zeros(shape=(dmeas,dlbl))
        const = fwhm**2/(4*np.log(2))
        for l,wmeas in enumerate(wave_meas):
            wdiff = wave - wmeas
            slit[l,:] = np.exp(-wdiff**2/const)
            slit[l,:] = slit[l,:]/np.sum(slit[l,:])

        rad_conv = slit.dot(rad_lbl)
        return(rad_conv)

recalc_xsec = True
recalc_xsec = False
######################################################
#get tropomi data 
wave_edges = [1590,1680]

# Spectral grid
wave_extend = 5 #nm
dwave = 0.001
wave_lbl =  np.arange(wave_edges[0]-wave_extend,wave_edges[1]+wave_extend,dwave) #nm

# Vertical layering
dzlay= 1.E3          #geometrical thickness of the model atmosphere  
nlay = 20            #number of layers
nlev = nlay + 1      #number of levels

zlay = (np.arange(nlay-1,-1,-1)+0.5)*dzlay  #altitude of layer midpoint
zlev = np.arange(nlev-1,-1,-1)*dzlay        #altitude of layer interfaces = levels 

sza_meas = 50. 
vza_meas = 0.

# Observation geometry
mu0 = np.cos(np.deg2rad(sza_meas))
if mu0 < 0.: print("ERROR! main: solar cosine < 0; needs to be > 0 by definition.")
muv      = np.cos(np.deg2rad(vza_meas))

######################################################
# read solar reference spectrum
file_sun = '../data/solar_spectrum_merged.dat'
sun_lbl  = get_solar_model_spectrum(file_sun, wave_lbl)

######################################################
# Read model atmosphere
psurf = 250 #hPa
atm = libRT.atmosphere_data(zlay, zlev, psurf)
atm.get_data_AFGL('../data/prof.AFGL.US.std')
#atm.get_data_ECMWF_ads_egg4('../data/ECMWF-ADS-EGG4_2016-monthly-means.nc', 6, 8., 49.)

# Set path for HITRAN database operations
######################################################
# Download molecular absorption parameter

iso_ids=[('CH4',32),('CO2',7),('H2O',1)]   #see hapi manual  sec 6.6
#iso_ids=[('CH4',32)]   #see hapi manual  sec 6.6
#iso_ids=[('CO2',07)]   #see hapi manual  sec 6.6
#iso_ids=[('H2O',1)]   #see hapi manual  sec 6.6
molec = libRT.molecular_data(wave_lbl)
molec.get_data_HITRAN('../data/',iso_ids)

# Read simulation surface properties
surface = libRT.surface_prop(wave_lbl)
surface.get_albedo_flat(.2)
#surface.get_albedo_flat(0.165)

#check cross sections

#nu,xs = hp.absorptionCoefficient_Voigt(SourceTables='ID37_WV00753-00776', Environment={'p':pi/PSTD, 'T':Ti},WavenumberStep=nu_samp)
######################################################
# Calculate optical properties
pklfile = '../tmp/optics_prop.pkl'
#If pickle file exists read from file
#if os.path.exists(pklfile):
if recalc_xsec == True:
    # Init class with optics.prop dictionary
    optics=libRT.optic_abs_prop(wave_lbl, zlay)
    # Rayleigh optical depth, single scattering albedo, phase function
    # optics.cal_rayleigh(rayleigh, atm, mu0, muv, deltaphi)
    # Molecular absorption optical properties
    optics.cal_molec(molec, atm)
    # Dump optics.prop dictionary into temporary pkl file
    pkl.dump(optics.prop,open(pklfile,'wb'))
else:
    # Init class with optics.prop dictionary
    optics=libRT.optic_ssc_prop(wave_lbl, zlay)
    # Read optics.prop dictionary from pickle file
    optics.prop = pkl.load(open(pklfile,'rb'))

#for key in optics.prop:
#    print(key)
fwhm = 0.5
samp_dist = 0.25
wave_meas = np.arange(wave_edges[0],wave_edges[1],samp_dist)

# Combine all optical properties to composite properties
optics.combine('molec_32','molec_07','molec_01')

rad_trans_tot, dev_rad = libRT.transmission(optics, surface, mu0, muv, 'molec_32')
rad_conv_tot = slit_conv(fwhm, wave_lbl, wave_meas, rad_trans_tot)

optics.combine('molec_32')
rad_trans_ch4 = libRT.transmission(optics, surface, mu0, muv)
rad_conv_ch4 = slit_conv(fwhm, wave_lbl, wave_meas, rad_trans_ch4)

optics.combine('molec_07')
rad_trans_co2 = libRT.transmission(optics, surface, mu0, muv)
rad_conv_co2 = slit_conv(fwhm, wave_lbl, wave_meas, rad_trans_co2)

optics.combine('molec_01')
rad_trans_h2o = libRT.transmission(optics, surface, mu0, muv)
rad_conv_h2o = slit_conv(fwhm, wave_lbl, wave_meas, rad_trans_h2o)

fig = plt.figure(figsize=[15, 10])
plt.subplot(2,2,1)
plt.plot(wave_meas, rad_conv_tot)
plt.ylim([0.035,0.041])
plt.title('total transmission')

plt.subplot(2,2,2)
plt.plot(wave_meas,rad_conv_ch4,color = 'green')
plt.ylim([0.035,0.041])
plt.title('CH$_4$ transmission')

plt.subplot(2,2,3)
plt.plot(wave_meas,rad_conv_co2,color = 'blue')
plt.ylim([0.035,0.041])
plt.title('CO$_2$ transmission')
plt.xlabel('$\lambda$ [nm]')

plt.subplot(2,2,4)
plt.title('H$_2$O transmission')
plt.plot(wave_meas,rad_conv_h2o,color = 'blue')
plt.ylim([0.0409,0.04093])
plt.xlabel('$\lambda$ [nm]')

plt.show()
sys.exit()