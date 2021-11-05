"""
###########################################################
# Heidelberg University, Institute of Environmental Physics
# RT lecture
# contact: andre.butz@iup.uni-heidelberg.de
###########################################################
# CONTAINS
# function clausius_clapeyron(T)
# function phase_HG(g,theta)
# function planck(T,lam)
# function read_sun_spectrum_S5P(filename)
# function read_sun_spectrum_TSIS1HSRS(filename)
# function single_scattering_LA(optics, surface, mu0, muv, phi)
# class atmosphere_data()
# class molecular_data()
# class optic_ssc_prop()
# class rayleigh_data()
# class StopExecution(Exception)
# class surface_prop()
###########################################################
"""
import os
import sys
import math
import numpy as np
import scipy.interpolate as interpolate
import netCDF4 as nc
import miepython as mie
import hapi as hp
import matplotlib.pyplot as plt
import bisect
###########################################################
# Some global constants
hplanck = 6.62607015E-34  # Planck's constant [J/s]
kboltzmann = 1.380649E-23 # Boltzmann's constant [J/K]
clight = 2.99792458E8     # Speed of light [m/s]
e0 = 1.6021766E-19        # Electron charge [C]
g0 = 9.80665              # Standard gravity at ground [m/s2]
NA = 6.02214076E23        # Avogadro number [#/mol]
Rgas = 8.314462           # Universal gas constant [J/mol/K]
XO2  = 0.2095             # Molecular oxygen mole fraction [-]
MDRYAIR = 28.9647E-3      # Molar mass dry air [kg/mol]
MH2O = 18.01528E-3        # Molar mass water [kg/mol]
MCO2 = 44.01E-3           # Molar mass CO2 [kg/mol]
MCH4 = 16.04E-3           # Molar mass CH4 [kg/mol]
LH2O = 2.5E6              # Latent heat of condensation for water [J/kg]
PSTD = 1013.25            # Standard pressure [hPa]

###########################################################
def find_nearest(array, value):
   array = np.asarray(array)
   idx = (np.abs(array - value)).argmin()
   return idx

def clausius_clapeyron(T):
    """
    # Calculates saturation water vapor pressure according to
    # Clausius-Clapeyron equation, exponential approximation
    #
    # input: temperature [K]
    # ouput: saturation water vapor pressure [hPa]
    """
    # check whether input is in range
    while True:
        if T > 0.:
           break
        else:
           print("ERROR! clausius_clapeyron: input out of range.")
           raise StopExecution
            
    T0 = 273.16 # ref. temperature[K], triple point
    p0 = 611.66 # ref. pressure [Pa], triple point
    
    # Saturation water vapor pressure according to
    # Clausius-Clapeyron equation, exponential approximation
    pS = p0 * np.exp(-LH2O*MH2O/Rgas*(1./T - 1./T0))
    
    return pS/100. # [hPa]

###########################################################
def phase_HG(g,theta):
    """
    # Heyney Greenstein phase function
    #
    # in: asymmetry paramter g [-1,1], scattering angle theta [degree]
    # out: Heyney Greenstein phase function [normalized to 4pi]
    """
    # check whether input is in range
    while True:
        if -1 <= g <= 1. and 0. <= theta <= 360.:
           break
        else:
           print("ERROR! phase_HG: input out of range.")
           raise StopExecution
    
    phase = (1 - g**2)/(1 + g**2 - 2* g * math.cos(theta*np.deg2rad))**(1.5)

    return phase  

###########################################################
def planck(T,lam):
    """
    # Planck spectral radiance at temperature T
    #
    # in: temperature T [K], wavelength lam [m]
    # out: Planck spectral radiance B [W m-2 m-1 sr-1]
    """
    # check whether input is in range
    while True:
        if T > 0. and lam > 0.:
           break
        else:
           print("ERROR! planck: input out of range.")
           raise StopExecution
           
    h = hplanck # Planck's constant [J/s]
    kb = kboltzmann # Boltzmann's constant [J/K]
    c = clight # Speed of light [m/s]
    
    B = 2*h*c**2/lam**5/(np.exp(h*c/kb/T/lam)-1.)
    
    return B

###########################################################
def read_sun_spectrum_S5P(filename):
    """
    # Read sun spectrum Sentinel-5 Precursor (S5P) format
    #
    # in: filepath to solar spectrum
    # out: dictionary with wavelength [nm], irradiance [mW nm-1 m-2], irradiance [ph s-1 cm-2 nm-1]
    """
    # check whether input is in range
    while True:
        if os.path.exists(filename):
           break
        else:
           print("ERROR! read_spectrum_S5P: filename does not exist.")
           raise StopExecution
    
    # Read data from file
    raw = np.genfromtxt(filename,skip_header=42,unpack=True)

    # Write sun spectrum in dictionary
    sun = {};
    sun['wl']=raw[0,:]
    sun['mWnmm2']=raw[1,:]
    sun['phscm2nm']=raw[2,:]

    
    return sun

###########################################################
def read_sun_spectrum_TSIS1HSRS(filename):
    """
    # Read sun spectrum TSIS-1 HSRS, downloaded from 
    # https://lasp.colorado.edu/lisird/data/tsis1_hsrs, 
    # Coddington et al., GRL, 2021, https://doi.org/10.1029/2020GL091709
    # NETCDF format: 'pip install netCDF4'
    #
    # in: filepath to solar spectrum
    # out: dictionary with wavelength [nm], irradiance [W m-2 nm-1]
    """    
    # check whether input is in range
    while True:
        if os.path.exists(filename):
           break
        else:
           print("ERROR! read_spectrum_TSIS1HSRS: filename does not exist.")
           raise StopExecution
    
    # Open netcdf file
    ds = nc.Dataset(filename)
    #print(ds.variables)
    
    # Write sun spectrum in dictionary
    sun = {};
    sun['Wm2nm']=ds['SSI'][:] # Solar spectral irradiance [W m-2 nm-1]
    sun['wl']=ds['Vacuum Wavelength'][:] # Vacuum Wavelength [nm]
    
    ds.close
    
    return sun

###########################################################
def single_scattering_LA(optics, surface, mu0, muv, phi):
    """
    # Calculate single scattering solution given
    # geometry (mu0,muv,phi) using matrix algebra
    #
    # arguments:
    #            optics: optic_prop object
    #            surface: surface_prop object
    #            mu0: cosine of the solar zenith angle [-]
    #            muv: cosine of the viewing zenith angle [-]
    #            phi: relative azimuth angle [rad] [0...2pi]
    # returns:
    #            rad_scc: single scattering relative radiance [wavelength] [1/sr]
    """    
    while True:
        if 0. <= mu0 <= 1. and -1. <= muv <= 1. and 0. <= phi <= 2.*np.pi:
           break
        else:
           print("ERROR! single_scattering: input out of range.")
           raise StopExecution
    
    # Number of wavelengths and layers 
    nwave=optics.prop['taus'][:,0].size
    nlay=optics.prop['taus'][0,:].size
    
       
    # Total vertical optical thickness per layer (Delta tau_k) [nwave,nlay]
    tauk = optics.prop['taus'] + optics.prop['taua'] 
    # Overhead optical thickness on layer interfaces [nwave,nlay+1], 
    # last level is total optical depth at surface, 
    # first level is zero optical depth at TOA
    taukup = np.zeros([nwave,nlay+1]) 
    taukup[:,1:nlay+1] = np.cumsum(tauk, axis=1)
    # Threshold for treating mu-m_0 or phi (~cos(phi) for phi~0) as almucantar or direct-sun case
    eps_sun = 0.01
    # Threshold for validity of plane-parallel approximatin
    eps_pp = 0.34#0.34 #~20 degree above horizon
    
    
    # Branch into upward and downward radiance
    if muv > 0.+eps_pp: # upward
        # Some shortcuts for quantities that do not depend on height or wavelength
        mueffup = abs(1./mu0) + abs(1./muv)
        prefup = mu0/(4.*np.pi*(abs(mu0)+abs(muv)))
        prefsurf = mu0/np.pi
        # Term (1-exp(-Delta tau_k * (1/muv +1/mu0))) [nwave,nlay]
        expk = 1.-np.exp(-tauk * mueffup)
        # Term (exp(-tau_k * (1/muv +1/mu0))) [nwave,nlay+1], last layer is surface
        expkup = np.exp(-taukup * mueffup)
      
        # Single scattering radiance [nwave]
        rad_atm = prefup*np.sum(np.multiply(np.multiply(np.multiply(optics.prop['ssa'],optics.prop['phase']),expk),expkup[:,0:nlay]),axis=1)
        # Add surface reflection term [nwave]
        rad_ssc = rad_atm + prefsurf*surface.alb*expkup[:,nlay] 
    
    elif muv < 0.-eps_pp: # downward
        
        # Branch into direct sun, almucantar, general case
        if abs(muv + mu0) < eps_sun and abs(phi) < eps_sun: # direct sun
                       
            # Solar solid angle [sr]
            omega_sun = 6.8E-5 
            # Term exp(-tau_tot/|mu_0|) [nwave]
            expkup = np.exp(-taukup[:,nlay] / abs(mu0))
            
            # Single scattering radiance = direct radiance [nwave]
            rad_ssc = 1./omega_sun*expkup
        
        elif abs(muv + mu0) < eps_sun and abs(phi) >= eps_sun: # almucantar plane
            
            # Some shortcuts for quantities that do not depend on height or wavelength
            prefdown = 1./(4.*np.pi*abs(muv))
            # Term exp(-tau_tot/|mu|) [nwave]
            expkup = np.exp(-taukup[:,nlay] / abs(muv))

            # Single scattering radiance [nwave]
            rad_ssc = prefdown*np.multiply(expkup,np.sum(np.multiply(np.multiply(optics.prop['ssa'],optics.prop['phase']),tauk),axis=1))
    
        else: # general case, no almucantar, no direct sun
            
            # Some shortcuts for quantities that do not depend on height or wavelength
            mueffdown = abs(1./mu0) - abs(1./muv)
            prefdown = mu0/(4.*np.pi*(abs(muv)-abs(mu0)))
            # Term (1-exp(-Delta tau_k * (1/|mu0| - 1/|muv|))) [nwave,nlay]
            expk = 1.-np.exp(-tauk * mueffdown)
            # Term (exp(-tau_k * (1/|mu0| - 1/|muv|))) [nwave,nlay+1], last layer is surface
            expkup = np.exp(-taukup * mueffdown)
            # Replace surface layer in expkup with exp(-tau_tot/|mu|) 
            expkup[:,nlay] = np.exp(-taukup[:,nlay] / abs(muv))
        
            # Single scattering radiance [nwave]
            rad_ssc = prefdown*np.multiply(expkup[:,nlay],np.sum(np.multiply(np.multiply(optics.prop['ssa'],optics.prop['phase'],expk),expkup[:,0:nlay]),axis=1))
    else:
        
        print("ERROR! single_scattering_LA: zenith angles close to the horizon lead to errors in plane parallel approx.")
        raise StopExecution
    
    return rad_ssc

###########################################################
class atmosphere_data:
    """
    # The atmosphere_data class collects data of
    # the thermodynamic state and the composition of
    # the atmosphere.
    #
    # CONTAINS
    # method __init__(self,zlay,zlev)
    # method get_data_AFGL(self,filename)
    # method get_data_ECMWF_ads_egg4(self,filename, month, longitude, latitude)
    """
    ###########################################################
    def __init__(self,zlay,zlev,psurf):
        """
        # init class
        #
        # arguments: 
        #            zlay: array of vertical height layers, mid-points [nlay] [m]
        #            zlev: array of vertical height levels, boundaries [nlev=nlay+1] [m]
        #            psurf: scalar of surface pressure [hPa]
        #
        # returns:   
        #            atmo: dictionary with atmospheric data
        #            atmo[zlev]: array of vertical height levels, boundaries [nlev=nlay+1] [m]
        #            atmo[zlay]: array of vertical height layers [nlay] [m]
        """
        self.atmo   = {}
        self.atmo['zlay']  = zlay
        self.atmo['zlev']  = zlev
        self.atmo['psurf'] = psurf
 
    ###########################################################
    def get_data_AFGL(self,filename):
        """
        # Read atmospheric data from AFGL database
        # file, interpolate on output height grid
        #
        # arguments: 
        #            filename: file with AFGL data
        # returns:   
        #            atmo[tlev]: temperature level profile [nlev] [K]
        #            atmo[tlay]: temperature layer profile [nlev] [K]
        #            atmo[plev: pressure level profile [nlev] [hPa]
        #            atmo[play: pressure layer profile [nlay] [hPa]
        #            atmo[AIR]: air partial column profile [nlay] [#/m^2]
        #            atmo[O3]: o3 partial column profile [nlay] [#/m^2]
        #            atmo[H2O]: h2o partial column profile [nlay] [#/m^2]
        #            atmo[CO2]: co2 partial column profile [nlay] [#/m^2]
        #            atmo[NO2]: no2 partial column profile [nlay] [#/m^2]
        #            atmo[O2]: o2 partial column profile [nlay] [#/m^2]
        """
        #some constants
        Rgas = 8.3144598    #universal gas constant [J/(mol*K)]
        grav = 9.80665      #gravitational acceleration [m/s2]
        Mair = 0.0289644    #molar mass of Earth's air [kg/mol]
        Avog = 6.022E23     #Avogadro constant [part./mol]
        #check whether input is in range
        while True:
            if os.path.exists(filename):
               break
            else:
               print("ERROR! atmosphere_data.get_data_AFGL: file does not exist.")
               raise StopExecution
        
        # Read AFGL file
        atm_in = np.genfromtxt(filename,skip_header=2) 

        zalt_in  = atm_in[:,0]*1.E3         # height [km] -> [m]
        press_in = atm_in[:,1]              # pressure [hPa]
        temp_in  = atm_in[:,2]              # temperature [K]
        air_in   = atm_in[:,3]              # air number density [#/cm^3] 
        o3_in    = atm_in[:,4]/air_in       # o3 number density -> mole fraction [-]
        o2_in    = atm_in[:,5]/air_in       # o2 number density -> mole fraction [-]
        h2o_in   = atm_in[:,6]/air_in       # h2o number density -> mole fraction [-]
        co2_in   = atm_in[:,7]/air_in       # co2 number density -> mole fraction [-]
        no2_in   = atm_in[:,8]/air_in       # no2 number density -> mole fraction [-]
        nlev_in  = zalt_in.size             # number of input levels

        sp = NA/(MDRYAIR*g0)*1.E2 # [#/m^2 * 1/hPa] air column above P is P*NA/Mair/g from p = m*g/area

        #truncate or extrapolate the AFGL profile depending on psurf
        print(press_in[press_in.size-1], self.atmo['psurf'])
        if press_in[press_in.size-1] < self.atmo['psurf']:
            #extrapolation required
            dz       = np.log(self.atmo['psurf']/press_in[press_in.size-1])*Rgas*temp_in[temp_in.size-1]/(grav*Mair)
            press_in = np.append(press_in, self.atmo['psurf'])
            zalt_in  = np.append(zalt_in-dz, 0.)
            temp_in  = np.append(temp_in, temp_in[temp_in.size-1]) 
            air_in   = np.append(air_in,press_in[press_in.size-1]*Avog*1.E-4/(Rgas *temp_in[temp_in.size-1]))
            o3_in    = np.append(o3_in, o3_in[o3_in.size-1]) 
            o2_in    = np.append(o2_in, o2_in[o2_in.size-1]) 
            h2o_in   = np.append(h2o_in, h2o_in[h2o_in.size-1]) 
            co2_in   = np.append(co2_in, co2_in[co2_in.size-1]) 
            no2_in   = np.append(no2_in, no2_in[no2_in.size-1]) 
            nlev_in  = nlev_in-1
        elif press_in[press_in.size-1] > self.atmo['psurf']:
            #interpolation required
            intv = np.searchsorted(press_in,self.atmo['psurf'])  # self.atmo['psurf'] is in the interval [press_in[intv], press_in[intv-1]]
            press_in = np.append(press_in[0:intv],self.atmo['psurf'])
            temp_in  = temp_in[0:intv+1]
            air_in   = np.append(air_in[0:intv],press_in[press_in.size-1]*Avog*1.E-4/(Rgas *temp_in[temp_in.size-1]))
            o3_in    = o3_in[0:intv+1]
            o2_in    = o2_in[0:intv+1]
            h2o_in   = h2o_in[0:intv+1]
            co2_in   = co2_in[0:intv+1]
            no2_in   = no2_in[0:intv+1]
            zalt_in  = zalt_in[0:intv+1]
            dz = np.log(press_in[press_in.size-1]/press_in[press_in.size-2])*Rgas*temp_in[temp_in.size-1]/(grav*Mair)
            zalt_in  = np.append(zalt_in[0:intv]-zalt_in[intv-1]+dz,0)
            
        # Interpolate temperature [K] on output layers
        # Flip arrays because our heights are descending
        # (from top to bottom), while np.interp expects ascending order
        self.atmo['tlev'] = np.flip(np.interp(np.flip(self.atmo['zlev']),np.flip(zalt_in),np.flip(temp_in)))
        self.atmo['tlay'] = np.flip(np.interp(np.flip(self.atmo['zlay']),np.flip(zalt_in),np.flip(temp_in)))
        
        # Calculate pressure [hPa] on output levels and layers
        self.atmo['plev'] = np.flip(np.interp(np.flip(self.atmo['zlev']),np.flip(zalt_in),np.flip(press_in)))
        self.atmo['play'] = np.flip(np.interp(np.flip(self.atmo['zlay']),np.flip(zalt_in),np.flip(press_in)))
        
        # Calculate the vertical column of air above pressure level 
        # and use this to calculate the partial vertical air columns per layer [#/m^2].
        # Partial columns have the advantage that multiplication with cross sections
        # yields optical depth.
        nlev = len(self.atmo['zlev'])
        sp = NA/(MDRYAIR*g0)*1.E2 # [#/m^2 * 1/hPa] air column above P is P*NA/Mair/g from p = m*g/area
        vc_air = sp*self.atmo['plev'] # air column [#/m^2] above pressure level
        self.atmo['AIR'] = (vc_air[1:nlev]-vc_air[0:nlev-1])#[#/m^2]
        self.atmo['AIR'][0] = vc_air[0]#[#/m^2] uppermost layer extends to infinity in terms of number of molecules

        # Interpolate mole fractions on output height grid
        # and then calculate partial columns per layer [#/m^2]
        # ozone
        self.atmo['O3']  = np.flip(np.interp(np.flip(self.atmo['zlay']),np.flip(zalt_in),np.flip(o3_in)))*self.atmo['AIR']
        # water vapor
        self.atmo['H2O'] = np.flip(np.interp(np.flip(self.atmo['zlay']),np.flip(zalt_in),np.flip(h2o_in)))*self.atmo['AIR']
        # co2
        self.atmo['CO2'] = np.flip(np.interp(np.flip(self.atmo['zlay']),np.flip(zalt_in),np.flip(co2_in)))*self.atmo['AIR']
        # no2
        self.atmo['NO2'] = np.flip(np.interp(np.flip(self.atmo['zlay']),np.flip(zalt_in),np.flip(no2_in)))*self.atmo['AIR']
        # o2 use a constant mixing ratio
        self.atmo['O2']  = XO2*self.atmo['AIR']

    ###########################################################
    def get_data_ECMWF_ads_egg4(self,filename, month, longitude, latitude):
        """
        # Read atmospheric data provided by ECMWF ADS EGG4 run:
        # https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-ghg-reanalysis-egg4-monthly
        # 
        # Download script:
        # 
        # import cdsapi
        # c = cdsapi.Client()
        # c.retrieve(
        #     'cams-global-ghg-reanalysis-egg4-monthly',
        #     {
        #         'variable': [
        #             'carbon_dioxide', 'geopotential', 'methane',
        #             'relative_humidity', 'temperature',
        #         ],
        #         'pressure_level': [
        #             '1', '2', '3',
        #             '5', '7', '10',
        #             '20', '30', '50',
        #             '70', '100', '150',
        #             '200', '250', '300',
        #             '400', '500', '600',
        #             '700', '800', '850',
        #             '900', '925', '950',
        #             '1000',
        #         ],
        #         'year': '2016',
        #         'month': [
        #             '01', '02', '03',
        #             '04', '05', '06',
        #             '07', '08', '09',
        #             '10', '11', '12',
        #         ],
        #         'product_type': 'monthly_mean',
        #         'format': 'netcdf',
        #     },
        #     'download.nc'
        #
        # arguments: 
        #            filename: filepath to ECMWF netcdf file
        #            month: [01 ... 12]
        #            longitude [0 ... 360 degree]
        #            latitude [-90 ... 90 degree]
        # returns:   
        #            temp: temperature profile [nlay] [K]
        #            plev: pressure level profile [nlev] [hPa]
        #            atmo[tlev]: temperature level profile [nlev] [K]
        #            atmo[tlay]: temperature layer profile [nlev] [K]
        #            atmo[plev]: pressure level profile [nlev] [hPa]
        #            atmo[play]: pressure layer profile [nlay] [hPa]
        #            atmo[AIR]: air partial column profile [nlay] [#/m^2]
        #            atmo[H2O]: h2o partial column profile [nlay] [#/m^2]
        #            atmo[CO2]: co2 partial column profile [nlay] [#/m^2]
        #            atmo[CH4]: no2 partial column profile [nlay] [#/m^2]
        #            atmo[O2]: o2 partial column profile [nlay] [#/m^2]
        """    
        # check whether input is in range
        while True:
            if os.path.exists(filename) and 1 <= month <= 12 and -90. <= latitude <= 90. and 0. <= longitude <= 360.:
               break
            elif not os.path.exists(filename):
               print("ERROR! read_ecmwf_ads_egg4: filename does not exist.")
               raise StopExecution
            else:
               print("ERROR! read_ecmwf_ads_egg4: input out of range.")
               raise StopExecution
           
        # Open netcdf file
        ds = nc.Dataset(filename)
        #print(ds.variables)

        # Select month index, latitude/longitude index (next neighbour)
        itime = int(month-1)
        ilat = np.argmin(abs(ds['latitude'][:] - latitude))
        ilon = np.argmin(abs(ds['longitude'][:] - longitude))
        
        # ECMWF: Geopotential [m2 s-2] converted to height [m], approximate use of g0 
        z_in = np.array([d/g0 for d in ds['z'][itime,:,ilat,ilon]])
        nlev_in = z_in.size
        # ECMWF: Pressure [hPa]
        p_in = ds['level'][:]
        # ECMWF: Temperature [K]
        temp_in = ds['t'][itime,:,ilat,ilon]
        # ECMWF: Humidity [%] converted to water vapor mole fraction [mol/mol] via Clausius-Clapeyron equation
        pS = [clausius_clapeyron(Ti) for Ti in temp_in]
        # ECMWF: Mole fraction is partial pressure over dry total pressure, partial pressure is rel. hum. * sat. vapor pressure
        h2o_in = np.array([d/100.*pSi/(pi - d/100.*pSi) for d,pSi,pi in zip(ds['r'][itime,:,ilat,ilon],pS,p_in)])
        # ECMWF: Carbon dioxide mass mixing ratio [kg kg-1] converted to mole fraction [mol/mol]
        co2_in = np.array([d/MCO2*MDRYAIR for d in ds['co2'][itime,:,ilat,ilon]])
        # ECMWF: Methane mass mixing ratio [kg kg-1] converted to mole fraction [mol/mol]
        ch4_in = np.array([d/MCH4*MDRYAIR for d in ds['ch4'][itime,:,ilat,ilon]])
        ds.close
        
        # Interpolate temperature [K] on output layers
        # Flip arrays because our heights are descending
        # (from top to bottom), while np.interp expects ascending order
        self.atmo['tlay'] = np.flip(np.interp(np.flip(self.atmo['zlay']),np.flip(z_in),np.flip(temp_in)))
        self.atmo['tlev'] = np.flip(np.interp(np.flip(self.atmo['zlev']),np.flip(z_in),np.flip(temp_in)))
        
        # Calculate pressure [hPa]
        self.atmo['play'] = np.flip(np.interp(np.flip(self.atmo['zlay']),np.flip(z_in),np.flip(p_in)))
        self.atmo['plev'] = np.flip(np.interp(np.flip(self.atmo['zlev']),np.flip(z_in),np.flip(p_in)))
        
        # Calculate the vertical column of air above pressure level 
        # and use this to calculate the partial vertical air columns per layer [#/m^2].
        # Partial columns have the advantage that multiplication with cross sections
        # yields optical depth.
        nlev = len(self.zlev)
        sp = NA/(MDRYAIR*g0)*1.E2 # [#/m^2 * 1/hPa] air column above P is P*NA/Mair/g from p = m*g/area
        vc_air = sp*self.atmo['plev'] # air column [#/m^2] above pressure level
        self.atmo['AIR'] = (vc_air[1:nlev]-vc_air[0:nlev-1])#[#/m^2]
        self.atmo['AIR'][0] = vc_air[0]#[#/m^2] uppermost layer extends to infinity in terms of number of molecules
              
        # Interpolate mole fractions on output height grid
        # and then calculate partial columns per layer [#/m^2]
        # water vapor
        self.atmo['H2O'] = np.flip(np.interp(np.flip(self.atmo['zlay']),np.flip(z_in),np.flip(h2o_in)))*self.atmo['AIR']
        # co2
        self.atmo['CO2'] = np.flip(np.interp(np.flip(self.atmo['zlay']),np.flip(z_in),np.flip(co2_in)))*self.atmo['AIR']
        # no2
        self.atmo['CH4'] = np.flip(np.interp(np.flip(self.atmo['zlay']),np.flip(z_in),np.flip(ch4_in)))*self.atmo['AIR']
        # o2 use a constant mixing ratio
        self.atmo['O2']  = XO2*self.atmo['AIR']

###########################################################
class mie_data:
    """
    # The mie_data class collects data to 
    # calculate the scattering properties for
    # Mie scattering
    #
    # CONTAINS
    # method __init__(self,wave)
    # method get_data_Mie(self,filename)
    """
    ###########################################################
    def __init__(self,wave):
        """
        # init class
        #
        # arguments: 
        #            wave: array of wavelengths [wavelength] [nm]
        """
        self.wave = wave

    ###########################################################
    def get_data_Mie(self):
        """
        # Not yet implemented.
        """
        #check whether input is in range
        print("ERROR! mie_data.get_data_Mie: not yet implemented.")
        raise StopExecution

###########################################################
class molecular_data:
    """
    # The molecular_data class collects method for calculating
    # the absorption cross sections of molecular absorbers
    #
    # CONTAINS
    # method __init__(self,wave)
    # method get_data_HITRAN(self,xsdbpath, hp_ids)
    """
    ###########################################################
    def __init__(self,wave):
        """
        # init class
        #
        # arguments: 
        #            wave: array of wavelengths [wavelength] [nm]
        #            xsdb: dictionary with cross section data
        """
        self.xsdb = {}
        self.wave = wave
        
    ###########################################################
    def get_data_HITRAN(self,xsdbpath,hp_ids):
        """
        # Download line parameters from HITRAN web ressource via
        # the hapi tools, needs hapy.py in the same directory
        #
        # arguments: 
        #            xsdbpath: path to location where to store the absorption data
        #            hp_ids: list of isotopologue ids, format [(name1, id1),(name2, id2) ...]
        #                    (see hp.gethelp(hp.ISO_ID))
        # returns:   
        #            xsdb[id][path]: dictionary with paths to HITRAN parameter files
        """
        #check whether input is in range
        while True:
            if len(hp_ids)>0:
               break
            else:
               print("ERROR! molecular_data.get_data_HITRAN: provide at least one species.")
               raise StopExecution
        
        hp.db_begin(xsdbpath)
        wv_start = self.wave[0]
        wv_stop = self.wave[-1]
        
        #hp.gethelp(hp.ISO_ID)
        for id in hp_ids:
            key = '%2.2d'%id[1]
            self.xsdb[key]={}
            self.xsdb[key]['species']=id[0]
            self.xsdb[key]['name']='ID%2.2d_WV%5.5d-%5.5d'%(id[1],wv_start,wv_stop)# write 1 file per isotopologue
            # Check if data files are already inplace, if not: download
            if (not os.path.exists(os.path.join(xsdbpath,self.xsdb[key]['name']+'.data')) and not os.path.exists(os.path.join(xsdbpath,self.xsdb[key]['name']+'.header'))):
               hp.fetch_by_ids(self.xsdb[key]['name'],[id[1]],1.E7/wv_stop,1.E7/wv_start) # wavelength input is [nm], hapi requires wavenumbers [1/cm]

###########################################################
class optic_ssc_prop:
    """
    # The optic_ssc_prop class collects methods to
    # calculate optical scattering and absorption
    # properties of the single scattering atmosphere
    #
    # CONTAINS
    # method __init__(self, wave, zlay)
    # method cal_isoflat(self, atmosphere, taus_prior, taua_prior)
    # method cal_rayleigh(self, rayleigh, atmosphere, mu0, muv, phi)
    # method combine(self)
    """
    def __init__(self, wave, zlay):
        """
        # init class
        #
        # arguments: 
        #            prop: dictionary of contributing phenomena
        #            prop[wave]: array of wavelengths [wavelength] [nm]
        #            prop[zlay]: array of vertical height layers, midpoints [nlay] [m]
        """
        self.prop   = {}  
        self.wave   = wave  
        self.zlay   = zlay  

    def cal_isoflat(self, atm_data, taus_prior, taua_prior):
        """
        # Calculates isotropic and spectrally flat scattering optical properties
        # 
        # arguments: 
        #            atm_data: atmosphere_data object
        #            taus_prior: height-integrated (total) scattering optical depth prior
        #            taua_prior: height-integrated (total) absorption optical depth prior
        # returns:   
        #            prop['isoflat']: dictionary with optical properties
        #            prop['isoflat']['taus']: scattering optical thickness [wavelength, nlay] [-]
        #            prop['isoflat']['taua']: absorption optical thickness [wavelength, nlay] [-]
        #            prop['isoflat']['ssa']: single scattering albedo [wavelength, nlay] [-]
        #            prop['isoflat']['phase']: scattering phase function [wavelength, nlay] [-]
        """
        #check whether input is in range
        while True:
            if 0. <= taus_prior and 0 <= taua_prior:
               break
            else:
               print("ERROR! optic_prop.cal_rayleigh: input out of range.")
               raise StopExecution
        
        nlay  = self.zlay.size
        nwave = self.wave.size
        name = 'isoflat'
        self.prop[name]={}
        
        # Calculate scattering optical thickness, 
        # single scattering albedo and phase function 
        self.prop[name]['taus'] = np.zeros((nwave,nlay))  #scattering optical depth
        self.prop[name]['taua'] = np.zeros((nwave,nlay))  #absorption optical depth
        self.prop[name]['ssa']  = np.zeros((nwave,nlay))  #single scattering albedo
        self.prop[name]['phase']= np.zeros((nwave,nlay))  #phase function in ssc geometry
        
        # Adopt air density profile from atmosphere object and assume density-like
        # height distribution for optical depth
        scale_s=taus_prior/sum(atmosphere.dvc_air)
        scale_a=taua_prior/sum(atmosphere.dvc_air)
        for k,dvc_air in enumerate(atm_data.atmo['AIR']):
            self.prop[name]['taus'][:,k] = scale_s*dvc_air
            self.prop[name]['taua'][:,k] = scale_a*dvc_air
        self.prop[name]['phase'] = 1.
     
        self.prop[name]['ssa'] = np.divide(self.prop[name]['taus'],self.prop[name]['taus']+self.prop[name]['taua'])
          
    def cal_rayleigh(self, ray_data, atm_data, mu0, muv, phi):
        """
        # Calculates rayleigh scattering optical properties
        # for given single scattering geometry (mu0,muv,phi)
        #
        # arguments: 
        #            ray_data: rayleigh_data object
        #            atm_data: atmosphere_data object
        #            mu0: cosine of incoming zenith angle [-]
        #            muv: cosine of outgoing zenith angle [-]
        #            phi: relative azimuth angle [rad] [0:2pi]
        # returns:   
        #            prop['rayleigh']: dictionary with optical properties
        #            prop['rayleigh']['taus']: scattering optical thickness [wavelength, nlay] [-]
        #            prop['rayleigh']['ssa']: 1 [wavelength, nlay] [-]
        #            prop['rayleigh']['phase']: scattering phase function [wavelength, nlay] [-]
        #            prop['rayleigh']['cos_scat']: cosine of scattering anlge [-]
        """
        #check whether input is in range
        while True:
            if -1 <= mu0 <=1 and -1 <= muv <=1 and 0 <= phi <= 2.*np.pi:
               break
            else:
               print("ERROR! optic_prop.cal_rayleigh: input out of range.")
               raise StopExecution
        
        nlay  = self.zlay.size
        nwave = self.wave.size
        name = 'rayleigh'
        self.prop[name]={}
        
        # Scattering angle for solar beam (requires minus sign since beam is downward, mu0 by definition positive)  
        self.prop[name]['cos_scat'] = -mu0*muv + np.sqrt((1-mu0**2)*(1-muv**2))*np.cos(phi)

        # Calculate scattering optical thickness, 
        # single scattering albedo and phase function 
        self.prop[name]['taus']  = np.zeros((nwave,nlay))  #scattering optical depth
        self.prop[name]['ssa']   = np.zeros((nwave,nlay))  #single scattering albedo
        self.prop[name]['phase'] = np.zeros((nwave,nlay))  #phase function in ssc geometry
        
        for k,dvc_air in enumerate(atm_data.atmo['AIR']):
            self.prop[name]['taus'][:,k] = ray_data.xsec*dvc_air*1E-4 # factor 1E-4: [#/m2] -> [#/cm2]
            # Phase function 
            # equivalent to equation (12) in Bucholtz, Appl. Opt., 1995
            # self.phase_ray[:,k]=3./(4.*(1.+2*ray_data.gamma))*((1.+3.*ray_data.gamma)+(1.-ray_data.gamma)*self.cos_scat**2)
            self.prop[name]['phase'][:,k]= 1.5/(2.+ray_data.depol)*(1+ray_data.depol + (1-ray_data.depol)*self.prop[name]['cos_scat']**2)
        
        self.prop[name]['ssa'] = 1.

    def cal_mie(self, mie_data, atm_data, mu0, muv, phi):
        """
        # Calculates Mie scattering optical properties
        # for given single scattering geometry (mu0,muv,phi)
        #
        # arguments: 
        #            mie_data: mie_data object
        #            atm_data: atmosphere_data object
        #            mu0: cosine of incoming zenith angle [-]
        #            muv: cosine of outgoing zenith angle [-]
        #            phi: relative azimuth angle [rad] [0:2pi]
        # returns:   
        #            prop['mie']: dictionary with optical properties
        #            prop['mie']['taus']: scattering optical thickness [wavelength, nlay] [-]
        #            prop['mie']['taua']: absorption optical thickness [wavelength, nlay] [-]
        #            prop['mie']['ssa']: single scattering albedo [wavelength, nlay] [-]
        #            prop['mie']['phase']: scattering phase function [wavelength, nlay] [-]
        #            prop['mie']['cos_scat']: cosine of scattering anlge [-]
        """
        #check whether input is in range
        while True:
            if -1 <= mu0 <=1 and -1 <= muv <=1 and 0 <= phi <= 2.*np.pi:
               break
            else:
               print("ERROR! optic_prop.cal_mie: input out of range.")
               raise StopExecution
        
        print("ERROR! optic_prop.cal_mie: not yet implemented.")
        raise StopExecution
        
    def cal_molec(self, molec_data, atm_data):
        """
        # Calculates molecular absorption properties
        # for given single scattering geometry (mu0,muv,phi)
        #
        # arguments: 
        #            molec_data: molec_data object
        #            atm_data: atmosphere_data object
        # returns:   
        #            prop['molec_XX']: dictionary with optical properties with XXXX HITRAN identifier code
        #            prop['molec_XX']['taua']: absorption optical thickness [wavelength, nlay] [-]
        """
        nlay  = self.zlay.size
        nwave = self.wave.size
        nu_samp = 0.005 # Wavenumber sampling [1/cm] of cross sections
        # Loop over all isotopologues, id = HITRAN global isotopologue ID
        for id in molec_data.xsdb.keys():            
            name = 'molec_'+id
            species = molec_data.xsdb[id]['species']

            # Write absorption optical depth [nwave,nlay] in dictionary / per isotopologue
            self.prop[name]={}
            self.prop[name]['taua']  = np.zeros((nwave,nlay))

            # Check whether absorber type is in the atmospheric data structure
            if species not in atm_data.atmo.keys():
               print("WARNING! optic_prop.cal_molec: absorber type not in atmospheric data.",id,species)
            else:
               # Loop over all atmospheric layers
               for ki,pi,Ti,ni in zip(range(len(atm_data.atmo['zlay'])),atm_data.atmo['play'],atm_data.atmo['tlay'],atm_data.atmo[species]):
                   # Calculate absorption cross section for layer
                   nu,xs = hp.absorptionCoefficient_Voigt(SourceTables=molec_data.xsdb[id]['name'], Environment={'p':pi/PSTD, 'T':Ti},WavenumberStep=nu_samp)
                   # Interpolate on wavelength grid provided on input
                   xs_ip = np.interp(self.wave,np.flip(1E7/nu),np.flip(xs))
                   # Calculate absorption optical depth
                   self.prop[name]['taua'][:,ki] = xs_ip*ni*1E-4 # factor 1E-4: [#/m2] -> [#/cm2]

    def combine(self, *args):
        """
        # Combines optical scattering and absorption properties from
        # various contributors to composite optical properties
        #
        # arguments:
        #            variable number of names of optic_prop.prop dictionaries
        # returns:   
        #            prop[taus]: total scattering optical thickness array [wavelength, nlay] [-]
        #            prop[taua]: total absorption optical thickness array [wavelength, nlay] [-]
        #            prop[ssa]: total single scattering albedo array [wavelength, nlay] [-]
        #            prop[phase]: total scattering phase function array [wavelength, nlay] [-]
        """
        #check whether input is in range
        while True:
            if len(args) > 0:
               break
            else:
               print("ERROR! optic_prop.combine: name of prop dictionary required.")
               raise StopExecution
        
        nlay  = self.zlay.size
        nwave = self.wave.size

        self.prop['taus'] = np.zeros((nwave,nlay))
        self.prop['taua'] = np.zeros((nwave,nlay))
        self.prop['ssa'] = np.ones((nwave,nlay))
        self.prop['phase'] = np.zeros((nwave,nlay))
        
        for name in args:
            # Composite scattering optical thickness
            if 'taus' in self.prop[name].keys():
                self.prop['taus']  = self.prop['taus'] + self.prop[name]['taus']
            # Composite absorption optical thickness
            if 'taua' in self.prop[name].keys():
                self.prop['taua'] = self.prop['taua'] + self.prop[name]['taua']
        for name in args:
            if 'phase' in self.prop[name].keys() and np.all(self.prop['taus']):
               # Composite phase function, sum weighted by scattering optical thickness
               self.prop['phase'] = self.prop['phase'] + np.divide(np.multiply(self.prop[name]['phase'],self.prop[name]['taus']),self.prop['taus'])
        
        # Composite single scattering albedo
        if np.all(self.prop['taus']+self.prop['taua']):
            self.prop['ssa']  = np.divide(self.prop['taus'],self.prop['taus']+self.prop['taua'])

###########################################################
class rayleigh_data:
    """
    # The rayleigh_data class collects data to 
    # calculate the scattering properties for
    # Rayleigh scattering: scattering cross section,
    # depolarization factor, gamma factor. 
    # It largely follows Bucholtz, Appl. Optics, 1995, 
    # https://doi.org/10.1364/AO.34.002765
    # 
    # CONTAINS
    # method __init__(self,wave)
    # method get_data_Rayleigh(self,filename)
    """
    ###########################################################
    def __init__(self,wave):
        """
        # init class
        #
        # arguments: 
        #            wave: array of wavelengths [wavelength] [nm]
        """
        self.wave = wave

    ###########################################################
    def get_data_Rayleigh(self,filename):
        """
        # Read and calculate Rayleigh scattering cross sections
        # depolarization factor, and gamma = depol/(2-depol) 
        # according to the definitions in Bucholtz, 
        # Appl. Opt., 1995, https://doi.org/10.1364/AO.34.002765
        #
        # arguments: 
        #            filename: file with depolarization and gamma data
        # returns:   
        #            xsec:  scattering cross section [wavelength] [cm2]
        #            depol: depolarization factor [wavelength] [-]
        #            gamma: factor for phase function calculation [wavelength] [-]
        """
        #check whether input is in range
        while True:
            if os.path.exists(filename):
               break
            else:
               print("ERROR! rayleigh_data.get_data_Rayleigh: file does not exist.")
               raise StopExecution
                
        # Tab. 3 in Bucholtz
        a_ray =[3.01577E-28, 4.01061E-28]
        b_ray =[3.55212    , 3.99668    ]
        c_ray =[1.35579    , 1.10298E-03]
        d_ray =[0.11563    , 2.71393E-02]

        # Scattering cross section [cm2] according to eq. (8) in Bucholtz
        wmu = self.wave*1E-3 # wavelength in [micron]
        id1=np.where(wmu<=0.5) # select first column index for wavelength <500 nm
        id2=np.where(wmu>0.5)  # select second column index for wavelength >500 nm
        self.xsec = np.zeros(wmu.size) # generate scattering cross section array
        self.xsec[id1] = a_ray[0]*wmu[id1]**(-(b_ray[0]+c_ray[0]*wmu[id1]+d_ray[0]/wmu[id1]))  
        self.xsec[id2] = a_ray[1]*wmu[id2]**(-(b_ray[1]+c_ray[1]*wmu[id2]+d_ray[1]/wmu[id2]))
            
        # Tab. 1 in Bucholtz
        atm = np.genfromtxt(filename,skip_header=5)   
        wave_data  = atm[:,0]
        depol_data = atm[:,1]
        gamma_data = atm[:,2]
        # Use cubic spline for data interpolation
        coeffs = interpolate.splrep(wave_data,depol_data)
        self.depol  = interpolate.splev(self.wave,coeffs) # depolarization factor
        coeffs = interpolate.splrep(wave_data,gamma_data)
        self.gamma  = interpolate.splev(self.wave,coeffs) # gamma factor

###########################################################
class StopExecution(Exception):
    """
    # Class to stop execution in jupyter notebook and ipython
    # Call via 'raise StopExecution'
    """    
    def _render_traceback_(self):
        pass

###########################################################
class surface_prop:
    """
    # The surface_prop class collects methods to
    # calculate surface properties
    #
    # CONTAINS
    # method __init__(self,wave)
    # method get_albedo_flat(alb_prior)
    # method get_albedo_CUSTOM(filename,sfname)
    # method get_albedo_ECOSTRESS(filename)
    """
    ###########################################################
    def __init__(self, wave):
        """
        # init class
        #
        # arguments: 
        #            wave: array of wavelengths [wavelength] [nm]
        """
        self.wave = wave
        self.alb = np.zeros_like(wave)
        
    ###########################################################
    def get_albedo_flat(self, alb_prior):
        """
        # Generate spectrally flat albedo array
        #    
        # arguments: 
        #            alb_prior: albedo value to be used throughout spectral range
        # returns:  
        #            alb: constant albedo array [wavelength]
        """    
        #check whether input is in range
        while True:
            if 0. <= alb_prior <= 1.:
               break
            else:
               print("ERROR! surface_prop.get_albedo_flat: albedo prior needs to be in [0...1].")
               raise StopExecution
    
        # Read data from file
        self.alb = np.array([alb_prior for wi in self.wave])
    
    ###########################################################
    def get_albedo_CUSTOM(self,filename,sfname):
        """
        # Read albedo from custom database. This is generic typical 
        # data. For a comprehensive albedo database, have a look 
        # at: https://speclib.jpl.nasa.gov/
        #    
        # arguments: 
        #            filename: file with albedo database
        #            sfname: name of surface type 
        #                    [sand,soil,snow,vegetation,water]
        # returns:  
        #            alb: albedo array interpolated to wavelength [wavelength]
        """    
        #check whether input is in range
        sftypes=['sand','soil','snow','vegetation','water']
        while True:
            if os.path.exists(filename) and sfname in sftypes:
               break
            else:
               print("ERROR! surface_prop.get_albedo_CUSTOM: input out of range.")
               raise StopExecution
    
        # Read data from file
        raw = np.genfromtxt(filename,skip_header=15) # read file into numpy array, skip 15 header lines
        
        # Index of surface type
        isf = sftypes.index(sfname)
        
        # Interpolate albedo to wavelength array
        wave_org = raw[:,0]
        alb_org = raw[:,isf+1]
        self.alb = np.interp(self.wave,wave_org,alb_org)

    ###########################################################
    def get_albedo_ECOSTRESS(self,filename):
        """
        # Read albedo from ECOSTRESS database
        # at: https://speclib.jpl.nasa.gov/
        #
        # arguments: 
        #            filename: file with albedo database
        # returns:  
        #            alb: albedo array interpolated to wavelength [wavelength]
        """ 
        # Check whether input is in range
        while True:
            if os.path.exists(filename):
               break
            else:
               print("ERROR! surface_prop.get_albedo_ECOSTRESS: input out of range.")
               raise StopExecution
            
        # Read data from file
        raw = np.genfromtxt(filename,skip_header=21,unpack=True) 
    
        wv_in = np.array([a*1E3 for a in raw[0,:]]) # wavelength [nm]
        alb_in = np.array([a/1E2 for a in raw[1,:]]) # albedo [0...1]
        # Check if wavelength in ascending order. If not, flip arrays.
        if wv_in[0] > wv_in[-1]:
           # Interpolate albedo to wavelength array
           self.alb = np.interp(self.wave,np.flip(wv_in),np.flip(alb_in))
        else:
           self.alb = np.interp(self.wave,wv_in,alb_in)
