import numpy as np

'''
TODO:
- Figure out how this Forward class actually gets called (am I running this over a range of temperatures, pressures, wavenumbers)?
- Implement a main that calls this class so I can start testing the basic formulas.
- What is the difference between p_self and p (partial pressure of methane and pressure). How is p_self calculated?
- What is the difference between the transition wavenumber and the wavenumber?
- Where is the threshold between upper and lower atmosphere?
'''

class Forward:
    def __init__(self, temperature, pressure, wavenumber):
        self.t_wavenumber = []
        self.spec_line_intensity = []
        self.gamma_air = []
        self.gamma_self = []
        self.delta_air = []
        self.n_air = []

        self.light_speed = 3.00e-8
        self.avogadro = 6.022e23
        self.boltzmann = 1.38e-23
        self.temp_ref = 296
        self.temp = temperature
        self.molar_mass = 16.04
        self.p_self = 1
        self.pressure = pressure
        self.wavenumber = wavenumber
    
    def read(self, molecule):
        if molecule == "methane":
            with open('ch4_line_by_line.par') as methane:
                for row in methane:
                    self.t_wavenumber.append(float(row[4:15]))
                    self.spec_line_intensity.append(float(row[17:26]))
                    self.gamma_air.append(float(row[37:42]))
                    self.gamma_self.append(float(row[42:47]))
                    self.n_air.append(float(row[57:60]))
                    self.delta_air.append(float(row[60:68]))
        elif molecule == "carbon_dioxide":
            with open('co2_line_by_line.par') as carbon_dioxide:
                for row in carbon_dioxide:
                    self.t_wavenumber.append(float(row[4:15]))
                    self.spec_line_intensity.append(float(row[17:26]))
                    self.gamma_air.append(float(row[37:42]))
                    self.gamma_self.append(float(row[42:47]))
                    self.n_air.append(float(row[57:60]))
                    self.delta_air.append(float(row[60:68]))
        elif molecule == "water_vapour":
            with open('h2o_line_by_line.par') as water_vapour:
                for row in water_vapour:
                    self.t_wavenumber.append(float(row[4:15]))
                    self.spec_line_intensity.append(float(row[17:26]))
                    self.gamma_air.append(float(row[37:42]))
                    self.gamma_self.append(float(row[42:47]))
                    self.n_air.append(float(row[57:60]))
                    self.delta_air.append(float(row[60:68]))
                
        self.t_wavenumber = np.asarray(self.t_wavenumber)
        self.spec_line_intensity = np.asarray(self.spec_line_intensity)
        self.gamma_air = np.asarray(self.gamma_air)
        self.gamma_self = np.asarray(self.gamma_self)
        self.delta_air = np.asarray(self.delta_air)
        self.n_air = np.asarray(self.n_air)
        return

    # This function will likely need to take another argument once I figure out the indexing situation.
    def absorption(self):
        # Half width at half-maximum of the Doppler-broadened component
        self.alpha_d = (self.t_wavenumber/self.light_speed) * np.sqrt((2*self.avogadro*self.boltzmann*self.temp*np.log(2))/self.molar_mass)
        
        # Lorentzian pressure broadened half-width half maximum
        self.gamma = ((self.temp_ref/self.temp)**self.n_air) * (self.gamma_air*(self.pressure - self.p_self) - self.gamma_self*self.p_self)

        # Normalized line shape functions:
        # Lower Atmosphere    
        self.f_L = (1/np.pi)*self.gamma/(self.gamma**2 + (self.wavenumber - (self.t_wavenumber + self.delta_air*self.pressure))**2)
        # Upper Atmosphere
        self.f_G = np.sqrt(np.log(2)/(np.pi*self.alpha_d**2))*np.exp(-((self.wavenumber - self.t_wavenumber)**2 * np.log(2)/ self.alpha_d**2))

        # How do I split up the atmosphere into upper and lower?
