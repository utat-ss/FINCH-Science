import numpy as np

'''
TODO:
- Figure out how this Forward class actually gets called (am I running this over a range of temperatures, pressures, wavenumbers)?
- Figure out how different isotopes factor into the LEA.
- Confirm the way of calculating column density is correct?
- What is the difference between p_self and p (partial pressure of methane and pressure). How is p_self calculated?
- What is the difference between the transition wavenumber and the wavenumber?
- Shiqi: Confirm the distinction between upper and lower atmosphere with climate science professor.
'''

class Forward:
    def __init__(self, molecule, config):

        '''
        Initializes the class.

        Parameters:
            self: for containing configuration details from the initialization
            molecule: initialize a specific molecule: "methane", "carbon_dioxide", and "water_vapour"


        Returns:
            None - all lists are stored in the class.

        Instructions on parsing the .par file format are given:
            https://www.sciencedirect.com/science/article/abs/pii/S0022407305001081

        '''
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
        self.temp = config.temperature
        self.p_self = 1
        self.pressure = config.pressure
        self.wavenumber = (10^7) / config.wavelength
        self.volume_number_density = 0.02504e27

        self.cfg = config

        if molecule == "methane":
            self.molar_mass = 16.04
        elif molecule == "carbon_dioxide":
            self.molar_mass = 44.01 
        elif molecule == "water_vapour":
            self.molar_mass = 18.02
    
    def read(self, molecule):

        '''
        Reads .par files from HITRAN and stores their values.

        Parameters:
            self: contains configuration details from the initialization
            molecule: read HITRAN data from a specified molecule: "methane", "carbon_dioxide", and "water_vapour"

        Returns:
            None - all lists are stored in the class.

        Instructions on parsing the .par file format are given:
            https://www.sciencedirect.com/science/article/abs/pii/S0022407305001081

        '''

        if molecule == "methane":
            with open(self.cfg.ch4_file) as methane:
                for row in methane:
                    self.t_wavenumber.append(float(row[4:15]))
                    self.spec_line_intensity.append(float(row[16:25]))
                    self.gamma_air.append(float(row[35:40]))
                    self.gamma_self.append(float(row[40:45]))
                    self.n_air.append(float(row[55:59]))
                    self.delta_air.append(float(row[59:67]))
        elif molecule == "carbon_dioxide":
            with open(self.cfg.co2_file) as carbon_dioxide:
                for row in carbon_dioxide:
                    self.t_wavenumber.append(float(row[4:15]))
                    self.spec_line_intensity.append(float(row[16:25]))
                    self.gamma_air.append(float(row[35:40]))
                    self.gamma_self.append(float(row[40:45]))
                    self.n_air.append(float(row[55:59]))
                    self.delta_air.append(float(row[59:67]))
        elif molecule == "water_vapour":
            with open(self.cfg.h2o_file) as water_vapour:
                for row in water_vapour:
                    self.t_wavenumber.append(float(row[4:15]))
                    self.spec_line_intensity.append(float(row[16:25]))
                    self.gamma_air.append(float(row[35:40]))
                    self.gamma_self.append(float(row[40:45]))
                    self.n_air.append(float(row[55:59]))
                    self.delta_air.append(float(row[59:67]))
                
        self.t_wavenumber = np.asarray(self.t_wavenumber)
        self.spec_line_intensity = np.asarray(self.spec_line_intensity)
        self.gamma_air = np.asarray(self.gamma_air)
        self.gamma_self = np.asarray(self.gamma_self)
        self.delta_air = np.asarray(self.delta_air)
        self.n_air = np.asarray(self.n_air)
        return

    def absorption(self):
        '''
        Returns the optical depth of the desired molecule.

        Parameters:
            self: contains configuration details from the initialization

        Returns:
            optical_depth: ratio of transmitted to incident power through atmospheric columns of the desired molecule.

        Equations are sourced from:
            https://hitran.org/docs/definitions-and-units/

        '''

        self.z_value = 500
        self.column_number_density = self.volume_number_density*self.z_value

        # Half width at half-maximum of the Doppler-broadened component
        self.alpha_d = (self.t_wavenumber/self.light_speed) * np.sqrt((2*self.avogadro*self.boltzmann*self.temp*np.log(2))/self.molar_mass)
        
        # Lorentzian pressure broadened half-width half maximum
        self.gamma = ((self.temp_ref/self.temp)**self.n_air) * (self.gamma_air*(self.pressure - self.p_self) - self.gamma_self*self.p_self)

        # Normalized line shape functions:
        # Source claiming the troposphere corresponds to lower atmosphere, and above that corresponds to the upper atmosphere. 
        # Link: https://encyclopedia2.thefreedictionary.com/lower+atmosphere#:~:text=The%20atmosphere%20may%20be%20divided%20into%20two%20parts,above%20the%20tropopause%20is%20called%20the%20upper%20atmosphere.
        # Troposphere is 12 km above the Earth's surface.
        # Lower Atmosphere   
        if self.z_value <= 12: 
            self.line_shape = (1/np.pi)*self.gamma/(self.gamma**2 + (self.wavenumber - (self.t_wavenumber + self.delta_air*self.pressure))**2)
        else:
            # Upper Atmosphere
            self.line_shape = np.sqrt(np.log(2)/(np.pi*self.alpha_d**2))*np.exp(-((self.wavenumber - self.t_wavenumber)**2 * np.log(2)/ self.alpha_d**2))
        
        self.mono_absorption = self.spec_line_intensity*self.line_shape
        self.optical_depth = self.column_number_density*self.mono_absorption
        print (self.line_shape)