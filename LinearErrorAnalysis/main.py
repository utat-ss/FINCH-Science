"""
main.py  

Main driver for the Linear Error Analysis program.

Author(s): Adyn Miles, Shiqi Xu, Rosie Liang
"""

import numpy as np
from config import parse_config
from forward_hapi import Forward

if __name__ == "__main__":
    
    cfg = parse_config()

    # Methane absorption
    forward_ch4 = Forward(cfg, 'CH4', [cfg.spectral_lower, cfg.spectral_upper])
    absoprtion_coeff_ch4 = forward_ch4.absorption_coeff()
    optical_depth_ch4 = forward_ch4.optical_depth()
    forward_ch4.radiance()
    forward_ch4.transmittance()


    # Carbon Dioxide absorption
    forward_co2 = Forward(cfg, 'CO2', [cfg.spectral_lower, cfg.spectral_upper])
    absoprtion_coeff_co2 = forward_co2.absorption_coeff()
    optical_depth_co2 = forward_co2.optical_depth()
    forward_co2.radiance()
    forward_ch4.transmittance()



    # Water Vapour absorption
    forward_h2o = Forward(cfg, 'H2O', [cfg.spectral_lower, cfg.spectral_upper])
    absoprtion_coeff_h2o = forward_h2o.absorption_coeff()
    optical_depth_h2o = forward_h2o.optical_depth()
    forward_h2o.radiance()
    forward_ch4.transmittance()





