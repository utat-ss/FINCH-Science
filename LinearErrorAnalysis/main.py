"""
main.py  

Main driver for the Linear Error Analysis program.

Author(s): Adyn Miles, Shiqi Xu, Rosie Liang
"""

import numpy as np
from config import parse_config
from forward import Forward
from optim import Optim
import photon_noise
from isrf import ISRF

if __name__ == "__main__":
    
    cfg = parse_config()

    forward = Forward(cfg)
    surface, molec, atm, sun_lbl = forward.get_atm_params()
    optics = forward.opt_properties()
    wave_meas, rad_tot, rad_ch4, rad_co2, rad_h2o = forward.plot_transmittance()

    isrf = ISRF(cfg)
    isrf_func = isrf.define_isrf()
    isrf_conv = isrf.convolve_isrf(rad_tot)
    
    lea = Optim(cfg, wave_meas)
    sys_errors = lea.sys_errors()
    rand_errors = lea.rand_errors()