"""
main.py  

Main driver for the Linear Error Analysis program.

Author(s): Adyn Miles, Shiqi Xu, Rosie Liang
"""

import numpy as np
import matplotlib.pyplot as plt

import config
import lib.gta_xch4 as gta_xch4
import lib.photon_noise as pn
from forward import Forward
from isrf import ISRF
from optim import Optim

if __name__ == "__main__":
    
    cfg = config.parse_config()

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
    # sys_error_1 = lea.sys_err_vector(1)
    # sys_error_2 = lea.sys_err_vector(2)
    # sys_error_3 = lea.sys_err_vector(3)
    # sys_error_4 = lea.sys_err_vector(4)
    # sys_error_5 = lea.sys_err_vector(5)
    # sys_error_6 = lea.sys_err_vector(6)
    # sys_error_7 = lea.sys_err_vector(7)
    # sys_error_8 = lea.sys_err_vector(8)
    ecm = lea.error_covariance()

    # plot interpolated photon noise
    plt.plot(lea.wave_meas, lea.photon_noise_interp)
    plt.show()
