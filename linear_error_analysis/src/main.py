"""
main.py  

Main driver for the Linear Error Analysis program.
Can be run using `lea.sh`.
Can choose which plots to see by toggling on/off `show_fig` param.

Author(s): Adyn Miles, Shiqi Xu, Rosie Liang
"""

import os

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
    wave_meas, rad_tot, rad_ch4, rad_co2, rad_h2o = forward.plot_transmittance(
        show_fig=False)


    isrf = ISRF(cfg)
    isrf_func = isrf.define_isrf(show_fig=False)
    isrf_conv = isrf.convolve_isrf(rad_tot, show_fig=False)


    lea = Optim(cfg, wave_meas)
    sys_errors = lea.sys_errors()
    rand_errors = lea.rand_errors()

    # sys_nonlinearity = lea.sys_err_vector(1)
    # sys_stray_light = lea.sys_err_vector(2)
    # sys_crosstalk = lea.sys_err_vector(3)
    # sys_flat_field = lea.sys_err_vector(4)
    # sys_bad_px = lea.sys_err_vector(5)
    # sys_key_smile = lea.sys_err_vector(6)
    # sys_striping = lea.sys_err_vector(7)
    # sys_memory = lea.sys_err_vector(8)

    ecm = lea.error_covariance()
    path_root = os.path.dirname(os.path.dirname(__file__))
    np.savetxt(os.path.join(path_root, "outputs", "ecm.csv"), ecm, delimiter=",")


    # plot interpolated photon noise
    plt.plot(lea.wave_meas, lea.photon_noise_interp)
    plt.title("Interpolated Photon Noise")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Photon Noise (UNITS?)")    # TODO
    # plt.show()
