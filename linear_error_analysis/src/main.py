"""
main.py

Main driver for the Linear Error Analysis program.
Can be run using `lea.sh`.
Can choose which plots to see by toggling on/off `show_fig` param.

Author(s): Adyn Miles, Shiqi Xu, Rosie Liang
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import config
import libs.gta_xch4 as gta_xch4
import libs.photon_noise as pn
from errors import Errors
from forward import Forward
from isrf import ISRF
from optim import Optim

if __name__ == "__main__":

    cfg = config.parse_config()

    forward = Forward(cfg)
    surface, molec, atm, sun_lbl = forward.get_atm_params()
    optics = forward.opt_properties()
    (
        wave_meas,
        rad_tot,
        rad_ch4,
        rad_co2,
        rad_h2o,
        d_rad_ch4,
        d_rad_co2,
        d_rad_h2o,
        rad_conv_tot,
        rad_conv_ch4,
        rad_conv_co2,
        rad_conv_h2o,
        dev_conv_ch4,
        dev_conv_co2,
        dev_conv_h2o,
    ) = forward.plot_transmittance(show_fig=False)
    state_vector = forward.produce_state_vec()

    isrf = ISRF(cfg)
    isrf_func = isrf.define_isrf(show_fig=False)
    isrf_conv = isrf.convolve_isrf(rad_tot, show_fig=False)

    lea = Errors(cfg, wave_meas)
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

    optim = Optim(cfg, wave_meas)
    jacobian = optim.jacobian(dev_conv_ch4, dev_conv_co2, dev_conv_h2o, show_fig=False)
    gain = optim.gain(ecm)
    modified_meas_vector = optim.modify_meas_vector(state_vector, rad_conv_tot, ecm)
    spectral_res, snr = optim.state_estimate(ecm, modified_meas_vector, sys_errors)

    print("Estimated Solution: " + str(spectral_res))
    print("Uncertainty of Solution: " + str(snr))

    # plot interpolated photon noise
    # plt.plot(lea.wave_meas, lea.photon_noise_interp)
    # plt.title("Interpolated Photon Noise")
    # plt.xlabel("Wavelength (nm)")
    # plt.ylabel("Photon Noise (UNITS?)")    # TODO
    # plt.show()
