"""
main.py

Main driver for the Linear Error Analysis program.
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

def main(cfg):
    # Forward model and state vector assessment
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

    # Instrument spectral response function and convolutions with the forward model.
    # isrf = ISRF(cfg)
    # isrf_func = isrf.define_isrf(show_fig=False)
    # wave_meas_conv, rad_conv, i1, i2, slit = isrf.convolve_isrf(rad_tot, wave_meas, show_fig=False)

    # Generate error vectors and matrices
    errors = Errors(cfg, wave_meas)
    sys_errors = errors.sys_errors()
    non_linearity = errors.sys_err_vector(1)
    stray_light = errors.sys_err_vector(2)
    cross_talk = errors.sys_err_vector(3)
    flat_field = errors.sys_err_vector(4)
    bad_pixel = errors.sys_err_vector(5)
    smile = errors.sys_err_vector(6)
    memory = errors.sys_err_vector(7)
    striping = errors.sys_err_vector(8)
    rand_errors = errors.rand_errors()
    average_snr = 1/np.mean(rand_errors[:, 1])
    ecm = errors.error_covariance()

    # path_root = os.path.dirname(os.path.dirname(__file__))
    # np.savetxt(os.path.join(path_root, "outputs", "ecm.csv"), ecm, delimiter=",")

    optim = Optim(cfg, wave_meas)
    jacobian = optim.jacobian(dev_conv_ch4, dev_conv_co2, dev_conv_h2o, show_fig=False)
    gain = optim.gain(ecm)
    modified_meas_vector = optim.modify_meas_vector(state_vector, rad_conv_tot, ecm)
    estimate, uncertainty = optim.state_estimate(ecm, modified_meas_vector)

    # Measure effects of systematic errors on the precision estimates.
    d_nonlinearity = optim.system_est_error(non_linearity)
    d_straylight = optim.system_est_error(stray_light)
    d_crosstalk = optim.system_est_error(cross_talk)
    d_flatfield = optim.system_est_error(flat_field)
    d_badpixel = optim.system_est_error(bad_pixel)
    d_smile = optim.system_est_error(smile)
    d_memory = optim.system_est_error(memory)
    d_striping = optim.system_est_error(striping)

    print("Estimated Solution: " + str(estimate))
    print("Uncertainty of Solution: " + str(uncertainty))
    print("Non-Linearity Precision Error: " + str(d_nonlinearity))
    print("Stray Light Precision Error: " + str(d_straylight))
    print("Cross Talk Precision Error: " + str(d_crosstalk))
    print("Flat Field Precision Error: " + str(d_flatfield))
    print("Bad Pixel Precision Error: " + str(d_badpixel))
    print("Smile Precision Error: " + str(d_smile))
    print("Memory Precision Error: " + str(d_memory))
    print("Striping Precision Error: " + str(d_striping))

    return estimate, average_snr, d_nonlinearity, d_straylight, d_crosstalk, d_flatfield, d_badpixel, d_smile, d_memory, d_striping

if __name__ == "__main__":
    dark_current = [10, 8, 6, 4, 2, 0.0000001]
    readout_noise = [500, 300, 100, 0.0000001]
    integration_time = [0.0025, 0.05, 0.1, 0.1666667, 0.3333333, 0.5]
    spectral_resolution = [0.2, 0.5, 1.0, 1.5, 2.0]

    estimate_list = np.zeros((len(dark_current), 3))
    snr_list = np.zeros((len(dark_current), 1))
    nonlinearity_list = np.zeros((len(dark_current), 3))
    straylight_list = np.zeros((len(dark_current), 3))
    crosstalk_list = np.zeros((len(dark_current), 3))
    flatfield_list = np.zeros((len(dark_current), 3))
    badpixel_list = np.zeros((len(dark_current), 3))
    smile_list = np.zeros((len(dark_current), 3))
    memory_list = np.zeros((len(dark_current), 3))
    striping_list = np.zeros((len(dark_current), 3))

    state_vector = np.zeros((len(dark_current), 3))
    for idx in range(0, len(dark_current)):
        state_vector[idx, 0] = 1.87
        state_vector[idx, 1] = 420
        state_vector[idx, 2] = 50000 

    for idx in range(0, len(dark_current)):
        cfg = config.parse_config(dark_current[idx], 500, 0.16667, 1.5)
        estimate_list[idx, :], snr_list[idx, :], nonlinearity_list[idx, :], straylight_list[idx, :], crosstalk_list[idx, :], flatfield_list[idx, :], badpixel_list[idx, :], smile_list[idx, :], memory_list[idx, :], striping_list[idx, :] = main(cfg)

    for molecule in range(0, 3):
        plt.plot(dark_current, abs(state_vector[:, molecule] - estimate_list[:, molecule])/state_vector[:, molecule] * 100)
    plt.title("Estimate error as a function of Dark Current")
    plt.xlabel("Dark Current (in nA/cm^2)")
    plt.ylabel("Estimate error (%)")
    plt.legend(["Methane", "Carbon Dioxide", "Water Vapour"])
    plt.show()
    plt.close()

    for molecule in range(0, 3):
        plt.plot(snr_list, abs(state_vector[:, molecule] - estimate_list[:, molecule])/state_vector[:, molecule] * 100)
    plt.title("Estimate error as a function of Signal to Noise Ratio")
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Estimate error (%)")
    plt.legend(["Methane", "Carbon Dioxide", "Water Vapour"])
    plt.show()
    plt.close()

    estimate_list = np.zeros((len(readout_noise), 3))
    snr_list = np.zeros((len(readout_noise), 1))
    nonlinearity_list = np.zeros((len(readout_noise), 3))
    straylight_list = np.zeros((len(readout_noise), 3))
    crosstalk_list = np.zeros((len(readout_noise), 3))
    flatfield_list = np.zeros((len(readout_noise), 3))
    badpixel_list = np.zeros((len(readout_noise), 3))
    smile_list = np.zeros((len(readout_noise), 3))
    memory_list = np.zeros((len(readout_noise), 3))
    striping_list = np.zeros((len(readout_noise), 3))

    state_vector = np.zeros((len(readout_noise), 3))
    for idx in range(0, len(readout_noise)):
        state_vector[idx, 0] = 1.87
        state_vector[idx, 1] = 420
        state_vector[idx, 2] = 50000 

    for idx in range(0, len(readout_noise)):
        cfg = config.parse_config(10, readout_noise[idx], 0.16667, 1.5)
        estimate_list[idx, :], snr_list[idx, :], nonlinearity_list[idx], straylight_list[idx], crosstalk_list[idx], flatfield_list[idx], badpixel_list[idx], smile_list[idx], memory_list[idx], striping_list[idx] = main(cfg)
    
    for molecule in range(0, 3):
        plt.plot(readout_noise, abs(state_vector[:, molecule] - estimate_list[:, molecule])/state_vector[:, molecule] * 100)
    plt.title("Estimate error as a function of Readout Noise")
    plt.xlabel("Readout Noise (in e-)")
    plt.ylabel("Estimate error (%)")
    plt.legend(["Methane", "Carbon Dioxide", "Water Vapour"])
    plt.show()
    plt.close()

    estimate_list = np.zeros((len(integration_time), 3))
    snr_list = np.zeros((len(integration_time), 1))
    nonlinearity_list = np.zeros((len(integration_time), 3))
    straylight_list = np.zeros((len(integration_time), 3))
    crosstalk_list = np.zeros((len(integration_time), 3))
    flatfield_list = np.zeros((len(integration_time), 3))
    badpixel_list = np.zeros((len(integration_time), 3))
    smile_list = np.zeros((len(integration_time), 3))
    memory_list = np.zeros((len(integration_time), 3))
    striping_list = np.zeros((len(integration_time), 3))

    state_vector = np.zeros((len(integration_time), 3))
    for idx in range(0, len(integration_time)):
        state_vector[idx, 0] = 1.87
        state_vector[idx, 1] = 420
        state_vector[idx, 2] = 50000 

    for idx in range(0, len(integration_time)):
        cfg = config.parse_config(10, 500, integration_time[idx], 1.5)
        estimate_list[idx], snr_list[idx], nonlinearity_list[idx], straylight_list[idx], crosstalk_list[idx], flatfield_list[idx], badpixel_list[idx], smile_list[idx], memory_list[idx], striping_list[idx] = main(cfg)
        
    for molecule in range(0, 3):
        plt.plot(integration_time, abs(state_vector[:, molecule] - estimate_list[:, molecule])/state_vector[:, molecule] * 100)
    plt.title("Estimate error as a function of Integration Time")
    plt.xlabel("Integration Time (in s)")
    plt.ylabel("Estimate error (%)")
    plt.legend(["Methane", "Carbon Dioxide", "Water Vapour"])
    plt.show()

    estimate_list = np.zeros((len(spectral_resolution), 3))
    snr_list = np.zeros((len(spectral_resolution), 1))
    nonlinearity_list = np.zeros((len(spectral_resolution), 3))
    straylight_list = np.zeros((len(spectral_resolution), 3))
    crosstalk_list = np.zeros((len(spectral_resolution), 3))
    flatfield_list = np.zeros((len(spectral_resolution), 3))
    badpixel_list = np.zeros((len(spectral_resolution), 3))
    smile_list = np.zeros((len(spectral_resolution), 3))
    memory_list = np.zeros((len(spectral_resolution), 3))
    striping_list = np.zeros((len(spectral_resolution), 3))  

    state_vector = np.zeros((len(spectral_resolution), 3))
    for idx in range(0, len(spectral_resolution)):
        state_vector[idx, 0] = 1.87
        state_vector[idx, 1] = 420
        state_vector[idx, 2] = 50000   

    for idx in range(0, len(spectral_resolution)):
        cfg = config.parse_config(10, 500, 0.16667, spectral_resolution[idx])
        estimate_list[idx], snr_list[idx], nonlinearity_list[idx], straylight_list[idx], crosstalk_list[idx], flatfield_list[idx], badpixel_list[idx], smile_list[idx], memory_list[idx], striping_list[idx] = main(cfg)
    
    for molecule in range(0, 3):
        plt.plot(spectral_resolution, abs(state_vector[:, molecule] - estimate_list[:, molecule])/state_vector[:, molecule] * 100)
    plt.title("Estimate error as a function of Spectral Resolution")
    plt.xlabel("Spectral Resolution (nm)")
    plt.ylabel("Estimate error (%)")
    plt.legend(["Methane", "Carbon Dioxide", "Water Vapour"])
    plt.show()    