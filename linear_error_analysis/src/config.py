"""
config.py  

Configuration file for the Linear Error Analysis program. All 
config settings can be run through the command line using lea.sh

Author(s): Adyn Miles, Shiqi Xu, Rosie Liang
"""

import argparse

def parse_config():
    """Sets up configurability of inputs to LEA program through terminal.
    """

    parser = argparse.ArgumentParser()

    ### Forward Model

    parser.add_argument("--spectral_lower", type=float, default=1590, 
                        help='wavelength (nm)')
    parser.add_argument("--spectral_upper", type=float, default=1680, 
                        help='wavelength (nm)')
    parser.add_argument("--fwhm", type=float, default=1.5, 
                        help='full-width half maximum for the spectral grid.')
    parser.add_argument("--recalc_xsec", type=bool, default=False, 
                        help='choose whether or not to regather data \
                            from online through HAPI')
    parser.add_argument("--sza", type=float, default=50., help='solar zenith angle')
    parser.add_argument("--vza", type=float, default=0., help='viewing zenith angle')

    # Relevant File Paths
    parser.add_argument("--output_folder", type=str, metavar='PATH', default='./data', 
                        help='specify output path to save lbl data to')
    parser.add_argument("--solar_spectrum", type=str, metavar='PATH', 
                        default='./data/solar_spectrum_merged.dat', 
                        help='enter the path to the solar spectrum .dat file')
    parser.add_argument("--atm_model", type=str, metavar='PATH', 
                        default='./data/prof.AFGL.US.std', 
                        help='enter the path to the atmospheric model .dat file')
    parser.add_argument("--pickle_file", type=str, metavar='PATH', 
                        default='./tmp/optics_prop.pkl', 
                        help='enter the path to the temporary optics pickle file')

    ### Systematic Error Quantification

    # Non-Linearity
    parser.add_argument("--nonlinearity", type=float, default=0.01, 
                        help='expected relative error due to non-linearity.')
    # Pixel Crosstalk
    parser.add_argument("--crosstalk", type=float, default=0.0039, 
                        help='crosstalk between pixels')
    # Stray Light
    parser.add_argument("--fo_reflectivity", type=float, default=0.20, 
                        help='reflectivity of the fore optics')
    parser.add_argument("--lens_reflectivity", type=float, default=0.20, 
                        help='reflectivity of collimating and diverging lenses')
    parser.add_argument("--mirror_reflectivity", type=float, default=0.02, 
                        help='unwanted reflectivity from mirror due to \
                            surface imperfections')
    parser.add_argument("--ar_coatings", type=float, default=0.005, 
                        help='anti-reflective coating unwanted reflectivity')
    parser.add_argument("--leakage", type=float, default=0.10, 
                        help='stray light leakage')
    parser.add_argument("--ghosting", type=float, default=0.00, 
                        help='ghost orders from diffraction grating')
    # Flat Field Response
    parser.add_argument("--uniformity", type=float, default=0.05, 
                        help='uniformity of detector pixel readings')
    # Bad Pixels
    parser.add_argument("--bad_pixels", type=float, default=0.02, 
                        help='percent of pixels expected to be inoperable')
    # Keystone and Smile
    parser.add_argument("--keystone", type=float, default=0.01, 
                        help='keystone distortion error') 
    parser.add_argument("--smile", type=float, default=0.01, 
                        help='smile distortion error')
    # Striping
    parser.add_argument("--striping", type=float, default=0.05, 
                        help='striping distortion error')
    # Memory
    parser.add_argument("--memory", type=float, default=0.02, 
                        help='memory effect pixel error')

    ### Random Error Quantification
    parser.add_argument("--x_pixels", type=int, default=512, 
                        help='number of pixels in the X direction')
    parser.add_argument("--y_pixels", type=int, default=640, 
                        help='number of pixels in the Y direction')
    parser.add_argument("--pixel_pitch", type=int, default=15, 
                        help='pixel pitch for the detector (um)')
    parser.add_argument("--t_int", type=int, default=0.1667, 
                        help='integration time (in s)')
    parser.add_argument("--well_depth", type=int, default=19000, 
                        help='well depth for the detector (in e-)')
    parser.add_argument("--dynamic_range", type=int, default=14, 
                        help='dynamic range of sensor (in bits)')
    parser.add_argument("--dark_current", type=int, default=50, 
                        help='dark current (in nA/cm^2)')
    parser.add_argument("--readout_noise", type=int, default=500, 
                        help='readout noise (in e-)')

    return parser.parse_args()
