"""
config.py  

Configuration file for the Linear Error Analysis program. All config settings can be run through the command line using lea.sh

Author(s): Adyn Miles, Shiqi Xu, Rosie Liang
"""

import argparse

def parse_config(): 
    parser = argparse.ArgumentParser()

    # Forward Model
    parser.add_argument("--temperature", type=float, default=296, help='temperature (K)')
    parser.add_argument("--pressure", type=float, default=1, help='pressure (atm)')
    parser.add_argument("--spectral_lower", type=float, default=1590, help='wavelength (nm)')
    parser.add_argument("--spectral_upper", type=float, default=1680, help='wavelength (nm)')
    parser.add_argument("--altitude", type=float, default=50, help='altitude (km)')
    parser.add_argument("--co2_file", type=str, metavar='PATH', default='./hitran_data/co2_line_by_line.par', help='Enter the path of the co2_line_by_line.par file')
    parser.add_argument("--ch4_file", type=str, metavar='PATH', default='./hitran_data/ch4_line_by_line.par', help='Enter the path of the ch4_line_by_line.par file')
    parser.add_argument("--h2o_file", type=str, metavar='PATH', default='./hitran_data/h2o_line_by_line.par', help='Enter the path of the h2o_line_by_line.par file')

    return parser.parse_args()
