# Input variables here to run through the LEA Analysis

import argparse

def parse_config(): 
    parser = argparse.ArgumentParser()

    # Forward Model
    parser.add_argument("--temperature", type=float, default=296, help='temperature (K)')
    parser.add_argument("--pressure", type=float, default=1, help='pressure (atm)')
    parser.add_argument("--wavelength", type=float, default=1590, help='wavelength (nm)')
    parser.add_argument("--co2_file", type=str, metavar='PATH', default='./hitran_data/co2_line_by_line.par', help='Enter the path of the co2_line_by_line.par file')
    parser.add_argument("--ch4_file", type=str, metavar='PATH', default='./hitran_data/ch4_line_by_line.par', help='Enter the path of the ch4_line_by_line.par file')
    parser.add_argument("--h2o_file", type=str, metavar='PATH', default='./hitran_data/h2o_line_by_line.par', help='Enter the path of the h2o_line_by_line.par file')

    return parser.parse_args()
