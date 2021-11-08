"""
main.py  

Main driver for the Linear Error Analysis program.

Author(s): Adyn Miles, Shiqi Xu, Rosie Liang
"""

import numpy as np
from config import parse_config
from forward import Forward
from optim import Optim

if __name__ == "__main__":
    
    cfg = parse_config()

    forward = Forward(cfg)
    surface, molec, atm, sun_lbl = forward.get_atm_params()
    optics = forward.opt_properties()
    forward.plot_transmittance()
    
    lea = Optim(cfg)
    sys_errors = lea.sys_errors()
    rand_errors = lea.rand_errors()





