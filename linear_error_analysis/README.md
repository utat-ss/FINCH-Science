# FINCH Linear Error Analysis

Authors: Adyn Miles, Shiqi Xu, Rosie Liang (University of Toronto Aerospace Team)

With support from Dr. Jochen Landgraf (SRON)

Start Date: October 1, 2021

-----------------------------
## Overview

The Linear Error Analysis tool helps UTAT's Space Systems Division effectively translate between scientific requirements and optical design requirements to provide quantitative measures of a payload profile's usefulness for atmospheric monitoring. More information on the specifics of this program is available in the [Linear Error Analysis Documentation](https://spacesys.utat.ca/confluence/display/FIN/Linear+Error+Analysis+Documentation), as well as the FINCH team's recent [SmallSat publication](https://spacesys.utat.ca/confluence/display/FIN/FINCH+Eye+2022+SmallSat+Publication). 

-----------------------------

## Usage

### main.py

This file will execute the entire program when run. This includes generating the forward model for the investigated molecules (methane, carbon dioxide, and water vapour), convolving them with an instrument response function, and then performing a linear projection onto an estimation space to produce parts per million concentration estimates for methane, carbon dioxide, and water vapour. The program can then compare these estimates to the original state vector (determined through research), and can generate trends based on the variance of select payload parameters. More information on these trends can be found in the [Linear Error Analysis Documentation](https://spacesys.utat.ca/confluence/display/FIN/Linear+Error+Analysis+Documentation).

### config.py

This file contains configurations for payload parameters, atmospheric parameters, as well as parameters that control the operation of the code. Payload and atmospheric parameter inputs and rationale are included in the [Linear Error Analysis Documentation](https://spacesys.utat.ca/confluence/display/FIN/Linear+Error+Analysis+Documentation). The one parameter controlling code operation is `recalc_xsec`, which can be set `True` or `False` depending on whether or not you would like the program to regather data online from Hitran using HAPI. If this has already been done on a previous run, set this to `False` to save significant time on future executions.

### libs/photon_noise.py

This file is a lookup table for the photon noise found at wavelengths within the FINCH Eye's spectral range, based on a Signal to Noise Ratio Analysis. The program uses this to investigate the relative effects of various errors, and to determine the relationship between signal to noise ratio and estimate error.

-----------------------------
