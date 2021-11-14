#!/bin/bash

python ./src/main.py \
--temperature 296 \
--pressure 1 \
--wavelength 1580 \ 
--co2_file './hitran_data/co2_line_by_line.par' \
--ch4_file './hitran_data/ch4_line_by_line.par' \
--h2o_file './hitran_data/h2o_line_by_line.par'
