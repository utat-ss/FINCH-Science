# FINCH Linear Error Analysis

Authors: Adyn Miles, Shiqi Xu, Rosie Liang (UTAT)

With support from Dr. Jochen Landgraf (SRON)

Fall 2021

-----------------------------

## TODO

### `optim.py`

* update `S_y` (ECM) after clarifying `len(photon_noise)` vs `len(self.wave_meas)`
* set up $K$ from `forward.py` output ($F$)

### `multiple or all`

* distinguish between FWHM in nm and in cm<sup>-1</sup> (`photon_noise` specifically takes input in nm)
* incorporate argument typing
* add documentation in this README
* credit Jochen for modules/excerpts of his code

-----------------------------

## Questions

### For Jochen

* [Shiqi, 2021-11-14] Why is spectral sampling distance half the FWHM? Does this have anything to do with Nyquist?
* [Shiqi, 2021-11-21] Can our state vector $x$ be 1 by 1 (i.e. just total column XCH<sub>4</sub>)?
* [Shiqi, 2021-11-21] Can we choose $x_0$ arbitrarily?
* [Shiqi, 2021-11-21] Why is $F_i$ a function of transition wavelength $\lambda_i$?

### For us

* [Shiqi, 2021-11-21] For Adyn: What goes into `photon_noise.py` array length? Currently seems to be constant. Would it not be dependent on spectral res input?

-----------------------------

## Section

### Subsection

Paragraph

(This is a template)

-----------------------------

## Known GTA XCH<sub>4</sub> Concentrations

### Module
`lib/gta_xch4.py`

### Data
`data/ta_20180601_20190930.oof.csv`

### Source
Downloaded from Debra Wunch Dataverse \
https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/RNCAWQ

### Content
Column-Averaged Mixing Ratios of Atmospheric CO<sub>2</sub>, CO, CH<sub>4</sub> \
Toronto, Ontario \
2018-2019

### Objective
To obtain estimates of total column dry mixing ratio of methane (XCH<sub>4</sub>) in the Greater Toronto Area

Satisfied (y/n)?
* Not sure as of 2021-11-06; Shiqi to double check whether (column-averaged == total column)
* Yes as of 2021-11-13; column-averaged is total column divided through by dry air column

### Findings
`plots/GTA_XCH4_2018-19_hist.png`
![](plots/GTA_XCH4_2018-19_hist.png)

XCH<sub>4</sub> (column-averaged)
* Mean: 1.87 ppm
* Std dev: 0.01 ppm
