# FINCH Linear Error Analysis

Authors: Adyn Miles, Shiqi Xu, Rosie Liang (UTAT)

With support from Dr. Jochen Landgraf (SRON)

Fall 2021

-----------------------------

## TODO

### `lea.sh`

* update configs to match current state of program

### `photon_noise.py`

* figure out why output array has length 861

### `optim.py`

* loop through photon noise - right now `S_y` (ECM) is using `photon_noise[0]` universally

### `multiple or all`

* distinguish between FWHM in nm and in cm<sup>-1</sup> (`photon_noise` specifically takes input in nm)
* incorporate argument typing
* add documentation in this README
* credit Jochen for modules/excerpts of his code

-----------------------------

## Questions

### For Jochen

* [Shiqi, 2021-11-14] Why is spectral sampling distance half the FWHM? Does this have anything to do with Nyquist?

### For us

* [Shiqi, 2021-11-14] Why is `len(photon_noise)` 861? Double check what indexes the array

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
