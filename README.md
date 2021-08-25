# GIR is not an active project and its use is depreciated. It was developed into the [FaIRv2.0.0 model](https://gmd.copernicus.org/articles/14/3007/2021/). Please use the development version of FaIRv2.0.0 for your projects, found [here](https://github.com/njleach/FAIR/tree/v2.0.0-alpha).

Begin old README:

# GIR
A Generalised Impulse-Response modelling framework.

GIR is a maximally reduced simple climate model for exploring globally averaged climate impacts of greenhouse gas (GHG) or aerosol emissions, changes in GHG concentrations, or external forcings. GIR is based on a set of six equations - three gas cycle equations to convert emissions to concentrations, one equation to convert concentrations to effective radiative forcings, and two equations representing a simple energy-balance thermal response. All GHGs (and aerosols) are treated identically by the equations - differences in the gas cycles or atmospheric chemistries of individual species are introduced via the parameters - which allows GIR to be fully parallelised and is therefore highly efficient to run.
## Introduction (in brief)
The gas cycle of GIR is based on the carbon cycle component of the Finite amplitude Impulse Response model (FaIR) [see Millar et al. (2017) and Smith et al. (2017)]. We have made minor alterations to allow the atmospheric chemistry of some gas species (eg. methane) to be more accurately represented, and have replaced the iIRF100 numeric root finding scheme with an analytic approximation. By default, only the carbon gas cycle has more than one sink (we keep the same four carbon sinks as in FaIR).
The concentration-forcing equation in GIR has three terms: logarithmic, linear and square-root. These allow the dominant behaviours of increasing gas concentrations of species with differing spectral saturation levels to be represented faithfully (and can also allow for inclusion of some non-linearity beyond the dominant term).
The thermal response is a simple impulse-response energy balance model [Gregory et al. (2002)]. While the most commonly used version of this model calculate thermal response based on a two-layer system, work has suggested that the responses of complex GCMs can be emulated better by a three-layer system [Tsutsui (2017)], largely to improve the very short timescale response. As such, we keep the number of layer in GIR general, though our default thermal parameter set contains three.
## Installing GIR
GIR is compatible with python 3.5+ (due to the use of starred expressions in some routines).

Currently, GIR is not provided as a conda or pip python module (though we hope to provide this in the future). Users must download or pull the latest version from this [repo](https://github.com/njleach/GIR).
## Running GIR
To use GIR in a python script or jupyter notebook, you must include the following lines in your preamble:
```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/path/to/folder/containing/GIR')
from GIR import *
```
These import all the functions and modules required to run GIR.
## Usage
There are several methods of running GIR, listed below:

| Inputs (optional)  | Outputs |
| ------------- | ------------- |
| Emissions (other forcings)  | Concentrations / Effective radiative forcings / Temperature response  |
| Concentrations (other forcings)  | Diagnosed inverse emissions / Effective radiative forcings / Temperature response  |
| Forcings  | Temperature response  |

(Diagnosed) emissions, concentrations and forcings are outputted individually for distinct forcing agents, while temperature is outputted as a response to the combined forcing.
## Model Structure
The input and output of GIR is built around (multiindexed) pandas dataframes. While these are a little more restrictive than normal dataframes, this enforces full traceability of the outputs, if multiple parameter sets or scenarios are used.
GIR takes time-indexed dataframes for (corresponding) emission/concentration/forcing scenarios; and parameter-indexed dataframes for the (independent) gas cycle and thermal parameter set(s).
GIR converts the input parameter and emission/concentration/forcing dataframes into numpy arrays for calculation, and then back to dataframes for output.
Since GIR runs all scenarios and parameter sets in parallel, it is easy to overload the RAM of your machine. The approximate required RAM is of order (# of timesteps) * (# of scenarios) * (# of gas cycle parameter sets) * (# of gas species) * (# of thermal parameter sets).
## Further information
For more detailed information including some examples of running GIR, please see [here](https://github.com/njleach/GIR/blob/master/GIR/GIR_example_notebook.ipynb).
## Versioning
GIR is currently in version 1.0.0. Versioning is as follows:

| Index  | Change to model | 
| ------------- | ------------- |
| Major | Large scale changes to the model and how it is run. Not necessarily backwards compatible. |
| Intermediate | Changes to the functionality or operation of the model or packaged functions. Backwards compatible.  |
| Minor | Changes to the code structure with no user-end changes. Updates to the default parameter sets. |

# Zenodo DOI

[![DOI](https://zenodo.org/badge/231077183.svg)](https://zenodo.org/badge/latestdoi/231077183)

# License
[CC4.0](https://creativecommons.org/licenses/by/4.0/)
