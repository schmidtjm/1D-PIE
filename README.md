
# 1D-PIE - 1D -- Planetary Interior Evolution model

https://img.shields.io/badge/license-MIT-green

1D-PIE is a 1D interior evolution code (@Julia M. Schmidt) that calculates the thermal evolution of terrestrial planets up to Earth masses M_E=3.0. It encompasses a stagnant lid and mobile lid tectonic regime and includes crust and lithosphere evolution, local mineral/melt partition coefficient calculations, and first-order outgassing of H2O. The 1D model is based on the conservation of energy and runs the evolution of the planets forward in time with assigned starting conditions such as radii, initial temperature conditions, and heat source and volatile abundances.

This repository contains the model described in Schmidt, Vulpius, Brachmann and Noack (under Review), as well as plotting routines and input conditions for rocky solar system planets and model exoplanets from 0.1-3.0 M_E. 


## Installation

We recommend using Conda to manage dependencies and run the code in a Jupyter Notebook environment.

### Step by step

Clone the repository from Github:
```
git clone https://github.com/schmidtjm/1D-PIE.git
cd 1D-PIE
```

Create and activate a new Conda environment **1DPIE**:
```
conda create -n 1DPIE python=3.11
conda activate 1DPIE
```

Install jupyter:
```
conda install scipy jupyter
```

Launch jupyter notebook:
```
jupyter notebook
```

### Required Packages

* Python >= 3.11.5
* numpy
* pandas
* scipy
* statistics
* jupyter

## Getting started

To get started with the model check out ``introduction.ipynb``, where a more detailed description how to set up the model for specific configurations can be found. 


