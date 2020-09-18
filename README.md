# slum_detector

## Overview

Pilot project on multi-city Indian slum recognition from satellite images, supported by the World Bank.

Based on Gechter & Tsivanidis (2020) Spatial Spillovers from Urban Renewal: Evidence from the Mumbai Mills Redevelopment. 
Builds on and extends [slums-world](github.com/mgechter/slums-world).

Researchers: Inhyuk Choi, Michael Gechter, Tim Ohlenburg, Minas Sifakis, Nick Swansson & Nick Tsivanidis.

## Installation
Clone from GitHub. Navigate to the newly created base directory and `pip install requirements.txt`.
Package is under active development; set development mode via `pip install -e` in the base directory. 


## Repo structure
### src/detector
See docstrings of each module for usage.

#### data_prep.py
Load png files that follow slums-world conventions and prepare the data for training/prediction. 
Main script: prepare().

####evaluation.py: 
Compute evaluation metrics of model predictions.
Main script: evaluate().


### tests
Unit and integration tests for each module