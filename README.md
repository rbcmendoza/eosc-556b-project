# DC resistivity forward modelling to determine detectability of fault offset: A case study on two sites in the Fraser Canyon
_Raul Benjamin Mendoza_

This is my repo for my project and assignment 4 of EOSC 556B

## Project Description
### Motivation
In August of 2024, my colleagues and I conducted DC resistivity surveys at two areas in the Fraser Canyon,
with the objective of finding offsets in the underlying material to determine the presence (or lack thereof)
of active faults. I have inverted the field data into resistivity models using AGI’s EarthImager software. However, under
these specific conditions (i.e. the geology and the terrain) it is unclear to me what deformation from a
fault rupture would look like (or if it is even resolvable given the survey setup).
### Objectives
I intend to use discretize and SimPEG packages from the SimPEG ecosystem (Cockett et al., 2015) to
simulate data with and without layer offset related to faulting and see if the offset produces a detectable
signal in the inverted model.
### Progress
The project is still in Phase 1, wherein I aim to replicate the results from EarthImager using SimPEG. In notebook 1, 
I have imported the topography of a resistivity profile, and I am now working on generating the survey setup using SimPEG's
generate_survey_from_abmn_locations function and a survey data file (.stg format).

I have not started Phase 2, wherein I will introduce fault offset into the model I create in Phase 1. I will compare the predicted data
and the recovered models to see if the fault offset is detectable and/or resolvable. Stay tuned!

## Installation instructions
1. Clone the repository.
2. Create an environment using `conda env create -f environment.yml`.
3. Activate the environment using `conda env activate eosc556b_project`.
4. Open the _phase1.ipynb_ notebook with Jupyter and you're good to go!

## Important components
1. The necessary packages are indicated in _environment.yml_ file.
2. _phase1.ipynb_ contains all of the functional code for now.
3. The _fieldData_ folder contains the field data that will be used by phase1.ipynb. You do not need to manually open anything from this folder.

## References
Cockett, R., Kang, S., Heagy, L. J., Pidlisecky, A., & Oldenburg, D. W. (2015). SimPEG: An open source
framework for simulation and gradient based parameter estimation in geophysical applications.
Computers & Geosciences. https://doi.org/10.1016/j.cageo.2015.09.015
