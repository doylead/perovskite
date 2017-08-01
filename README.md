# Perovskite Screening for Accelerated Catalyst Development
## Authors: 
Andrew Doyle (doylead@stanford.edu)  
Brian Rohr (brohr@stanford.edu)

## Affiliation: 
SUNCAT Center for Interfacial Science & Catalysis  
Stanford University  
Stanford, CA 94305

## Disclaimer:
This represents research in progress.  Some files are intentionally unavailable to the public until this work has been published.  In addition, documentation may be lacking in some areas as code is still in active development and frequently reorganized.

## Objective:
In this work we attempt to design a machine learning model to reproduce and extrapolate on a data set containing calculated adsorption energies for a variety of chemicals on perovskite oxides (compounds with formula ABO3).  We can think of each observation as having three defining characteristics:
* Element A; One of the elements of a perovskite oxide.  A metallic element, often in groups 1, 2, or 3 of the periodic table.
* Element B; The other of the elements in a perovskite oxide.  Any metallic element.
* Adsorbate; The chemical species bound to the surface.  In this work, among the set:  
{'H', 'O', 'OH', 'OOH', 'N', 'NH', 'NH2', 'NNH'}

Given the ability to predict the adsorption energy (a chemical quantity specified by these three variables) for those values we would be able to increase the speed of computational catalyst searches massively, up to 85%.  This would save millions of CPU-Hours per longitudinal class of material and/or reaction considered.  This would make screening work more accessible, more affordable, and those saved resources could then be dedicated to other studies.

At this point our primary challenge is to find a set of physical parameters that can be fed into a model to reproduce the data available.  We are initially focused only on properties with standard tabulated values, as any input from DFT will diminish the effects of the intended acceleration.

## About the Data
Data has been generated through Density Functional Theory (DFT) calculations.  More information on those setups will be provided in publications to be listed here later.

This document last updated: 8/1/2017
