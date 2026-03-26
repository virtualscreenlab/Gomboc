# Gömböc-Inspired Protein Folding Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.1098/rsos.221594.svg)](https://doi.org/10.1098/rsos.221594)

> **When Does a Funnel Become a Trap? A Gömböc-Inspired Model of Protein Folding Landscapes**

This repository contains the computational framework, data, and analysis scripts for a novel protein folding model inspired by the Gömböc—a convex 3D shape with exactly one stable and one unstable equilibrium point. The model investigates how energy landscape ruggedness affects protein folding kinetics and thermodynamics.

## Overview

The protein folding problem remains a fundamental challenge in biophysics. While energy landscape theory resolves Levinthal's paradox through funnel-shaped landscapes, the quantitative effects of landscape ruggedness are not fully understood. This project introduces a computational model that uses a tunable ruggedness parameter to simulate folding across seven orders of magnitude of landscape ruggedness.

### Key Features

- **Gömböc-Inspired 2D Chain Model**: Closed chain representation with 48 discrete points capturing essential features of unique global energy minima
- **Tunable Ruggedness Parameter**: Systematic exploration from perfectly smooth funnels (r = 0) to highly frustrated landscapes (r = 10⁻³)
- **Hybrid Optimization Strategy**: Two-stage approach combining Conjugate Gradient and BFGS algorithms
- **Benchmarking**: Comparison against Rosetta *ab initio* simulations and AlphaFold3 predictions for the TPR1 domain
- **Comprehensive Energy Function**: Seven additive terms balancing native attraction, geometric constraints, and frustration

## Repository Structure

Gomboc/
├── Algorithm/           # Core folding algorithms and optimization routines
│   ├── gomboc_model.py         # Main simulation engine
│   ├── energy_functions.py     # Potential energy calculations
│   └── optimization.py         # CG and BFGS optimization wrappers
├── Alignment/           # Structural alignment tools and RMSD calculations
│   ├── structural_alignment.py
│   └── rmsd_utils.py
├── AlphaFold/          # AlphaFold3 analysis scripts and results
│   ├── af3_predictions.py
│   ├── pae_analysis.py
│   └── plddt_calculations.py
├── Matlab/             # MATLAB scripts for 3D Gömböc visualization
│   ├── gomboc_3d_generator.m
│   └── surface_plotting.m
├── Rosetta/            # Rosetta ab initio simulation protocols
│   ├── tpr1_abinitio.xml     # RosettaScripts protocol
│   ├── cluster_analysis.py
│   └── energy_landscape.py
└── Statistics/         # Statistical analysis and visualization
├── folding_kinetics.py
├── success_rates.py
└── figure_generation.py
