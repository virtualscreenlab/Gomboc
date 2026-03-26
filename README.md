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

## Installation

### Prerequisites

- Python 3.9+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- MATLAB R2020b+ (for 3D visualization)

### Setup

```bash
# Clone the repository
git clone https://github.com/virtualscreenlab/Gomboc.git
cd Gomboc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
