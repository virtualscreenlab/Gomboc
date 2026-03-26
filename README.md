# Gömböc-Inspired Model of Protein Folding Landscapes

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code and data for the paper, **"When Does a Funnel Become a Trap? A Gömböc-Inspired Model of Protein Folding Landscapes."** The project introduces a novel computational model inspired by the Gömböc—a convex shape with a single stable and a single unstable equilibrium—to investigate how the ruggedness of a protein's energy landscape affects its folding kinetics and success rate.

By systematically tuning a ruggedness parameter across seven orders of magnitude, this model quantitatively demonstrates the transition from a smooth, efficient folding funnel to a rugged, frustrated landscape that leads to kinetic trapping and misfolding. The model's success rates are benchmarked against the folding of the TPR1 domain using Rosetta *ab initio* simulations and AlphaFold3 predictions.

## Key Findings

- **Folding Transition**: A minimal increase in landscape ruggedness (by a factor of `10⁻⁷`) can reduce the folding success rate from 78% to 24-31%.
- **Gömböc Analogy**: The model leverages the mathematical properties of a Gömböc (a single stable state) as a powerful analogy for a well-evolved protein's folding funnel.
- **Benchmarking**: The model's success rates at low ruggedness qualitatively correspond to the native-like coverage achieved by state-of-the-art prediction methods like AlphaFold3 (90% coverage) and Rosetta (77% coverage) on the TPR1 domain.

## Repository Structure

