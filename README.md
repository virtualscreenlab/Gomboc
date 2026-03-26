# Gömböc-Inspired Model of Protein Folding Landscapes

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**When Does a Funnel Become a Trap? A Gömböc-Inspired Model of Protein Folding Landscapes**

*Sergey Shityakov, Ekaterina V. Skorb, and Michael Nosonovsky*

---

## Table of Contents

1. [Overview](#overview)
2. [Key Findings](#key-findings)
3. [The Gömböc Analogy](#the-gömböc-analogy)
4. [Repository Structure](#repository-structure)
5. [Getting Started](#getting-started)
6. [Usage](#usage)
7. [Model Description](#model-description)
8. [Results Summary](#results-summary)
9. [Data Availability](#data-availability)
10. [Citation](#citation)
11. [License](#license)
12. [Contact](#contact)
13. [Acknowledgments](#acknowledgments)

---

## Overview

This repository contains the complete code and data for the paper **"When Does a Funnel Become a Trap? A Gömböc-Inspired Model of Protein Folding Landscapes."** We introduce a novel computational model inspired by the Gömböc—a convex three-dimensional shape with exactly one stable and one unstable equilibrium point—as an analogy for protein folding energy landscapes.

The protein folding problem remains a fundamental challenge in biophysics. While energy landscape theory resolves Levinthal's paradox through funnel-shaped landscapes, the quantitative effects of landscape ruggedness on folding kinetics are not fully understood. Our model addresses this gap by:

- Representing protein conformations as closed two-dimensional chains with a unique global energy minimum
- Introducing a tunable ruggedness parameter spanning seven orders of magnitude
- Simulating folding trajectories to map the relationship between landscape topography and folding outcomes
- Benchmarking against the well-characterized TPR1 domain using Rosetta *ab initio* simulations and AlphaFold3 predictions

---

## Key Findings

- **Folding Transition**: At minimal ruggedness (10⁻⁸), the model achieves 78% success in reaching native-like states. Increasing ruggedness by a factor of 10⁻⁷–10⁻⁴ reduces success rates to 20–31%, reflecting kinetic trapping.

- **Optimal Folding**: At zero ruggedness (r = 0.0), the model achieves 93% success in reaching native-like states (RMSD ≤ 0.2), with rapid convergence in 20–30 optimization steps.

- **Benchmark Correspondence**: The model's success rates at low ruggedness qualitatively correspond to:
  - AlphaFold3: 90% native-like coverage at 0.30 Å RMSD
  - Rosetta: 77% native-like coverage at 0.43 Å RMSD

- **Energy Landscape Characterization**: The TPR1 domain exhibits a clear energetic gap of 30–40 REU between native-like conformations (RMSD < 2.0 Å) and misfolded structures, confirming it as a minimally frustrated system.

- **Prediction Accuracy**: AlphaFold3 substantially outperforms Rosetta for the TPR1 domain, achieving 0.30 Å RMSD with 90% coverage compared to Rosetta's 0.43 Å RMSD with 77% coverage.

---

## The Gömböc Analogy

The **Gömböc** (pronounced *goemboets*) is the first known convex 3D shape made of homogeneous material that is monomonostatic—it has exactly one stable and one unstable point of equilibrium. Discovered by Hungarian scientists Gábor Domokos and Péter Várkonyi in 2006, this mathematical curiosity provides a powerful analogy for protein folding:

| Gömböc Property | Protein Folding Analogy |
|-----------------|------------------------|
| Single stable equilibrium | Unique native state (global energy minimum) |
| Returns to stable position when perturbed | Consistent refolding to native structure |
| Multiple pathways to equilibrium | Multiple folding pathways |
| Fast departure from unstable equilibrium | Fast unfolding (~2 orders faster than folding) |

Our model generalizes this concept to higher dimensions and captures essential features of protein folding landscapes while remaining computationally tractable.

---

## Repository Structure
