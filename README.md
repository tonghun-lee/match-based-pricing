# Pricing in Dynamic Matching Platforms with Strategic Agents

The software and data in this repository are a snapshot of the software and data that were used in the research reported in the paper [*Pricing in Dynamic Matching Platforms with Strategic Agents*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7604120) by Tonghun Lee, Matthew Tsao, and Chiwei Yan.

## Description

The goal of this repository is to replicate the Chicago ridesharing experiments described in Section 5 of the paper.

The repository allows users to:

1. **Learn:** Compute riders' joining and requesting strategies for the myopic benchmark and the two strategic policies.
2. **Simulate:** Compare platform profit and rider outcomes, and reproduce Table 1 and Figures 3 and 4.

## Hardware Configuration

The local setup used for reproducibility checks is:

- **Processor:** 3.80 GHz Intel Core Ultra 7 155H
- **Memory:** 32 GB
- **OS:** Windows 11

## Dependencies

The code uses the following software and Python packages, with versions specified in `environment.yml`.

### Required Software

- **Python:** 3.12.4
- **Conda:** for setting up the environment.

### Python Packages

- `numpy==1.26.4`
- `pandas==2.2.2`
- `scipy==1.17.1`
- `scikit-learn==1.4.2`
- `networkx==3.2.1`
- `pyproj==3.6.1`
- `shapely==2.0.5`
- `geopandas==0.14.2`
- `matplotlib==3.9.2`

## Installation

```bash
conda env create -f environment.yml
conda activate match-based-pricing
```

## Replicating Results

Run the scripts from the repository root in the following order:

1. **`python scripts/1_run_experiments.py`**: builds the rider types and instances, learns the equilibrium strategy profiles (myopic benchmark, σ∞ and σ1) for each dispatch cost and waiting cost, and evaluates each profile over 100 simulation runs.
2. **`python scripts/2_make_table.py`**: writes Table 1 to `results/table1.tex`.
3. **`python scripts/3_make_figures.py`**: draws Figures 3 and 4 and computes the statistics quoted in Section 5.2, in `results/figures/`.

All simulations use fixed random seeds for reproducibility.

## Codebase Structure

### `src/`
- `chicago_data.py`: rider types from the trip data (pickup zones and k-means clustering of drop-off locations).
- `network.py`: shortest-path distances on the road network.
- `instance.py`: compatible pairs, Shapley mileage shares and prices.
- `simulation.py`: discrete-event simulation of the platform.
- `learning.py`: learning and validation of the equilibrium strategy profiles.

### `data/`
- `Chicago_data.csv`: shared-authorized trips in Chicago during Monday morning peak hours (7:30–8:30 a.m., from 2019-10-07 to 2019-11-25), from the [Chicago Data Portal](https://data.cityofchicago.org/d/m6dm-c72p).
- `Chicago_network.pickle`: Chicago road network based on OpenStreetMap (© OpenStreetMap contributors). Source: [Yan et al. (2026)](https://doi.org/10.1287/opre.2023.0513), [software and data](https://doi.org/10.1287/opre.2023.0513.cd).
- `Chicago_zones.pickle`: Chicago community areas and pickup zones. Source: [Yan et al. (2026)](https://doi.org/10.1287/opre.2023.0513), [software and data](https://doi.org/10.1287/opre.2023.0513.cd).

## Acknowledgements

To cite the contents of this repository, please cite the [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7604120).

```bibtex
@misc{lee2026pricing,
  author = {Lee, Tonghun and Tsao, Matthew and Yan, Chiwei},
  title  = {Pricing in Dynamic Matching Platforms with Strategic Agents},
  year   = {2026},
  note   = {SSRN Working Paper},
  doi    = {10.2139/ssrn.7604120},
  url    = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7604120},
}
```

The road network and pickup zones are taken from the software and data snapshot accompanying [*Pricing Shared Rides*](https://doi.org/10.1287/opre.2023.0513) by Chiwei Yan, Julia Yan, and Yifan Shen (*Operations Research*, 2026). If you use them, please cite both the article and its [repository](https://github.com/ORJournal/2023.0513). The repository lists 2025 as the year of its software/data snapshot.

```bibtex
@article{yan2026pricing,
  author  = {Yan, Chiwei and Yan, Julia and Shen, Yifan},
  title   = {Pricing Shared Rides},
  journal = {Operations Research},
  volume  = {74},
  number  = {4},
  pages   = {1834--1857},
  year    = {2026},
  doi     = {10.1287/opre.2023.0513},
}

@misc{yan2025pricingdata,
  author =        {Yan, Chiwei and Yan, Julia and Shen, Yifan},
  publisher =     {Operations Research},
  title =         {{Pricing Shared Rides}},
  year =          {2025},
  doi =           {10.1287/opre.2023.0513.cd},
  note =          {Software and data snapshot},
  url =           {https://github.com/ORJournal/2023.0513},
}
```
