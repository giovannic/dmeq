# DMEQ

## Overview

This package provides an implementation of the differentiable equilibrium
solution for Imperial College London's Malaria transmission model. A
derivation of the solution can be found in the supplementary material of
Griffin et al. (2016).

## Installation

Install from github using:

```bash
pip install git+https://github.com/giovannic/dmeq.git
```

Install from local source using:

```bash
pip install .
```

## Usage

```python
from dmeq import default_parameters, solve
solution = solve(default_parameters)
```

`default_parameters` is a dictionary of the model parameters as defined in Griffin et al. 2014.

`solve` is a function which takes a dictionary of model parameters and an optional float dtype parameter to specify precision.

`solution` will be in the form of a 5 x 100 jax array where the first dimension is the output type, (positive microscopy rate, incidence, pre-erythrocytic immunity level, clinical immunity level, detection immunity level), and the second dimension indexes 100 year wide age bands.


## References

Griffin, Jamie T., Neil M. Ferguson, and Azra C. Ghani (Feb. 11, 2014). “Estimates of the changing age-burden of Plasmodium falciparum malaria disease in sub-Saharan Africa”. In: Nature Communications 5.1. Number: 1 Publisher: Nature Publishing Group, p. 3136. issn: 2041-1723. doi: 10.1038/ncomms4136. url: https://www.nature.com/articles/ncomms4136 (visited on 06/13/2022).

Griffin, Jamie T. (July 26, 2016). “Is a reproduction number of one a threshold for Plasmodium falciparum malaria elimination?” In: Malaria Journal 15.1, p. 389. issn: 1475-2875. doi: 10. 1186/s12936-016-1437-9. url: https://doi.org/10.1186/s12936-016-1437-9 (visited on 06/21/2021).
