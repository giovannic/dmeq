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
solution = solve(default_parameters())
```

`default_parameters` is a dictionary of the model parameters as defined in Griffin et al. 2014.

`solve` is a function which takes a dictionary of model parameters and an optional float dtype parameter to specify precision.

`solution` will be in the form of a 7 x 100 jax array where the first dimension is the output type, (positive microscopy rate, positive PCR rate, incidence, pre-erythrocytic immunity level, clinical immunity level, detection immunity level, proportion of the population), and the second dimension indexes 100 year wide age bands.

### Age grid

`solve` defaults to a uniform 100 x 1 year grid. The grid Griffin fitted on is
finer through infancy, where the immunity recursions move fastest, and is
shipped as `GRIFFIN_AGE_GRID`:

```python
from dmeq import GRIFFIN_AGE_GRID, default_parameters, solve
solution = solve(default_parameters(), age_bins_years=GRIFFIN_AGE_GRID)
```

The quantities `solve` derives from the grid — class widths, midpoints, ageing
rates, the index of the age 20 class — are public as `age_grid(ages)`, and the
demography as `ageing_rates(ages)` and `age_proportions(ages, mu)`.

### Demography

`p['eta']` is the per-day death rate. It is a scalar by default, which leaves the
population exponentially distributed in age, but it accepts one rate per age
class:

```python
from dmeq import deathrates_to_grid, GRIFFIN_AGE_GRID, default_parameters, solve

# per-day death rates over coarse (e.g. five year) bands, given by their upper edges
mu = deathrates_to_grid(age_high, deathrates, GRIFFIN_AGE_GRID)
solution = solve(
    {**default_parameters(), 'eta': mu},
    age_bins_years=GRIFFIN_AGE_GRID
)
```

`deathrates` may carry leading axes, so an `(n_site, n_band)` table maps to
`(n_site, n_age)` in one call, ready to `vmap` `solve` over. `dmeq` ships no
mortality table of its own: demography is a property of the population being
modelled, not of the simulator.

An age varying death rate is not only a reweighting. Individuals leave an age
class at rate `r + eta`, so it moves the immunity recursions and the equilibrium
states as well as the age distribution. Expect fitted output to move.

The terminal age class is absorbing — nobody ages out of it — so its death rate
must be strictly positive, and `age_proportions` refuses a table where it is not.

### Immunity

The immunity block is injectable. It takes the per-day EIR by age class, the age
grid, the rate `re = r + eta` of leaving each class, and the parameters, and
returns an `Immunity` named tuple; the states depend on it only through `foi` and
`phi`.

```python
from dmeq import Immunity, griffin_immunity, solve

solve(p, immunity=griffin_immunity)  # the default
```

The equilibrium state solve, the heterogeneity quadrature and the demography sit
outside the injection point.

## Tests

```bash
pip install -e '.[test]'
pytest
```

The age varying demography is tested against `get_equilibrium_population` in
[malariasimulation](https://github.com/mrc-ide/malariasimulation)'s
`R/population_parameters.R`, which is the definition being ported.


## References

Griffin, Jamie T., Neil M. Ferguson, and Azra C. Ghani (Feb. 11, 2014). “Estimates of the changing age-burden of Plasmodium falciparum malaria disease in sub-Saharan Africa”. In: Nature Communications 5.1. Number: 1 Publisher: Nature Publishing Group, p. 3136. issn: 2041-1723. doi: 10.1038/ncomms4136. url: https://www.nature.com/articles/ncomms4136 (visited on 06/13/2022).

Griffin, Jamie T. (July 26, 2016). “Is a reproduction number of one a threshold for Plasmodium falciparum malaria elimination?” In: Malaria Journal 15.1, p. 389. issn: 1475-2875. doi: 10. 1186/s12936-016-1437-9. url: https://doi.org/10.1186/s12936-016-1437-9 (visited on 06/21/2021).
