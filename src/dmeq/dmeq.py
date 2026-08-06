from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.lax import fori_loop, scan
from jax import vmap

def _resolve_dtype(dtype):
    """Float dtype to solve in.

    ``None`` means "whatever JAX is configured for": float64 when the caller has
    enabled ``jax_enable_x64``, float32 otherwise. Importing this module never
    changes that setting -- it is the caller's to make.
    """
    if dtype is None:
        return jnp.result_type(float)
    return jnp.dtype(dtype)

def _default_parameters():
    return {
        'EIR': 33.,
        'ft': 0.,
        'eta': 0.0001305,
        'rho': 0.85,
        'a0': 2920.,
        's2': 1.67,
        'rA': 0.00512821,
        'rT': 0.2,
        'rD': 0.2,
        'rU': 0.00906627,
        'rP': 0.2,
        'dE': 12.,
        'tl': 12.5,
        'cD': 0.0676909,
        'cT': 0.0034482,
        'cU': 0.006203,
        'g_inf':1.82425,
        'd1': 0.160527,
        'dd': 3650.,
        'ID0': 1.577533,
        'kd': 0.476614,
        'ud': 9.44512,
        'ad0': 8001.99,
        'fd0': 0.007055,
        'gd': 4.8183,
        'aA': 0.757,
        'aU': 0.186,
        'b0': 0.590076,
        'b1': 0.5,
        'db': 3650.,
        'IB0': 43.8787,
        'kb': 2.15506,
        'ub': 7.19919,
        'phi0': 0.791666,
        'phi1': 0.000737,
        'dc': 10950.,
        'IC0': 18.02366,
        'kc': 2.36949,
        'uc': 6.06349,
        'PM': 0.774368,
        'dm': 67.6952,
        'tau': 10.,
        'mu': 0.132,
        'f': 0.33333333,
        'Q0': 0.92
    }

# -- age grid and demography --------------------------------------------------

class AgeGrid(NamedTuple):
    """Everything derived from the age class boundaries alone.

    ``years`` are the lower edges of the model age classes; class ``i`` spans
    ``[years[i], years[i+1])`` and the last class is an absorbing
    ``[years[-1], inf)`` compartment with no ageing out of it. Rates elsewhere
    in the model are per day, so the day-scaled quantities are what the solver
    actually uses.
    """
    years: jnp.ndarray        # (n_age,) lower edge of each class, years
    days: jnp.ndarray         # (n_age,) lower edge of each class, days
    widths: jnp.ndarray       # (n_age - 1,) width of each non-terminal class, days
    midpoints: jnp.ndarray    # (n_age,) class midpoint in days; terminal = its lower edge
    ageing: jnp.ndarray       # (n_age,) ageing rate per day, terminal class zero
    age20: jnp.ndarray        # index of the class closest to 20 years

def ageing_rates(ages, dtype=None):
    """Per-day rate of ageing out of each age class: ``1/diff``, terminal zero.

    ``ages`` are class lower edges in years. The terminal class is absorbing, so
    nobody ages out of it and its rate is zero -- which makes its death rate the
    only outflow (see :func:`age_proportions`).
    """
    dtype = _resolve_dtype(dtype)
    age_days = jnp.asarray(ages, dtype=dtype) * 365.
    return jnp.append(1. / jnp.diff(age_days), jnp.zeros((), dtype=dtype))

def age_proportions(ages, mu, dtype=None):
    """Equilibrium proportion of the population in each age class.

    Ports ``get_equilibrium_population`` from malariasimulation's
    ``R/population_parameters.R``: inflow from the class below divided by the
    outflow of ageing plus death,

        pop[0] = b / (r[0] + mu[0])
        pop[i] = pop[i-1] * r[i-1] / (r[i] + mu[i])

    The recursion is linear in the birth rate ``b`` and only proportions are
    wanted, so ``b = 1`` and the result is normalised -- no root finding, and the
    whole thing stays differentiable. ``mu`` is a per-day death rate, either a
    scalar (uniform mortality, exponential age distribution) or one value per age
    class. It must be strictly positive in the terminal class, which has ``r =
    0`` and so would otherwise accumulate the whole population.
    """
    dtype = _resolve_dtype(dtype)
    r = ageing_rates(ages, dtype=dtype)
    mu = jnp.broadcast_to(jnp.asarray(mu, dtype=dtype), r.shape)
    _check_terminal_mortality(mu)
    pop_0 = 1. / (r[0] + mu[0])
    _, pop_rest = scan(
        lambda prev, x: (prev * x[0] / (x[1] + x[2]),) * 2,
        pop_0,
        (r[:-1], r[1:], mu[1:])
    )
    pop = jnp.append(pop_0, pop_rest)
    return pop / jnp.sum(pop)

def _check_terminal_mortality(mu):
    """Reject a zero terminal death rate, when it can be seen at trace time."""
    if isinstance(mu, jax.core.Tracer):
        return  # a traced value cannot be inspected; the caller owns this one
    if not float(mu[-1]) > 0.:
        raise ValueError(
            'the terminal age class must have a strictly positive death rate: '
            'it is absorbing, so death is the only way out of it'
        )

def deathrates_to_grid(age_high, deathrates, ages):
    """Map death rates given over coarse age bands onto the model age classes.

    ``age_high`` are the upper edges (years, ascending) of the bands the rates
    are supplied over -- WorldPop/WPP style five year bands, say -- and
    ``deathrates`` holds one per-day rate per band along its trailing axis.
    Leading axes are carried through, so an ``(n_site, n_band)`` table maps to
    ``(n_site, n_age)`` without a ``vmap``.

    The mapping is the step function R's ``.bincode`` applies in
    ``mortality_processes.R``, but applied to whole age classes rather than to
    individual ages: a class takes the rate of the band containing its lower
    edge, so a class starting exactly on a band boundary belongs to the band
    above. Classes past the last band edge take the last band's rate.
    """
    age_high = jnp.asarray(age_high)
    deathrates = jnp.asarray(deathrates)
    idx = jnp.searchsorted(age_high, jnp.asarray(ages), side='right')
    return deathrates[..., jnp.minimum(idx, age_high.shape[-1] - 1)]

def age_grid(ages, dtype=None):
    dtype = _resolve_dtype(dtype)
    years = jnp.asarray(ages, dtype=dtype)
    days = years * 365.
    widths = jnp.diff(days)
    return AgeGrid(
        years=years,
        days=days,
        widths=widths,
        midpoints=jnp.append(days[:-1] + widths / 2., days[-1]),
        ageing=jnp.append(1. / widths, jnp.zeros((), dtype=dtype)),
        age20=jnp.argmin(jnp.abs(years - 20.))
    )

# -- immunity -----------------------------------------------------------------

class Immunity(NamedTuple):
    """What the equilibrium state solve needs from an immunity model, plus the
    immunity levels themselves for plotting and diagnostics.

    Only ``foi`` and ``phi`` feed the states; ``q`` and ``cA`` are read off
    afterwards for detectability and onward infectiousness. A replacement
    immunity model can be checked structurally against this.
    """
    foi: jnp.ndarray   # force of infection per day
    phi: jnp.ndarray   # probability an infection is clinical
    q: jnp.ndarray     # probability an asymptomatic infection is detected by microscopy
    cA: jnp.ndarray    # onward infectiousness of state A
    b: jnp.ndarray     # probability an infectious bite infects
    ib: jnp.ndarray    # pre-erythrocytic immunity
    ic: jnp.ndarray    # acquired clinical immunity
    id_: jnp.ndarray   # detection immunity
    icm: jnp.ndarray   # maternal clinical immunity

def griffin_immunity(eps, grid, re, p):
    """Griffin's immunity model: entomological inoculation rate to ``Immunity``.

    ``eps`` is the per-day EIR experienced by each age class (already scaled by
    the heterogeneity node and the relative biting rate), ``grid`` the
    :class:`AgeGrid`, ``re`` the per-day rate of leaving an age class by ageing
    or death. Immunity depends on the model states not at all, so this is the
    whole of the model upstream of the equilibrium solve; the states depend on it
    only through ``foi`` and ``phi``.

    The age grid is needed and not just ``re``, because maternal immunity ``icm``
    is a decaying share of the clinical immunity of a 20 year old.
    """
    # calculate pre-erythrocytic immunity IB
    ib = _calculate_immunity(eps, p['ub'], p['db'], re)

    b = p['b0']*(p['b1'] + (1-p['b1'])/(1+(ib/p['IB0'])**p['kb']))

    # calculate clinical immunity IC
    foi = b * eps

    # calculate probability that an asymptomatic infection (state A) will be
    # detected by microscopy
    ic = _calculate_immunity(foi, p['uc'], p['dc'], re)
    id_ = _calculate_immunity(foi, p['ud'], p['dd'], re)
    fd = 1 - (1-p['fd0'])/(1 + (grid.midpoints/p['ad0'])**p['gd'])
    q = p['d1'] + (1-p['d1'])/(1 + (id_/p['ID0'])**p['kd']*fd)

    # calculate onward infectiousness to mosquitoes
    cA = p['cU'] + (p['cD']-p['cU'])*q**p['g_inf']

    # calculate maternal clinical immunity,
    # assumed to be at birth a proportion of the acquired immunity of a
    # 20 year old
    icm = jnp.append(
        ic[grid.age20] * p['PM'] * p['dm'] / grid.widths * (
            jnp.exp(-grid.days[:-1] / p['dm']) - jnp.exp(-grid.days[1:] / p['dm'])
        ),
        0.
    )

    # calculate probability of acquiring clinical disease as a function of
    # different immunity types
    phi = p['phi0']*(p['phi1'] + (1-p['phi1'])/(
        1 + ((ic+icm)/p['IC0'])**p['kc']
    ))

    return Immunity(
        foi=foi, phi=phi, q=q, cA=cA, b=b, ib=ib, ic=ic, id_=id_, icm=icm
    )

# -- solver -------------------------------------------------------------------

def _solve(
    p,
    dtype=None,
    age_bins_years=None,
    gh_nodes=None,
    gh_weights=None,
    immunity=griffin_immunity,
):
    """Equilibrium solution of the Griffin model.

    ``p['eta']`` is the per-day death rate: a scalar for a uniform rate, or one
    value per age class (see :func:`deathrates_to_grid`). It sets both the age
    distribution of the population and, through ``r + eta``, the rate at which
    individuals leave every age class, so an age varying rate moves the immunity
    recursions and the equilibrium states and not only the age weighting.

    ``immunity`` is the immunity model, defaulting to :func:`griffin_immunity`;
    a replacement takes ``(eps, grid, re, p)`` and returns an :class:`Immunity`.
    """
    dtype = _resolve_dtype(dtype)
    if age_bins_years is None:
        ages = jnp.arange(100, dtype=dtype)
    else:
        ages = jnp.asarray(age_bins_years, dtype=dtype)
    nodes = jnp.array([
        -4.8594628,
        -3.5818235,
        -2.4843258,
        -1.4659891,
        -0.4849357,
        0.4849357,
        1.4659891,
        2.4843258,
        3.5818235,
        4.8594628
    ], dtype=dtype)
    weights = jnp.array([
        4.310653e-06,
        7.580709e-04,
        1.911158e-02,
        1.354837e-01,
        3.446423e-01,
        3.446423e-01,
        1.354837e-01,
        1.911158e-02,
        7.580709e-04,
        4.310653e-06
    ], dtype=dtype)
    if gh_nodes is not None:
        nodes = jnp.asarray(gh_nodes, dtype=dtype)
    if gh_weights is not None:
        weights = jnp.asarray(gh_weights, dtype=dtype)
    grid = age_grid(ages, dtype=dtype)

    # calculate proportion in each age group
    prop = age_proportions(ages, p['eta'], dtype=dtype)

    # rate of leaving an age class: ageing plus death
    re = grid.ageing + jnp.asarray(p['eta'], dtype=dtype)

    # calculate relative biting rate
    psi = 1. - p['rho'] * jnp.exp(-grid.midpoints/p['a0'])

    # calculate EIR scaling factor over Gaussian quadrature nodes
    zeta = jnp.exp(-p['s2']*.5 + jnp.sqrt(p['s2'])*nodes)

    # vmap over the quadrature nodes only; everything else (including dtype) is
    # closed over, so no non-array argument has to travel through vmap.
    het_prev = vmap(
        lambda zeta_i: _non_het_prev(
            grid,
            prop,
            re,
            psi,
            p,
            zeta_i,
            immunity,
            dtype
        )
    )
    return jnp.append(
        jnp.sum(
            het_prev(zeta) * jnp.expand_dims(weights, [1,2]),
            axis = 0
        ), # prev and incidence statistics
        jnp.expand_dims(prop, 0), # proportions
        axis = 0
    )

def _non_het_prev(
        grid,
        prop,
        re,
        psi,
        p,
        zeta,
        immunity,
        dtype
    ):
    # per-day EIR experienced by each age class at this heterogeneity node
    eps = p['EIR']/365. * zeta * psi

    imm = immunity(eps, grid, re, p)
    foi, phi = imm.foi, imm.phi

    # calculate equilibrium solution of all model states.

    # calculate beta values
    betaT = p['rT'] + re
    betaD = p['rD'] + re
    betaA = foi*phi + p['rA'] + re
    betaU = foi + p['rU'] + re
    betaP = p['rP'] + re

    # calculate a and b values
    aT = p['ft'] * phi*foi/betaT
    aP = p['rT'] * aT/betaP
    aD = (1-p['ft'])*phi*foi/betaD

    r = grid.ageing
    states = jnp.zeros((6, grid.years.size), dtype=dtype)
    states = states.at[:,0].set(
        _compute_state(0, 0, 0, 0, 0, 0, betaT,
              betaD, betaP, betaA, betaU, aT, aD, aP, phi, foi, prop, p)
    )

    # calculate states
    states = fori_loop(
        1,
        states.shape[1],
        lambda i, a: a.at[:, i].set(
            _next_state(states, i, betaT, betaD, betaP, betaA, betaU,
                aT, aD, aP, phi, foi, prop, p, r)
        ),
        states
    )

    q = imm.q

    # calculate prevalence/incidence
    pos_M = states[0] + states[1] + states[3] * q
    pos_PCR = states[0] + states[1] + states[3] * (q**p['aA']) + states[4] * (q**p['aU'])
    inc = (states[5] + states[4] + states[3]) * foi * phi

    # stack the return values
    return jnp.stack([pos_M, pos_PCR, inc, imm.b, phi, q])


def _calculate_immunity(foi, rate, delay, re):
    init_imm = (foi[0]/(foi[0] * rate + 1))/(1/delay + re[0])
    _, other_imm = scan(
        lambda prev_imm, i: (
            _next_immunity(foi[i], rate, re[i], prev_imm, delay),
        ) * 2,
        init_imm,
        jnp.arange(1, len(foi))
    )
    return jnp.append(init_imm, other_imm)

def _next_immunity(foi, rate, re, imm, delay):
    return (foi/(foi * rate + 1) + re*imm)/(1/delay + re)

def _next_state(states, i, betaT, betaD, betaP, betaA, betaU, aT, aD, aP, phi,
        foi, prop, p, r):
    bT = r[i-1]*states[0,i-1]/betaT[i]
    bD = r[i-1]*states[1,i-1]/betaD[i]
    bP = p['rT']*bT + r[i-1]*states[2,i-1]/betaP[i]
    rA = r[i-1]*states[3,i-1]
    rU = r[i-1]*states[4,i-1]
    return _compute_state(i, bT, bD, bP, rA, rU, betaT,
              betaD, betaP, betaA, betaU, aT, aD, aP, phi, foi, prop, p)


def _compute_state(i, bT, bD, bP, rA, rU, betaT, betaD, betaP, betaA, betaU,
       aT, aD, aP, phi, foi, prop, p):
    Y = (prop[i] - (bT + bD + bP))/(1 + aT[i] + aD[i] + aP[i])
    D = aD[i] * Y + bD
    A = (rA + (1-phi[i])*Y*foi[i] + p['rD']*D)/(
            betaA[i] + (1-phi[i])*foi[i])
    U = (rU + p['rA']*A)/betaU[i]
    return jnp.array([
        aT[i] * Y + bT, #T
        D, #D
        aP[i] * Y + bP, #P
        A, #A
        U, #U
        Y - A - U #S
    ])
