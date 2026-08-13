"""The immunity block is separable from the equilibrium state solve."""

import jax.numpy as jnp
import numpy as np
import pytest

from dmeq import (
    GRIFFIN_AGE_GRID,
    Immunity,
    age_grid,
    age_proportions,
    calculate_immunity,
    default_parameters,
    griffin_immunity,
    solve,
)

GRIFFIN = np.asarray(GRIFFIN_AGE_GRID, dtype=np.float64)


def _inputs(ages=GRIFFIN, p=None):
    p = default_parameters() if p is None else p
    grid = age_grid(ages, dtype='float64')
    re = grid.ageing + p['eta']
    psi = 1. - p['rho'] * jnp.exp(-grid.midpoints / p['a0'])
    eps = p['EIR'] / 365. * psi
    return eps, grid, re, p


def test_default_injection_is_the_incumbent():
    p = default_parameters()
    np.testing.assert_array_equal(
        np.asarray(solve(p, dtype='float64', age_bins_years=GRIFFIN)),
        np.asarray(solve(
            p, dtype='float64', age_bins_years=GRIFFIN, immunity=griffin_immunity
        ))
    )


def test_immunity_returns_the_documented_fields():
    imm = griffin_immunity(*_inputs())
    assert isinstance(imm, Immunity)
    # exactly these, and not the internal levels: the contract a candidate
    # satisfies is the incumbent's own return type
    assert Immunity._fields == ('foi', 'phi', 'q', 'b')
    for name in Immunity._fields:
        value = np.asarray(getattr(imm, name))
        assert value.shape == GRIFFIN.shape, name
        assert np.all(np.isfinite(value)), name


def test_immunity_depends_on_the_states_not_at_all():
    """Its only inputs are eps, the grid, re and the parameters.

    Nothing downstream of the state solve reaches back into it, so calling it
    directly reproduces exactly what ``solve`` uses internally: the b/phi/q rows
    of a single-node solve are the immunity function's own outputs.
    """
    p = default_parameters()
    eps, grid, re, _ = _inputs(p=p)
    imm = griffin_immunity(eps, grid, re, p)
    # one quadrature node at zeta = 1 (node 0 of a degenerate rule, weight 1)
    out = np.asarray(solve(
        {**p, 's2': 1e-300},
        dtype='float64',
        age_bins_years=GRIFFIN,
        gh_nodes=np.zeros(1),
        gh_weights=np.ones(1)
    ))
    np.testing.assert_allclose(out[3], np.asarray(imm.b), rtol=1e-9)
    np.testing.assert_allclose(out[4], np.asarray(imm.phi), rtol=1e-9)
    np.testing.assert_allclose(out[5], np.asarray(imm.q), rtol=1e-9)


def test_maternal_immunity_uses_clinical_immunity_at_age_twenty():
    """icm needs the age grid, not just re.

    icm is internal to the immunity model rather than a field of the contract,
    so it is checked through `phi`, which is the only way it reaches the solve.
    Rebuilding it here from the public recursion is the point: it pins that the
    maternal term is the ic-at-20 decay and that it is load-bearing in `phi`.
    """
    eps, grid, re, p = _inputs()
    imm = griffin_immunity(eps, grid, re, p)
    days = np.asarray(grid.days)

    ic = np.asarray(calculate_immunity(imm.foi, p['uc'], p['dc'], re))
    icm = np.append(
        float(ic[grid.age20]) * p['PM'] * p['dm'] / np.asarray(grid.widths)
        * (np.exp(-days[:-1] / p['dm']) - np.exp(-days[1:] / p['dm'])),
        0.
    )
    assert icm[0] > 0.
    assert icm[-1] == 0.
    # decays over a couple of months, so it is negligible by school age
    assert icm[np.searchsorted(GRIFFIN, 5.)] < 1e-6 * icm[0]

    def clinical(maternal):
        return p['phi0'] * (p['phi1'] + (1 - p['phi1']) / (
            1 + ((ic + maternal) / p['IC0']) ** p['kc']
        ))

    np.testing.assert_allclose(np.asarray(imm.phi), clinical(icm), rtol=1e-9)
    # and it changes `phi` in infancy, so dropping the grid would not be free
    assert float(imm.phi[0]) == pytest.approx(clinical(icm)[0], rel=1e-9)
    assert abs(float(imm.phi[0]) - clinical(np.zeros_like(icm))[0]) > 1e-3


def test_a_replacement_immunity_model_is_used():
    """A candidate can replace the whole block, keeping states and demography."""
    p = default_parameters()

    def flat_immunity(eps, grid, re, p):
        ones = jnp.ones_like(eps)
        return Immunity(foi=0.5 * eps, phi=0.1 * ones, q=0.3 * ones, b=0.5 * ones)

    out = np.asarray(solve(
        p, dtype='float64', age_bins_years=GRIFFIN, immunity=flat_immunity
    ))
    assert np.all(np.isfinite(out))
    # b, phi and q rows come straight from the replacement, up to the quadrature
    # weights, which sum to 1 only to single precision
    np.testing.assert_allclose(out[3], 0.5, rtol=1e-6)
    np.testing.assert_allclose(out[4], 0.1, rtol=1e-6)
    np.testing.assert_allclose(out[5], 0.3, rtol=1e-6)
    # the demography is outside the injection point and is untouched
    np.testing.assert_allclose(
        out[6],
        np.asarray(age_proportions(GRIFFIN, p['eta'], dtype='float64')),
        rtol=1e-12
    )
    # so is the state solve: everything positive is still a share of the
    # population, and prevalence cannot exceed it
    assert np.all(out[0] <= out[6] + 1e-12)
