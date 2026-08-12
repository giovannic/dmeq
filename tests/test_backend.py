"""The two backends compute the same model, and selecting one imports only it.

Two separate claims, and the second is the one that motivated the split. The
numerical claim is checked in-process by running the same call twice; the import
claim can only be checked in a subprocess, because by the time this file has
been collected the interpreter has long since imported jax.
"""

import subprocess
import sys

import numpy as np
import pytest

from dmeq import (
    GRIFFIN_AGE_GRID,
    Immunity,
    age_grid,
    age_proportions,
    ageing_rates,
    deathrates_to_grid,
    default_parameters,
    griffin_immunity,
    set_backend,
    solve,
    use_backend,
)
from dmeq._backend import backend

GRIFFIN = np.asarray(GRIFFIN_AGE_GRID, dtype=np.float64)
BACKENDS = ('numpy', 'jax')

#: A U shaped per-day hazard, as in test_demography. Exercises the age varying
#: path, which is the one with a scan over a non-constant mu.
MORTALITY = (
    5e-5
    + 6e-4 * np.exp(-GRIFFIN / 2.)
    + 5e-6 * np.exp(GRIFFIN / 20.)
)


def both(call):
    """``call`` run under each backend, as a dict of numpy arrays by name."""
    out = {}
    for name in BACKENDS:
        with use_backend(name):
            out[name] = np.asarray(call())
    return out


# -- the two backends agree ---------------------------------------------------

def test_ageing_rates_agree():
    r = both(lambda: ageing_rates(GRIFFIN, dtype='float64'))
    np.testing.assert_array_equal(r['numpy'], r['jax'])


def test_age_grid_agrees():
    for field in ('years', 'days', 'widths', 'midpoints', 'ageing', 'age20'):
        got = both(lambda: getattr(age_grid(GRIFFIN, dtype='float64'), field))
        np.testing.assert_array_equal(got['numpy'], got['jax'], err_msg=field)


def test_deathrates_to_grid_agrees():
    age_high = np.append(GRIFFIN[1:], np.inf)
    table = MORTALITY * np.linspace(0.5, 1.5, 5)[:, None]
    mu = both(lambda: deathrates_to_grid(age_high, table, GRIFFIN))
    np.testing.assert_array_equal(mu['numpy'], mu['jax'])


@pytest.mark.parametrize('mu', [default_parameters()['eta'], MORTALITY],
                         ids=['uniform', 'age-varying'])
def test_age_proportions_agree(mu):
    """The scan, over a tuple of three arrays.

    Not bitwise: the two libraries are free to contract the multiply-divide
    differently. The bar is that the demography a fit weights by and the one an
    analysis worker reads are the same distribution.
    """
    prop = both(lambda: age_proportions(GRIFFIN, mu, dtype='float64'))
    np.testing.assert_allclose(prop['numpy'], prop['jax'], rtol=1e-13, atol=0.)
    assert prop['numpy'].sum() == pytest.approx(1., abs=1e-14)


def test_griffin_immunity_agrees():
    """The other scan, and the maternal immunity block that indexes by age20."""
    p = default_parameters()

    def call(field):
        def go():
            grid = age_grid(GRIFFIN, dtype='float64')
            re = grid.ageing + p['eta']
            xp = backend().xp
            psi = 1. - p['rho'] * xp.exp(-grid.midpoints / p['a0'])
            eps = p['EIR'] / 365. * psi
            return getattr(griffin_immunity(eps, grid, re, p), field)
        return go

    for field in Immunity._fields:
        got = both(call(field))
        np.testing.assert_allclose(
            got['numpy'], got['jax'], rtol=1e-12, atol=0., err_msg=field
        )


@pytest.mark.parametrize('eta', [default_parameters()['eta'], MORTALITY],
                         ids=['uniform', 'age-varying'])
def test_solve_agrees(eta):
    """The whole thing: vmap over quadrature nodes, fori_loop with set_at inside.

    This is the claim that matters -- a curve read in an analysis worker is the
    curve the fit would have produced.
    """
    p = {**default_parameters(), 'eta': eta}
    out = both(lambda: solve(p, dtype='float64', age_bins_years=GRIFFIN))
    assert out['numpy'].shape == (7, GRIFFIN.size)
    np.testing.assert_allclose(out['numpy'], out['jax'], rtol=1e-13, atol=1e-16)


def test_solve_agrees_on_the_default_uniform_grid():
    """The `age_bins_years=None` path, which builds its own grid with arange."""
    out = both(lambda: solve(default_parameters(), dtype='float64'))
    np.testing.assert_allclose(out['numpy'], out['jax'], rtol=1e-13, atol=1e-16)


def test_a_replacement_immunity_model_runs_on_both():
    """What immunity-only search needs: a candidate that is not the incumbent.

    Written against `backend()` rather than against either library, which is the
    spelling a portable candidate has to use.
    """
    def flat_immunity(eps, grid, re, p):
        xp = backend().xp
        ones = xp.ones_like(eps)
        zeros = xp.zeros_like(eps)
        return Immunity(
            foi=0.5 * eps, phi=0.1 * ones, q=0.3 * ones, cA=0.05 * ones,
            b=0.5 * ones, ib=zeros, ic=zeros, id_=zeros, icm=zeros,
        )

    out = both(lambda: solve(
        default_parameters(), dtype='float64', age_bins_years=GRIFFIN,
        immunity=flat_immunity,
    ))
    assert np.all(np.isfinite(out['numpy']))
    np.testing.assert_allclose(out['numpy'], out['jax'], rtol=1e-13, atol=1e-16)


def test_the_state_loop_reads_the_pre_loop_states_on_both_backends():
    """Characterisation, not approval: `_non_het_prev`'s loop body closes over
    the `states` from before the loop rather than over the loop's own carry, so
    for age class 2 upwards the inflow terms `bT/bD/bP/rA/rU` are read from a
    zero-filled column and come out zero. Every class above the second is
    therefore solved as if nobody aged into it.

    That is what `solve` has always computed and what msinf has fitted, so both
    backends must reproduce it or the refactor has changed the model. Pinned
    here so that a future fix has to be a deliberate one with a decision behind
    it -- see the note in `_non_het_prev`. Fixing it moves microscopy prevalence
    under 5 substantially and clinical incidence barely.
    """
    p = default_parameters()
    for name in BACKENDS:
        with use_backend(name):
            out = np.asarray(solve(
                {**p, 's2': 1e-300}, dtype='float64', age_bins_years=GRIFFIN,
                gh_nodes=np.zeros(1), gh_weights=np.ones(1),
            ))
        # class 1 has a real predecessor (class 0 is set before the loop) and
        # class 2 does not, which is the signature of the frozen closure
        assert out[0, 1] > 0., name
        prev = out[0] / out[6]
        assert prev[:20].max() < 0.35, name  # sequential recursion gives > 0.6


def test_numpy_backend_checks_the_terminal_death_rate():
    """Concrete under numpy, so the guard always fires rather than being skipped."""
    mu = np.full(GRIFFIN.size, 1e-4)
    mu[-1] = 0.
    with use_backend('numpy'):
        with pytest.raises(ValueError, match='terminal age class'):
            age_proportions(GRIFFIN, mu, dtype='float64')


# -- selection ----------------------------------------------------------------

def test_use_backend_restores_what_it_replaced():
    before = backend().name
    with use_backend('numpy'):
        assert backend().name == 'numpy'
        with use_backend('jax'):
            assert backend().name == 'jax'
        assert backend().name == 'numpy'
    assert backend().name == before


def test_the_default_is_jax():
    """Every caller that predates the split is unaffected."""
    assert backend().name == 'jax'


@pytest.mark.parametrize('select', [set_backend, lambda n: use_backend(n).__enter__()])
def test_an_unknown_backend_is_refused_at_the_call(select):
    with pytest.raises(ValueError, match='unknown backend'):
        select('torch')


# -- what none of the above can check in this interpreter ---------------------

def run(code, **env):
    """`code` in a fresh interpreter, returning its stdout."""
    import os
    done = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True, env=os.environ | env,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


#: what to look for in a subprocess's module table
PROBE = "print(sorted(m for m in ('jax', 'jaxlib', 'numpy') if m in sys.modules))"


def test_importing_dmeq_imports_no_backend_at_all():
    """`import dmeq` is pure Python: the driver can introspect it for free."""
    assert run(f'import sys, dmeq\n{PROBE}\n') == '[]'


def test_the_numpy_backend_never_imports_jax():
    """The reason for the split.

    An analysis worker holds a few hundred megabytes of working set; a jax it
    never differentiates through costs it gigabytes of address space.
    """
    code = (
        'import sys, numpy as np, dmeq\n'
        "dmeq.set_backend('numpy')\n"
        'out = dmeq.solve(dmeq.default_parameters(), dtype="float64")\n'
        'assert out.shape == (7, 100) and np.all(np.isfinite(out))\n'
        f'{PROBE}\n'
    )
    assert run(code) == "['numpy']"


def test_the_backend_can_be_selected_by_environment():
    """So a worker can be configured by how it is launched, without a call."""
    code = (
        'import sys, dmeq\n'
        'dmeq.solve(dmeq.default_parameters(), dtype="float64")\n'
        f'{PROBE}\n'
    )
    assert run(code, DMEQ_BACKEND='numpy') == "['numpy']"


def test_the_jax_backend_still_imports_jax():
    """The negative control: the probe above can actually see a backend load."""
    code = (
        'import sys, dmeq\n'
        'dmeq.solve(dmeq.default_parameters(), dtype="float64")\n'
        f'{PROBE}\n'
    )
    assert run(code, DMEQ_BACKEND='jax') == "['jax', 'jaxlib', 'numpy']"
