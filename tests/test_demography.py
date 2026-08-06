"""Age grid, age varying mortality, and ingestion of a coarse mortality table."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmeq import (
    GRIFFIN_AGE_GRID,
    age_grid,
    age_proportions,
    ageing_rates,
    deathrates_to_grid,
    default_parameters,
    solve,
)

GRIFFIN = np.asarray(GRIFFIN_AGE_GRID, dtype=np.float64)
UNIFORM = np.arange(100, dtype=np.float64)


def synthetic_mortality(n_site=5):
    """A U shaped per-day hazard on the Griffin grid, one row per site.

    Stands in for a real mortality table where only the shape matters: infant
    mortality falling away over the first few years, a flat middle, and a tail
    rising with age. Nothing here is calibrated to anywhere; the tests below ask
    that ``eta`` may vary over the grid, not that these are the right numbers.
    The terminal class is left with a positive rate, which the recursion
    requires.

    Scaled to land the sites between a mean age of 33 and one of 20, with 10 to
    22 percent under 5, which is the range the real tables sat in.
    """
    base = (
        5e-5
        + 6e-4 * np.exp(-GRIFFIN / 2.)      # infant hump
        + 5e-6 * np.exp(GRIFFIN / 20.)      # senescence
    )
    return base * np.linspace(0.5, 1.5, n_site)[:, None]


def test_ageing_rates_are_one_over_the_class_width_in_days():
    r = np.asarray(ageing_rates(GRIFFIN, dtype='float64'))
    np.testing.assert_allclose(
        r[:-1], 1. / (np.diff(GRIFFIN) * 365.), rtol=1e-14
    )
    assert r[-1] == 0.  # terminal class is absorbing


def test_age_grid_publishes_what_solve_used_to_inline():
    grid = age_grid(GRIFFIN, dtype='float64')
    days = GRIFFIN * 365.
    np.testing.assert_array_equal(np.asarray(grid.days), days)
    np.testing.assert_allclose(np.asarray(grid.widths), np.diff(days), rtol=1e-14)
    np.testing.assert_allclose(
        np.asarray(grid.midpoints)[:-1], days[:-1] + np.diff(days) / 2., rtol=1e-14
    )
    assert np.asarray(grid.midpoints)[-1] == days[-1]
    assert GRIFFIN[int(grid.age20)] == 20.


def test_age_proportions_reproduce_the_exponential_distribution():
    """A uniform death rate on a uniform grid leaves a geometric distribution.

    Which is the recursion's discrete stand-in for the exponential: the class to
    class ratio is a constant ``r / (r + eta)``, and the mean age comes out near
    ``1 / eta``. The discretisation is off the continuous exponential by a couple
    of percent per class at this width, so the ratio is what to pin down.
    """
    eta = default_parameters()['eta']
    prop = np.asarray(age_proportions(UNIFORM, eta, dtype='float64'))
    assert prop.sum() == pytest.approx(1., abs=1e-14)

    r = 1. / 365.
    ratios = prop[1:-1] / prop[:-2]  # terminal class is absorbing, so excluded
    np.testing.assert_allclose(ratios, r / (r + eta), rtol=1e-12)

    midpoints = np.append(UNIFORM[:-1] + 0.5, UNIFORM[-1])
    assert (prop * midpoints).sum() == pytest.approx(1. / (eta * 365.), rel=0.05)


def test_terminal_class_needs_a_positive_death_rate():
    mu = np.full(GRIFFIN.size, 1e-4)
    mu[-1] = 0.
    with pytest.raises(ValueError, match='terminal age class'):
        age_proportions(GRIFFIN, mu, dtype='float64')


def test_terminal_class_holds_the_whole_tail():
    """Nobody ages out of the last class, so its outflow is death alone."""
    mu = np.full(GRIFFIN.size, 2e-4)
    prop = np.asarray(age_proportions(GRIFFIN, mu, dtype='float64'))
    r = np.asarray(ageing_rates(GRIFFIN, dtype='float64'))
    assert prop[-1] == pytest.approx(prop[-2] * r[-2] / mu[-1], rel=1e-12)


def test_deathrates_to_grid_is_a_step_function():
    # three bands: [0, 5), [5, 10), [10, inf)
    age_high = np.array([5., 10., np.inf])
    rates = np.array([3e-5, 1e-5, 4e-4])
    ages = np.array([0., 1., 4.9, 5., 7., 10., 20.])
    mu = np.asarray(deathrates_to_grid(age_high, rates, ages))
    np.testing.assert_array_equal(
        mu, [3e-5, 3e-5, 3e-5, 1e-5, 1e-5, 4e-4, 4e-4]
    )


def test_deathrates_to_grid_broadcasts_over_leading_axes():
    age_high = np.array([5., np.inf])
    rates = np.array([[1., 2.], [3., 4.], [5., 6.]])  # (n_site, n_band)
    mu = np.asarray(deathrates_to_grid(age_high, rates, np.array([0., 6.])))
    np.testing.assert_array_equal(mu, [[1., 2.], [3., 4.], [5., 6.]])


def test_deathrates_to_grid_round_trips_the_fine_table():
    """A table already given per model class maps to itself."""
    table = synthetic_mortality()
    age_high = np.append(GRIFFIN[1:], np.inf)
    mu = np.asarray(deathrates_to_grid(age_high, table, GRIFFIN))
    np.testing.assert_array_equal(mu, table)


def test_solve_accepts_a_per_class_death_rate():
    p = default_parameters()
    out = np.asarray(solve(
        {**p, 'eta': synthetic_mortality()[0]},
        dtype='float64',
        age_bins_years=GRIFFIN
    ))
    assert out.shape == (7, GRIFFIN.size)
    assert np.all(np.isfinite(out))
    assert out[-1].sum() == pytest.approx(1., abs=1e-12)


def test_age_varying_mortality_moves_more_than_the_age_weighting():
    """``re = r + eta`` feeds the immunity recursions, not just the weighting."""
    p = default_parameters()
    uniform = np.asarray(solve(p, dtype='float64', age_bins_years=GRIFFIN))
    varying = np.asarray(solve(
        {**p, 'eta': synthetic_mortality()[0]},
        dtype='float64',
        age_bins_years=GRIFFIN
    ))
    # rows 0-2 are prevalence and incidence, weighted by the age proportions;
    # rows 3-5 (b, phi, q) are per-person and depend on eta only through re.
    assert not np.allclose(uniform[3:6], varying[3:6], rtol=1e-6)


def test_solve_vmaps_over_sites():
    table = synthetic_mortality()
    p = default_parameters()
    out = jax.vmap(lambda mu: solve(
        {**p, 'eta': mu}, dtype='float64', age_bins_years=GRIFFIN
    ))(jnp.asarray(table))
    assert out.shape == (table.shape[0], 7, GRIFFIN.size)
    assert np.all(np.isfinite(np.asarray(out)))
    single = np.asarray(solve(
        {**p, 'eta': table[3]}, dtype='float64', age_bins_years=GRIFFIN
    ))
    np.testing.assert_allclose(np.asarray(out[3]), single, rtol=1e-12)


def test_solve_is_differentiable_in_an_age_varying_death_rate():
    p = default_parameters()
    g = jax.grad(lambda mu: jnp.sum(solve(
        {**p, 'eta': mu}, dtype='float64', age_bins_years=GRIFFIN
    )[2]))(jnp.asarray(synthetic_mortality()[0]))
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.any(np.asarray(g) != 0.)