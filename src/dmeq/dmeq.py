from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import fori_loop

dtype = jnp.float32

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

def _solve(p):
    ages = jnp.arange(100, dtype=dtype)
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
    age_days = ages * 365.
    age_diff = jnp.diff(age_days)
    age_days_midpoint = jnp.append(
        age_days[:-1] + age_diff / 2.,
        age_days[-1]
    )
    age20 : int = 20
    r = jnp.append(1. / age_diff, 0.)

    # calculate proportion in each age group
    prop = jnp.full(ages.size, p['eta'] / (r + p['eta']), dtype=dtype)
    prop = fori_loop(
        1,
        prop.size,
        lambda i, a: a.at[i].set(r[i-1]*a[i-1]/(r[i]+p['eta'])),
        prop
    )

    # calculate relative biting rate
    psi = 1. - p['rho'] * jnp.exp(-age_days_midpoint/p['a0'])

    # calculate EIR scaling factor over Gaussian quadrature nodes
    zeta = jnp.exp(-p['s2']*.5 + jnp.sqrt(p['s2'])*nodes)

    prev = jnp.zeros((2, len(ages)), dtype=dtype) # prevalence and incidence

    # TODO: vectorize instead of loop
    return jnp.append(
        fori_loop(
            0,
            len(zeta),
            lambda i, a: a + _non_het_prev(
                    age_days,
                    age_days_midpoint,
                    age_diff,
                    prop,
                    r,
                    psi,
                    age20,
                    p,
                    zeta[i]
                ) * weights[i],
            prev
        ),
        jnp.expand_dims(prop, 0),
        axis = 0
    )

def _non_het_prev(
        age_days,
        age_days_midpoint,
        age_diff,
        prop,
        r,
        psi,
        age20,
        p,
        zeta
    ):
    # rate of ageing plus death
    re = r + p['eta']

    # calculate pre-erythrocytic immunity IB
    eps = p['EIR']/365. * zeta * psi
    ib = _calculate_immunity(eps, p['ub'], p['db'], re)

    b = p['b0']*(p['b1'] + (1-p['b1'])/(1+(ib/p['IB0'])**p['kb']))

    # calculate clinical immunity IC
    foi = b * eps

    # calculate probability that an asymptomatic infection (state A) will be
    # detected by microscopy
    ic = _calculate_immunity(foi, p['uc'], p['dc'], re)
    id_ = _calculate_immunity(foi, p['ud'], p['dd'], re)
    fd = 1 - (1-p['fd0'])/(1 + (age_days_midpoint/p['ad0'])**p['gd'])
    q = p['d1'] + (1-p['d1'])/(1 + (id_/p['ID0'])**p['kd']*fd)

    # calculate onward infectiousness to mosquitoes
    cA = p['cU'] + (p['cD']-p['cU'])*q**p['g_inf']

    # calculate maternal clinical immunity,
    # assumed to be at birth a proportion of the acquired immunity of a
    # 20 year old
    icm = jnp.append(
        ic[age20] * p['dm'] / age_diff * (
            jnp.exp(-age_days[:-1] / p['dm']) - jnp.exp(-age_days[1:] / p['dm'])
        ),
        0.
    )

    # calculate probability of acquiring clinical disease as a function of
    # different immunity types
    phi = p['phi0']*(p['phi1'] + (1-p['phi1'])/(
        1 + ((ic+icm)/p['IC0'])**p['kc']
    ))

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

    states = jnp.zeros((6, len(age_days_midpoint)), dtype=dtype)
    states = states.at[:,0].set(
        _compute_state(0, 0, 0, 0, 0, 0, betaT,
              betaD, betaP, betaA, betaU, aT, aD, aP, phi, foi, prop, p)
    )

    states = fori_loop(
        1,
        states.shape[1],
        lambda i, a: a.at[:, i].set(
            _next_state(states, i, betaT, betaD, betaP, betaA, betaU,
                aT, aD, aP, phi, foi, prop, p, r)
        ),
        states
    )


    pos_M = states[0] + states[1] + states[3] * q
    inc = (states[5] + states[4] + states[3]) * foi * phi
    return jnp.stack([pos_M, inc])


def _calculate_immunity(foi, rate, delay, re):
    imm = jnp.full(
        len(foi),
        (foi[0]/(foi[0] * rate + 1))/(1/delay + re[0]),
        dtype=dtype
    )
    # TODO: why does this not work?
    # return fori_loop(
        # 1,
        # len(imm),
        # lambda i, a: a.at[i].set(
            # _next_immunity(foi[i], rate, re[i], imm[i-1], delay)
        # ),
        # imm
    # )
    for i in range(len(imm)):
        imm = imm.at[i].set(
            _next_immunity(foi[i], rate, re[i], imm[i-1], delay)
        )
   
    return imm
    

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
