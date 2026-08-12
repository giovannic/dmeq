"""Which array library the solver runs on, resolved at first use.

`dmeq` is written against numpy's spelling of the array API, which `jax.numpy`
mirrors, so every array *function* the solver calls -- `asarray`, `append`,
`diff`, `exp`, `searchsorted`, and the rest -- is the same name in both
libraries and needs no adapter. Five things are not:

    scan        the immunity and demography recursions
    fori_loop   the equilibrium state recursion over age classes
    vmap        the map over heterogeneity quadrature nodes
    set_at      functional assignment, `x.at[i].set(v)`
    is_tracer   whether a value can be inspected at trace time

None of the five is in the array API standard -- control flow and functional
scatter are deliberately outside its scope -- which is why this is a handful of
lines here rather than a dependency. The numpy versions are plain Python loops;
they are not jittable and not differentiable, and are not meant to be. The jax
backend is what a fit runs on. The numpy backend is for reading the model: an
analysis worker that wants a curve, or a candidate immunity model being probed
before it is fitted, neither of which should pay gigabytes of address space for
a backend it will not differentiate through.

**Nothing here imports either library at module scope.** `import dmeq` touches
neither numpy nor jax; the first call into the solver imports whichever backend
is selected, and only that one. That is the property `dmeq` exists to offer the
analysis environment, and `tests/test_backend.py` pins it in a subprocess.

Selection, in order of precedence: the innermost `use_backend` block, then
whatever `set_backend` was last called with, then `$DMEQ_BACKEND`, then `jax`.
The default is jax so that every existing caller is unaffected.

The selection lives in a `ContextVar`, so `use_backend` is safe to nest and does
not leak across asyncio tasks or threads. The resolved backends themselves are
cached per name and shared -- resolving is an import, and doing it twice would
defeat the point.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, NamedTuple

#: environment variable naming the default backend
ENV_VAR = 'DMEQ_BACKEND'

#: what to use when nothing says otherwise
DEFAULT = 'jax'


class Backend(NamedTuple):
    """The five primitives that differ, plus the array namespace itself."""
    name: str
    xp: Any                  # the array namespace: numpy or jax.numpy
    scan: Callable           # (f, init, xs) -> (carry, ys), jax.lax.scan's contract
    fori_loop: Callable      # (lower, upper, f, init) -> val
    vmap: Callable           # (f) -> f mapped over axis 0 of each argument
    set_at: Callable         # (x, idx, value) -> a copy of x with x[idx] = value
    is_tracer: Callable      # (x) -> can this value not be inspected concretely?


def _numpy_backend() -> Backend:
    import numpy as np

    def scan(f, init, xs):
        """`jax.lax.scan` as a Python loop.

        `xs` is either an array or a tuple of arrays sliced together along
        their leading axis, which is as much of the pytree contract as the
        solver uses. An empty `xs` yields an empty `ys`, which is the case a
        single age class hits.
        """
        leaves = xs if isinstance(xs, tuple) else (xs,)
        n = len(leaves[0])
        carry, ys = init, []
        for i in range(n):
            x = tuple(leaf[i] for leaf in leaves) if isinstance(xs, tuple) else xs[i]
            carry, y = f(carry, x)
            ys.append(y)
        if not ys:
            return carry, np.empty((0,), dtype=np.asarray(init).dtype)
        return carry, np.stack(ys)

    def fori_loop(lower, upper, f, init):
        val = init
        for i in range(int(lower), int(upper)):
            val = f(i, val)
        return val

    def vmap(f):
        def mapped(*args):
            n = len(args[0])
            return np.stack([f(*(a[i] for a in args)) for i in range(n)])
        return mapped

    def set_at(x, idx, value):
        # a copy, not a write: jax's `.at[].set()` is functional and callers
        # rely on the original being left alone.
        out = np.array(x, copy=True)
        out[idx] = value
        return out

    return Backend(
        name='numpy',
        xp=np,
        scan=scan,
        fori_loop=fori_loop,
        vmap=vmap,
        set_at=set_at,
        is_tracer=lambda x: False,  # numpy values are always concrete
    )


def _jax_backend() -> Backend:
    import jax
    import jax.numpy as jnp
    from jax.lax import fori_loop, scan

    return Backend(
        name='jax',
        xp=jnp,
        scan=scan,
        fori_loop=fori_loop,
        vmap=jax.vmap,
        set_at=lambda x, idx, value: x.at[idx].set(value),
        is_tracer=lambda x: isinstance(x, jax.core.Tracer),
    )


_BUILDERS = {'numpy': _numpy_backend, 'jax': _jax_backend}

#: resolved backends, by name. Resolving imports a library, so it happens once.
_RESOLVED: dict = {}

#: the selection, innermost first. `None` means "ask the environment".
_SELECTED: ContextVar = ContextVar('dmeq_backend', default=None)


def _resolve(name: str) -> Backend:
    if name not in _BUILDERS:
        raise ValueError(
            f'unknown backend {name!r}; expected one of {sorted(_BUILDERS)}'
        )
    if name not in _RESOLVED:
        _RESOLVED[name] = _BUILDERS[name]()
    return _RESOLVED[name]


def backend() -> Backend:
    """The backend in force, importing it if this is the first call."""
    return _resolve(_SELECTED.get() or os.environ.get(ENV_VAR) or DEFAULT)


def set_backend(name: str) -> None:
    """Select a backend for the rest of this context.

    Intended to be called once, early, by whatever owns the process. A library
    that wants a backend for the duration of one call wants `use_backend`
    instead -- this one does not restore what it replaced.
    """
    _resolve(name)  # fail now, on the caller's line, rather than at first use
    _SELECTED.set(name)


@contextmanager
def use_backend(name: str):
    """Select a backend for the duration of a block, restoring it afterwards.

    What a caller inside a larger process reaches for: a discovery analysis
    worker wanting the demography in numpy has no business changing the backend
    for a fit running elsewhere in the same interpreter.
    """
    _resolve(name)
    token = _SELECTED.set(name)
    try:
        yield
    finally:
        _SELECTED.reset(token)
