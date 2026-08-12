from setuptools import setup, find_packages

setup(
    name='dmeq',
    version='0.3.0',
    url='https://github.com/giovannic/dmeq.git',
    author='Giovanni Charles',
    author_email='gc1610@ic.ac.uk',
    description='Differentiable equilibrium solution for malaria',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # numpy is a backend in its own right now, not just a test dependency; jax
    # stays required so that every existing caller is unaffected by the split.
    # A numpy-only install is possible -- nothing imports jax unless it is the
    # selected backend -- and would drop jax/jaxlib to an extra.
    install_requires=['numpy', 'jaxlib >= 0.4.1', 'jax >= 0.4.1'],
    extras_require={'test': ['pytest']}
)
