from setuptools import setup, find_packages

setup(
    name='dmeq',
    version='0.0.1',
    url='https://github.com/giovannic/dmeq.git',
    author='Giovanni Charles',
    author_email='gc1610@ic.ac.uk',
    description='Differentiable equilibrium solution for malaria',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=['jaxlib >= 0.4.1', 'jax >= 0.4.1']
)
