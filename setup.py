from setuptools import setup, find_packages

setup(name='modelling',
    version='0.0.1',
    description='modelling package',
    author='rruizendaal',
    url='https://github.com/ruizendaalr/modelling',
    packages=['train'],
    install_requires=[
        'numpy',
        'pandas'
    ]
)