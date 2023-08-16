from setuptools import setup, find_packages

setup(
    name="rpo",
    version="0.0",
    install_requires=['torch',
                      'numpy',
                      'scipy',
                      'scikit-learn',
                      'pandas',
                      'matplotlib'],
    author="dingsht",
    description="official implementation of reduced policy optimization",
    packages=['rpo']
)