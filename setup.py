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
    description="reduced policy optimization",
    packages=['rpo']
)