# (C) Inhyuk Choi, Mike Gechter, Tim Ohlenburg, Minas Sifakis, Nick Swansson & Nick Tsivanidis 2020
# This file is part of the Indian multi-city slum identification pilot project
# licensed under the terms of the MIT License. See LICENSE for the license details.

"""
    Set-up script for the slum detection project of .
	Includes entry point.
	Has pyyaml dependency.
	Excludes tests in installation.
"""

from setuptools import setup, find_packages

# Note: the list of conda env packages shown for the project is small, but `conda env export > environment.yml'
# lists what looks like all available Anaconda packages

setup(
    name = "slum_detector",
    version = "0.0.1",
    author = "Tim Ohlenburg",
    author_email = "email@timohlenburg.com",
    description = "Indian multi-city slum recognition pilot project",
    url = "https://github.com/tim-oh/slum_detector",
    license = "MIT",
    install_requires = ['imageio'],
    )