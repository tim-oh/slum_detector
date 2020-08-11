# (C) Inhyuk Choi, Mike Gechter, Tim Ohlenburg, Minas Sifakis, Nick Swansson & Nick Tsivanidis 2020
# This file is part of the Indian multi-city slum identification pilot project
# licensed under the terms of the MIT License. See LICENSE for the license details.

"""
    ...
"""
# TODO: Add doctstrings, here and to other files.
# TODO: add a requirements file. The conda env  is small, but `conda env export > environment.yml' yields a long list.

from setuptools import setup, find_packages

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