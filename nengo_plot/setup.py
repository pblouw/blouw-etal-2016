#!/usr/bin/env python
import imp
import sys
import os

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import find_packages, setup

root = os.path.dirname(os.path.realpath(__file__))
description = "Collection of plotting tools for nengo"
with open(os.path.join(root, 'README.md')) as readme:
    long_description = readme.read()

setup(
    name="nengo_plot",
    version=0.1,
    author="CNRGlab at UWaterloo",
    author_email="tcstewar@uwaterloo.ca",
    packages=find_packages(),
    scripts=[],
    url="https://github.com/tcstewar/nengo_plot",
    description=description,
    long_description=long_description,
    requires=[
        "nengo",
        "matplotlib",
    ],
)
