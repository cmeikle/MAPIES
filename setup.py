#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2024 Barcelona Supercomputing Center - Centro Nacional de
# Supercomputación (BSC-CNS)

# This file is part of MAPIES

# MAPIES is currently an in-house software for processing large amounts of satellite and in-situ data

from setuptools import find_packages
from setuptools import setup

# Could update this using versioneer
version="0.0.4"

setup(
    name="mapies",
    version=version,
    author="Calum Meikle",
    author_email="calum.meikle@bsc.es",
    packages=find_packages(),
    # Include all YAML files inside the 'config' folder
    package_data={"mapies": ["config/*.yaml"]},
    include_package_data=True,
)


