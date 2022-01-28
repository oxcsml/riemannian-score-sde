#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_namespace_packages, setup

with open("requirements.txt", "r") as file:
    requirements = [line.strip() for line in file]

with open("README.md", "r") as file:
    long_description = file.read()

with open("VERSION", "r") as file:
    version = file.read().strip()

setup(
    name="score-sde",
    version=version,
    author="Michael Hutchinson, Emile Mathieu",
    author_email="michael.hutchinson@stats.ox.ac.uk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="",
    # license="Apache License 2.0",
    keywords="",
    # install_requires=requirements,
    packages=find_namespace_packages(
        include=["score_sde", "riemannian_score-sde", "stochastic-process-score-sde"]
    ),
    classifiers=[
        # "License :: OSI Approved :: Apache Software License",
        # "Programming Language :: Python :: 3.6",
        # "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
