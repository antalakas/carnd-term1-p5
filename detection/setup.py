#!/usr/bin/env python

# -*- coding: utf-8 -*-

#///////////////////////////////////////////////////////////
#-----------------------------------------------------------
# File: setup.py
# Author: Andreas Ntalakas
#-----------------------------------------------------------

#///////////////////////////////////////////////////////////
# Import a setup module
from setuptools import setup, find_packages

#///////////////////////////////////////////////////////////
# Import package
import detection

#///////////////////////////////////////////////////////////
# Basic setup
setup(

    # Package Details
    name='detection',
    version=detection.__version__,
    author=detection.__author__,
    author_email=detection.__email__,
    url=detection.__url__,
    packages=find_packages(exclude=['tests','docs'],),
    include_package_data=True,

    # Disable for allowing copying files
    zip_safe=False,

    # Package Description
    description='Vehicle Detection, Project 5, Term 1 CarND Nanodegree Udacity.',
    long_description=detection.__description__,
    license='',

    # Dependent Packages (if any)
    requires=['Sphinx',],

    install_requires=[
        "Sphinx>=1.3.1",
    ],
)
