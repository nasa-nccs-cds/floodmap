#!/usr/bin/env python

from setuptools import setup, find_packages
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='floodmap_legacy',
      version='1.0.0',
      description='This project computes surface water extents from lake shapefile boundary maps',
      author='Thomas Maxwell',
      author_email='thomas.maxwell@nasa.gov',
      url='https://github.com/nasa-nccs-cds/floodmap.git',
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

