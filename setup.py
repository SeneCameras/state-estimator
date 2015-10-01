#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'cv2'],
    test_suite='tests',
)
