#!/usr/bin/env python3
"""
Setup script for Solar System Simulation package.
"""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="solar-system-simulation",
    version="1.0.0",
    author="Solar System Simulation Project",
    description="A scientifically accurate 3D solar system simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/solar-system-simulation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pygame>=2.0.0",
        "PyOpenGL>=3.1.5",
    ],
    extras_require={
        "accelerate": ["PyOpenGL_accelerate>=3.1.5"],
        "dev": ["pytest>=6.0", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "solar-system=solar_system.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "solar_system": ["assets/textures/*", "assets/shaders/*"],
    },
)
