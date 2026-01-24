#!/usr/bin/env python3
"""
Setup script for Star Wars RRT Path Planner - Python Version
"""

import os

from setuptools import find_packages, setup


def read_readme() -> str:
    """Read the README file and return its contents"""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return "Star Wars RRT Path Planner - Python Version"


def read_requirements() -> list[str]:
    """Read requirements.txt and return list of dependencies"""
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, encoding="utf-8") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return []


setup(
    name="star-wars-rrt-planner",
    version="2.0.0",
    description="Enhanced Star Wars RRT Path Planner with real-time 3D rendering",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Star Wars RRT Team",
    author_email="rrt@starwars.com",
    url="https://github.com/your-repo/star-wars-rrt-planner",
    packages=find_packages(),
    py_modules=["star_wars_rrt"],
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "gpu": [
            "cupy>=10.0.0",
        ],
        "gui": [
            "PyQt5>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "star-wars-rrt=star_wars_rrt:main",
        ],
    },
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Games/Entertainment",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "": ["*.stl", "*.obj", "*.mtl"],
    },
)
