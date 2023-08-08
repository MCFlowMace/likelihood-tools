"""

Author: F. Thomas
Date: October 14, 2021

"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

required = [
    "numpy",
    "scipy",
    "adaptive",
    "tqdm",
    "matplotlib",
    "seaborn",
    "holoviews",
    "bokeh",
    "dill>=0.3.7",
    "multiprocess"
]

setuptools.setup(
    name="likelihoodtools",
    version="0.1.0",
    author="Florian Thomas",
    author_email="fthomas@uni-mainz.de",
    description="https://github.com/MCFlowMace/likelihood-tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MCFlowMace/likelihood-tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=required)
