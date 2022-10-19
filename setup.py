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
    "uproot"
]

setuptools.setup(
    name="tsp",
    version="0.2.0",
    author="Florian Thomas",
    author_email="fthomas@uni-mainz.de",
    description="https://github.com/MCFlowMace/thesis-signal-processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MCFlowMace/thesis-signal-processing",
    packages=setuptools.find_packages(),
    #package_data={'cresana': ['hexbug/**/**/*', 'hexbug/**/*', 'settings/*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=required)
