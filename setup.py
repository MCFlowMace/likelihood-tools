"""

Author: F. Thomas
Date: October 14, 2021

"""

import setuptools
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    required = [line.strip() for line in fh]

setuptools.setup(
    name="likelihoodtools",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
