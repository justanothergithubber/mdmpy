"""
mdmpy is an implementation of Marginal Distribution Models (MDM) which can be
applied to discrete choice modelling.
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mdmpy",
    version="0.0.15",
    author="MDM Py Authors",
    author_email="3600019+justanothergithubber@users.noreply.github.com",
    description="A package that implements Marginal Distribution Models (MDMs)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/justanothergithubber/mdmpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
    'pandas',
    'numpy',
    'scipy',
    'pyomo'
    ],
)
