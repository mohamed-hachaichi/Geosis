from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Geographic knowledge production analysis'
LONG_DESCRIPTION = 'A machine learining-based package to preform spatial data analysis on large-scale textual data.'

# Setting up
setup(
    name="Geosis",
    version=VERSION,
    author="Mohamed HACHACHI",
    author_email="<hachaichi_mohamed@outlook.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['pandas', 'matplotlib', 'geopandas', 'Networkx'],
    keywords=['python', 'geography', 'analysis', 'knowledge production', 'camera stream', 'sockets'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ])