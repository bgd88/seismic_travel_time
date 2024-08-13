import os
import setuptools


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name="ttpy",
    author="Brent G. Delbridge",
    author_email="delbridge@lanl.gov",
    description=("Calculate 2pt travel times"),
    packages=['ttpy'],
    package_dir={'ttpy': 'src/'}
)
