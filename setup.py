from setuptools import setup, find_packages

setup(
    name="Realm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[requirement for requirement in open("requirements.txt").readlines()],
)
