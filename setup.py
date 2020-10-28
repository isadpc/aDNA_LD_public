"""Setup for public repository for aDNA and LD."""


from setuptools import setup

version = "0.0.1"

required = open("requirements.txt").read().split("\n")

setup(
    name="aDNA_LD_public",
    version=version,
    description=" ",
    author="Arjun Biddanda",
    author_email="aabiddanda@gmail.com",
    url="https://github.com/aabiddanda/aDNA_LD_public",
    packages=["aDNA_LD_public"],
    install_requires=required,
    long_description="See " + "https://github.com/aabiddanda/aDNA_LD_public",
    license="GPLv3",
)
