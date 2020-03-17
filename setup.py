from setuptools import find_packages, setup


def read_md(file):
    with open(file) as fin:
        return fin.read()


setup(
    name="proteinsolver",
    version="0.1.13",
    description="Learn to solve constraint satisfaction problems (CSPs) from data.",
    long_description=read_md("README.md"),
    author="Alexey Strokach",
    author_email="alex.strokach@utoronto.ca",
    url="https://gitlab.com/ostrokach/proteinsolver",
    packages=find_packages(exclude=["tests"]),
    package_data={"proteinsolver": ["data/inputs/*.pdb"]},
    include_package_data=True,
    zip_safe=False,
    keywords="proteinsolver",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
    ],
    test_suite="tests",
)
