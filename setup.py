import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="graphgen",
    version="0.0.7",
    author="Bhavesh Kalisetti",
    author_email="bhaveshk658@berkeley.edu",
    description="A package to generate graphs from GPS data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bhaveshk658/graphgen",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)