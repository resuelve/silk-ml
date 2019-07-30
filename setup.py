import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="silk-ml",
    version="0.0.1",
    author="Miguel Asencio",
    author_email="maasencioh@gmail.com",
    description="Simple Intelligent Learning Kit (SILK) for Machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/resuelve/silk-ml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
