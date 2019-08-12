# silk-ml

[![PyPI version][pypi-image]][pypi-url]
![PyPI python Version](https://img.shields.io/pypi/pyversions/silk-ml?style=flat-square)

> Simple Intelligent Learning Kit (SILK) for Machine learning

## About

In the area of ​​machine learning and data science, the most relevant is data management and knowledge. However, there are tasks such as the selection and aggregation of variables that best describe the event to be predicted. These tasks can be repetitive and manual. It has been observed that this part of the creation of a model takes up to 60% of the time of a data scientist.

One of the greatest qualities of a programmer is being lazy, since he thinks about doing a task so that he doesn't have to do it again, so we focus our time on less repetitive or experimental tasks, if not on the tasks of business knowledge and we started a task automation project for Machine learning.

In the automation process, a series of aids for the exploration and sanitation of data were created since it is what we see least developed in the published libraries. Among the tasks we perform, we include descriptive statistics, inferential statistics for binary classification and remediation of variables by type of data and their content.

## Usage
You can install it from [pip](https://pypi.org/project/silk-ml/) as
```bash
pip install silk-ml
```

If you want to have a very precise idea of the package, please read our [documentation](https://resuelve.github.io/silk-ml/):
- [`classification`](https://resuelve.github.io/silk-ml/_autosummary/classification.html)

## Contributing
Thank you, your help and ideas are very welcome! Please be sure to read the contributing guidelines and to respect the license.
- [Contributing guidelines](./CONTRIBUTING.md)
- [MIT License](./LICENSE)

There are also some useful `make` commands to have in mind:
- `test`: Runs the unit tests
- `publish`: Runs all the publish commands after the tests just passed
- `publish.docs`: Builds the HTML documentation from the Sphinx documentation
- `publish.package`: Builds the binary files to publish
- `publish.pypi`: Sends the binary files to pypi

[pypi-image]: https://img.shields.io/pypi/v/silk-ml?style=flat-square
[pypi-url]: https://pypi.org/project/silk-ml/
