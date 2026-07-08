# QPmR Python Package

<p align="center">
    <!-- <a href="https://github.com/LockeErasmus/qpmr/actions">
        <img alt="CI" src="https://github.com/LockeErasmus/qpmr/workflows/CI/badge.svg?event=push&branch=master">
    </a> -->
    <a href="https://pypi.org/project/qpmr/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/qpmr">
    </a>
    <!-- <a href="https://qpmr.readthedocs.io/en/latest/?badge=latest">
        <img src="https://readthedocs.org/projects/qpmr/badge/?version=latest" alt="Documentation Status" />
    </a> -->
    <a href="https://github.com/LockeErasmus/qpmr/blob/master/LICENSE">
        <img alt="License" src=https://img.shields.io/badge/license-%20%20GNU%20GPLv3%20-green?style=plastic>
    </a>
</p>

This Python package consists of: 

1. enhanced implementation of *quasi-polynomial based rootfinder*, algorithm for finding roots of given quasi polynomial in given rectangular region in complex plane [[1]](#1), [[2]](#2) enhanced with root-multiplicity heuristic and (in some cases) automatic region selection
1. Spectral distribution properties of quasi-polynomials: (i) delay distribution diagram, (ii) exponential asymptotes of root chains, (iii) safe upper bound of neutral spectrum and (iv) spectrum envelope
1. quasi-polynomial (and transfer function) algebra
1. various functions for quickly visualizing results

For original MATLAB® implementation of old QPmR algorithm from 2012, we refer to [this page](https://control.fs.cvut.cz/en/qpmr/).

<a id="1">[1]</a>
Vyhlidal, T., and Zítek, P. (2009). Mapping based algorithm for large-scale computation of quasi-polynomial zeros. IEEE Transactions on Automatic Control, 54(1), 171-177.

<a id="2">[2]</a>
Vyhlidal, T. and Zitek, P. (2014). QPmR-Quasi-polynomial root-finder: Algorithm update and examples
Editors: Vyhídal T., Lafay J.F., Sipahi R., Sringer 2014.

Please, keep in mind that thisproject is still **under development**.

## Installation

### Installing with `pip`

Using pipy (check the version as software is being developed)
```bash
pip install qpmr
```

From github
```bash
pip install qpmr@git+https://github.com/LockeErasmus/qpmr
```

Local install
```bash
pip install <path-to-qpmr>
```

### Installing from source

Clone repository
```bash
git clone https://github.com/LockeErasmus/qpmr.git
```

install from source with `-e`
```bash
pip install -e qpmr
```
## Citing this work

For now, please cite the following article
```
@article{vyhlidal2009mapping,
  title={Mapping based algorithm for large-scale computation of quasi-polynomial zeros},
  author={Vyhlidal, Tomas and Z{\'\i}tek, Pavel},
  journal={IEEE Transactions on Automatic Control},
  volume={54},
  number={1},
  pages={171--177},
  year={2009},
  publisher={IEEE}
}
```

Or alternatively, this chapter of book
```
@incollection{vyhlidal2014qpmr,
  title={QPmR-Quasi-polynomial root-finder: Algorithm update and examples},
  author={Vyhl{\'\i}dal, Tom{\'a}{\v{s}} and Z{\'\i}tek, Pavel},
  booktitle={Delay Systems: From Theory to Numerics and Applications},
  pages={299--312},
  year={2014},
  publisher={Springer}
}
```

## Metadata

```
Title: QPmR Python Package
ID: QPmRpythonPackage
Version: 1.0
Project: Robotics and Advanced Industrial Production
Project No.: CZ.02.01.01/00/22_008/0004590
Project RO: 1.1-Optimal control of interconnected time-delay systems
Date: 20.9.2025
Authors: Adam Peichl; Tomas Vyhlidal
Keywords: rootfinding algorithm, exponential polynomial, quasi-polynomial, time delay system, spectrum
```

## Acknowledgments

This work was supported by the European Union under the project Robotics and Advanced Industrial Production, reg. no. CZ.02.01.01/00/22_008/0004590; by the project 24-10301S of the Czech Science Foundation; and by by the Grant Agency of the Czech Technical University in Prague, grant No. SGS24/125/0HK2/3T/12.

## License

This package is available under [GNU GPLv3 license](./LICENSE).
