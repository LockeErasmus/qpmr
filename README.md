# QPmR - Quasi-Polynomial Mapping Based Rootfinder

<p align="center">
    <a href="https://github.com/LockeErasmus/qpmr/actions">
        <img alt="CI" src="https://github.com/LockeErasmus/qpmr/workflows/CI/badge.svg?event=push&branch=master">
    </a>
    <a href="https://pypi.org/project/qpmr/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/qpmr">
    <!-- </a>
    <a href="https://qpmr.readthedocs.io/en/latest/?badge=latest">
        <img src="https://readthedocs.org/projects/qpmr/badge/?version=latest" alt="Documentation Status" />
    </a> -->
    <a href="https://github.com/LockeErasmus/qpmr/blob/master/LICENSE">
        <img alt="License" src=https://img.shields.io/badge/license-%20%20GNU%20GPLv3%20-green?style=plastic>
    </a>
    <br/>
</p>

This package is python implementation of `QPmR v2` algorithm described in [[1]](#1). 

The `QPmR v2` algorithm is designed to find all roots of the quassi-polynomial
$$
h(s) = \sum\limits_{j=0}^{n} p_{j}(s) e^{-s\alpha_j}
$$
located in the complex plane region $\mathbb{D} \in \mathbb{C}$, with the 
boundaries $\beta_{min} \leq \Re(\mathbb{D}) \leq \beta_{max}$ and 
$\omega_{min} \leq \Im(\mathbb{D}) \leq \omega_{max}$, where $\alpha_{0}=0 < \alpha_{1} < \dots < \alpha_n$ are the delays and $p_j(s) = \sum\limits_{k=0}^{d}c_{j,k} s^{k}$ where $d = \max\{degree(p_j(s))|j=0,\dots, n\}$.

The following table may help undertand how is quasi-polynomial representation:

$$
\begin{array}{c|cccc|c}
 \alpha_{j} & s^{0} & s^{1} & \dots & s^{d} & p_j(s)e^{-s\alpha_j} \\ \hline
\alpha_{0} & c_{0,0} & c_{0,1} & \dots & c_{0,d} &  \sum\limits_{k=0}^{d}c_{0,k} s^{k} e^{-s\alpha_0} \\
\alpha_{1} & c_{1,0} & c_{1,1} & \dots & c_{1,d} &  \sum\limits_{k=0}^{d}c_{1,k} s^{k} e^{-s\alpha_1} \\
\vdots & \vdots & \vdots & \ddots & \vdots &  \vdots \\
\alpha_{n} & c_{n,0} & c_{n,1} & \dots & c_{n,d} &  \sum\limits_{k=0}^{d}c_{n,k} s^{k} e^{-s\alpha_n} \\
\end{array}
$$

<a id="1">[1]</a>
Vyhlidal, T. and Zitek, P. (2014). 
QPmR-Quasi-polynomial root-finder: Algorithm update and examples
Editors: Vyhídal T., Lafay J.F., Sipahi R., Sringer 2014.

### Examples

We will follow 3 examples from the 2014 paper.

### Example 1

Find all the roots of the quassi-polynomial
$$
h(s) = s + e^{-s}
$$
located in region $\mathbb{D} = \{\beta+j\omega| -10 \leq \beta\leq 2;~0 \leq \omega \leq 30\}$.
$$
\begin{array}{c|cc|c}
\alpha_{j} & s^{0} & s^{1} & p_j(s)e^{-s\alpha_j} \\ \hline
0 & 0 & 1 & s \\
1 & 1 & 0 & e^{-s} \\
\end{array}
$$

```python
delays = np.array([0., 1])
coefs = np.array([[0., 1], [1, 0]])
region = [-10, 2, 0, 30]
```

## Citing this work

Cite the following work
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



## Installation

### Installing with `pip`

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

install from source
```bash
pip install -e qpmr
```

## Usage

TODO <- move examples and additional comments to this section.

## License

This package is available under [GNU GPLv3 license](./LICENSE).
