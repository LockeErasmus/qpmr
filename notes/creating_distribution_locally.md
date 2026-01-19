# This document is for step-by-step local distribution build

First, make sure `build` and `twine` is installed:
```bash
python -m pip install build twine
```

Next, build via:
```bash
python -m build
```

Version is determined automatically via property `__version__` (defined in `__init__.py` in `qpmr` package).

Next, check via twine:
```bash
twine check dist/*
```

Upload package to pipy:
```bash
twine upload dist/*
```

Do not forget to input correct API token.


## Local install test

Locally, create new virtual environment and

```
pip install .path/to/qpmr/package/qpmr-0.0.1.tar.gz
```