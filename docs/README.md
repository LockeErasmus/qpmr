# Documentation

Assuming we are at root, make sure all `sphinx` dependencies are installed, i.e.

```
python -m pip install -r ./docs/requirements.txt
```

navigate to `docs` folder

```
cd docs
```

## Building HTML documentation

```
sphinx-build -b html source _build/html
```

### Starting from scratclearch (**DO NOT USE** - it will rewrite `.rst` files)

```
sphinx-quickstart
```

Genereate `.rst` files for modules

```
sphinx-apidoc -o ./source/ ../src/qpmr/
```

## Building PDF documentation

Make sure LaTex is installed

```bash
sphinx-build -b latex source _build/latex
```

Then depending on platform

```bash
make -C _build/latex
```

### Windows specific notes

You are using windows, that is already bad. Make sure LaTex is installed, you can check using

```bash
pdflatex --version
```

and run

```bash
cd _build\latex
pdflatex <name>.tex
```
