"""
Sphinx configuration file for qpmr documentation.

This module configures Sphinx documentation generation including:
- Path setup for the package
- Project metadata
- Extensions (autodoc, Napoleon, type hints, LaTeX)
- HTML theme and output settings
- Autodoc and Napoleon options
"""

import os
import sys

# -- Path setup --------------------------------------------------------------

PACKAGE_ROOT = os.path.abspath('../..')
SRC_PATH = os.path.join(PACKAGE_ROOT, 'src')
EXAMPLES_PATH = os.path.join(PACKAGE_ROOT, 'examples')
sys.path.insert(0, SRC_PATH)

# -- Project information -----------------------------------------------------

project = 'qpmr'
author = 'Adam Peichl'
copyright = '2026, Adam Peichl'
release = '0.0.1'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',          # Automatically document docstrings
    'sphinx.ext.napoleon',         # Google / NumPy style docstrings
    "sphinx.ext.autodoc.typehints",
    'sphinx.ext.mathjax',          # Render LaTeX math
    'sphinx.ext.todo',
    'sphinx_gallery.gen_gallery',  # This is for examples
    "sphinxcontrib.bibtex",        # LaTeX like citations
]

templates_path = ['_templates']
exclude_patterns = []

# -- HTML output -------------------------------------------------------------

html_theme = 'pydata_sphinx_theme' # 'alabaster'
html_static_path = [os.path.join(PACKAGE_ROOT, 'docs', '_static')]
html_favicon = os.path.join(html_static_path[0], 'favicon.ico')  # Path to favicon.ico, generated via: https://favicon.io/favicon-generator/
html_logo = os.path.join(html_static_path[0], 'logo.svg')  # Path to logo

# -- LaTeX output ------------------------------------------------------------
latex_engine = "pdflatex"

latex_documents = [
    (
        "index",                # root document
        "tdspy.tex",            # output .tex file
        "TDSpy Documentation",  # document title
        "Adam Peichl",          # author
        "manual",               # document class
    ),
]

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "figure_align": "htbp",
}

# -- Autodoc settings --------------------------------------------------------

autodoc_member_order = 'bysource'  # Order members as in the source code
autodoc_typehints = 'description'  # Show type hints in the description rather than signature

# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Example Gallery settings ------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": [EXAMPLES_PATH],
    'gallery_dirs': ['auto_examples'],
    # --- subgalleries ---
    "nested_sections": True,                # sub-sections for subfolders
    # "subsection_order": "by_folder",        # order subfolders as in file system
    # "within_subsection_order": "by_file",   # order examples inside subfolder
    'filename_pattern': r'.*\.py$', # include all .py files
    # 'within_subsection_order': FileNameSortKey,
    'plot_gallery': True,           # render plots
    # 'backreferences_dir': None,     # optional
    'run_stale_examples': False,     # force re-execution set True
    'download_all_examples': False,  # optional, button for download all .zip
    # 'matplotlib_animations': (True, 'html5'), # (True, 'mp4') - to save .rst size
}

# -- Bibtex settings ---------------------------------------------------------

bibtex_bibfiles = ["refs.bib"]
bibtex_reference_style = "author_year"
bibtex_scan_rst = True
bibtex_overwrite_cache = True