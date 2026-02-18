# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
# Ensure the project's src/ folder is on sys.path so Sphinx can import the package
sys.path.insert(0, os.path.abspath("../src"))


project = 'Compocyte'
copyright = '2026, Christopher'
author = 'Christopher'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",     # Google/NumPy style docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

# Support Markdown pages in the documentation
extensions += [
    "myst_parser",
    "sphinx.ext.intersphinx",
]

templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True  # Automatically generate summary files for documented modules/classes/functions
autodoc_mock_imports = [
    "scanpy",
    "catboost",
    "torch",
    "sklearn",
    "scipy",
    "numpy",
    "pandas",
    "networkx",
    "matplotlib",
    "dask",
    "keras",
    "balanced_loss",
]

# Allow Markdown files and configure myst-parser
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Intersphinx: link to common project docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = "_static/Compocyte.png"

# Prefer the ReadTheDocs theme when available, otherwise fall back to a builtin theme
try:
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    html_theme_options = {
        "collapse_navigation": False,
        "navigation_depth": 4,
        "sticky_navigation": True,
        "logo_only": False,
    }
except Exception:
    html_theme = 'alabaster'
    html_theme_options = {}
html_static_path = ['_static']
