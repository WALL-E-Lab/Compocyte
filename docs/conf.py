"""Configuration file for the Sphinx documentation builder."""

import os
import sys

# Add source path for autodoc
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information ---
project = 'Compocyte'
copyright = '2025, Christopher Beltz, Leon Sadowski, Thomas Walle'
author = 'Christopher Beltz, Leon Sadowski, Thomas Walle'
release = '0.1.0b1'

# -- General configuration ---
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',           # Auto-generate API docs
    'sphinx.ext.napoleon',          # Support for NumPy/Google docstring styles
    'sphinx.ext.viewcode',          # Link to source code
    'sphinx.ext.intersphinx',       # Link to other projects (e.g., numpy, scipy)
    'sphinx.ext.autosummary',       # Generate summary tables for modules/classes/functions
    'nbsphinx',                     # Jupyter notebook support
    'sphinx_design',                # Cards/grids for better layout
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Autodoc configuration ---
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'show-inheritance': True,
}

# -- Napoleon configuration (for docstring parsing) ---
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# -- nbsphinx configuration ---
nbsphinx_execute = 'never'  # Don't execute notebooks on build (can set to 'always' later)
nbsphinx_kernel_name = 'python3'

# -- Intersphinx mapping ---
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scanpy': ('https://scanpy.readthedocs.io/en/stable/', None),
    'anndata': ('https://anndata.readthedocs.io/en/stable/', None),
}

# -- HTML output configuration ---
html_theme = 'furo'  # Modern, responsive theme

html_theme_options = {
    'light_logo': "../Compocyte.png",
}

html_static_path = ['_static']
html_title = 'Compocyte Documentation'
html_logo = "../Compocyte.png"  # Set to a logo if you have one
html_favicon = "../Compocyte.png"

# -- Additional HTML context ---
html_context = {
    'github_user': 'WALL-E-Lab',
    'github_repo': 'Compocyte',
    'github_version': 'main',
    'doc_path': 'docs',
}
