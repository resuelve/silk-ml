# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'silk_ml'
copyright = '2019, Resuelve'
author = 'Miguel Asencio'

# The full version, including alpha/beta/rc tags
release = '0.0.3'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
]

# Include Python objects as they appear in source files
# Default: alphabetically ('alphabetical')
autodoc_member_order = 'bysource'
# Default flags used by autodoc directives
autodoc_default_flags = ['members', 'show-inheritance']
# This value contains a list of modules to be mocked up
autodoc_mock_imports = [
    'pandas',
    'scipy',
    'sklearn',
    'matplotlib',
    'seaborn',
    'imblearn',
]
# Generate autodoc stubs with summaries from code
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
# alabater theme opitons
html_theme_options = {
    'github_button': True,
    'github_type': 'star&v=2',
    'github_user': 'resuelve',
    'github_repo': 'silk-ml',
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.
html_static_path = ['_static']

# Sidebars configuration for alabaster theme

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html',
    ]
}

# I don't like links to page reST sources
html_show_sourcelink = True

# Add Python version number to the default address to corretcly reference
# the Python standard library
intersphinx_mapping = {'https://docs.python.org/3.7': None}
