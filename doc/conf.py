# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.append(os.path.abspath("../src/"))
sys.path.append(os.path.abspath("../test/"))
sys.path.append(os.path.abspath("../test_integ/"))


# -- Project information -----------------------------------------------------

project = "ProductionDemo"
copyright = "2022, Brandon Shurick"
author = "Brandon Shurick"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "m2r2",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
source_suffix = ['.rst', '.md']
master_doc = "index"
m2r_parse_relative_links = True

# autodoc
autoclass_content = "class"
autodoc_member_order = "bysource"
default_role = "py:obj"

# Theme 
html_theme = "alabaster"
html_static_path = []
html_theme_options = {
    'description': 'Production Data Science demo',
    'github_user': 'bshurick',
    'github_repo': 'production-demo',
    'github_banner': True,
    'github_type': 'mark',
    'github_count': False,
    'font_family': '"Charis SIL", "Noto Serif", serif',
    'head_font_family': 'Lato, sans-serif',
    'code_font_family': '"Code new roman", "Ubuntu Mono", monospace',
    'code_font_size': '1rem',
}
