# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# import sphinx_rtd_theme


"""
From the top-level run
```
sphinx-apidoc -f -o docs/source .
```
then go to docs and run
```
make html
```
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

project = 'Paicos'
copyright = '2024, Thomas Berlok'
author = 'Thomas Berlok'
release = '0.1.9'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_rtd_theme', "nbsphinx", "myst_parser",
              "sphinx.ext.napoleon", "sphinx_copybutton",
              "sphinx_rtd_dark_mode"]

default_dark_mode = False

nbsphinx_execute = 'never'

templates_path = ['_templates']
exclude_patterns = []


source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False
}
