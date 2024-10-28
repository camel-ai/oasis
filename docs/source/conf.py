# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sphinx_rtd_theme

import social_simulation

project = "social_simulation"
copyright = "2024, TODO"
author = "TODO"
version = social_simulation.__version__
release = social_simulation.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
intersphinx_mapping = {"python": ("https://docs.python.org/", None)}
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
}


def setup(app):

    def skip(app, what, name, obj, skip, options):
        members = [
            "__init__",
            "__repr__",
            "__weakref__",
            "__dict__",
            "__module__",
        ]
        return True if name in members else skip

    app.connect("autodoc-skip-member", skip)
