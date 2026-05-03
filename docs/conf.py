"""Sphinx configuration for the icu_scheduler docs."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "ICU Scheduler"
author = "Your Name"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",      # Google / NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []
