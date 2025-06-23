# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'DAPyr'
copyright = '2025, Maria Nikolaitchik'
author = 'Maria Nikolaitchik'

release = '0.1.0'
version = '0.1.0'

# -- General configuration
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'
master_doc = 'index'

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'


html_theme_options = {
    # 'logo_only': False,
    # 'display_version': True,
    # 'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    # 'vcs_pageview_mode': '',
    # Toc options
    'collapse_navigation': False,
    # 'sticky_navigation': True,
    # 'navigation_depth': 4,
    # 'includehidden': True,
    # 'titles_only': False
}