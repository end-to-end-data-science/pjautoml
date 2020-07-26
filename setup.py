"""Setup for pjautoml package."""
import setuptools

import pjautoml

NAME = "pjautoml"

VERSION = pjautoml.__version__

AUTHOR = ''

AUTHOR_EMAIL = 'edesio@usp.br'

DESCRIPTION = 'AutoML package'

with open('README.md', 'r') as fh:
    LONG_DESCRIPTION = fh.read()

LICENSE = 'GPL3'

URL = ''

DOWNLOAD_URL = ''

CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved :: GNU General Public License v3 ('
               'GPLv3)',
               'Natural Language :: English',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: OS Independent',
               'Programming Language :: Python :: 3.8']

INSTALL_REQUIRES = [
    'numpy', 'scipy', 'pjml'
]

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license=LICENSE,
    url=URL,
    download_url=DOWNLOAD_URL,
    packages=setuptools.find_packages(),
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
)

package_dir = {'': 'pjautoml'}  # For IDEs like Intellij to recognize the package.
